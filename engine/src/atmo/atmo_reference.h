// CPU reference for the atmosphere, in double precision. It runs the same integrals as the
// GPU tables (transmittance with 40 steps, multiple-scattering table with 64 directions and 20
// steps, sky-view with 16 view steps and exact sun transmittance), so differences measure the
// tables and their lookup, not the model. The model: Rayleigh plus Mie (Preetham turbidity 3),
// single scattering plus an isotropic multiple-scattering term, flat ground albedo 0.2 for rays
// into the planet. Radiance in cd/m^2 before exposure.
#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

namespace engine {
namespace atmo {

constexpr double kPi = 3.14159265358979323846;
constexpr double kRg = 6360000.0;
constexpr double kRt = 6460000.0;
constexpr double kBetaR[3] = {5.8e-6, 13.5e-6, 33.1e-6};
constexpr double kHR = 8000.0;
constexpr double kHM = 1200.0;
constexpr double kMieG = 0.76;
constexpr double kSolarLux = 128000.0;
constexpr int kTransSteps = 40;
constexpr int kViewSteps = 16;
constexpr int kMsW = 32;      // multiple-scattering table, same size as MS_W / MS_H in common.glsl
constexpr int kMsH = 32;
constexpr int kMsDirs = 64;
constexpr int kMsSteps = 20;

// Coastal haze: scales the Mie coefficient (Preetham turbidity 3 at 1.0; 3.0 for the violet sunset bay). Shared with the GPU
// shaders (HAZE_SCALE in shaders/common.glsl); validation fails if the two differ.
constexpr double kHazeScale = 1.5;
inline double betaM(int c) {
    const double lam[3] = {0.68, 0.55, 0.44};
    return kHazeScale * (0.04608 * 3.0 - 0.04586) * std::pow(lam[c], -1.3) / kHM;
}

struct V3 {
    double x, y, z;
};
inline double dotv(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

inline double sphereRoot(double r0, double mu, double R, bool farRoot) {
    const double b = r0 * mu;
    const double disc = b * b - (r0 * r0 - R * R);
    if (disc < 0.0) return -1.0;
    const double s = std::sqrt(disc);
    return farRoot ? (-b + s) : (-b - s);
}

inline bool hitsGround(double r0, double mu) {
    if (mu >= 0.0) return false;
    const double b = r0 * mu;
    return b * b - (r0 * r0 - kRg * kRg) >= 0.0;
}

// Transmittance from radius r to the top of the atmosphere along cos zenith mu.
inline void transmittance(double r, double mu, double out[3]) {
    if (hitsGround(r, mu)) {
        out[0] = out[1] = out[2] = 0.0;
        return;
    }
    const double tt = sphereRoot(r, mu, kRt, true);
    double tau[3] = {0, 0, 0};
    double tPrev = 0.0;
    for (int i = 1; i <= kTransSteps; i++) {
        const double f = static_cast<double>(i) / kTransSteps;
        const double t1 = tt * f * f;
        const double tm = 0.5 * (tPrev + t1);
        const double rm = std::sqrt(std::max(0.0, r * r + 2.0 * r * mu * tm + tm * tm));
        const double h = rm - kRg;
        for (int c = 0; c < 3; c++) tau[c] += (kBetaR[c] * std::exp(-h / kHR) + betaM(c) * std::exp(-h / kHM)) * (t1 - tPrev);
        tPrev = t1;
    }
    for (int c = 0; c < 3; c++) out[c] = std::exp(-tau[c]);
}

// Mean multiple-scattering radiance table, row-major RGB, texel (x, y) at mu = 2x/(W-1) - 1 and
// r = RG + (RT - RG) (y/(H-1))^2. Same integrals as ms.frag: for each of 64 near-uniform
// directions, the once-scattered sun light L1 and the scattered fraction fsum; the result is
// (L1 / (4 pi N)) / (1 - fms), with fms clamped to 0.95.
inline std::vector<double> multiScatterTable() {
    std::vector<double> tab(static_cast<size_t>(kMsW) * kMsH * 3, 0.0);
    for (int y = 0; y < kMsH; y++)
        for (int x = 0; x < kMsW; x++) {
            const double mu = 2.0 * x / (kMsW - 1) - 1.0;
            const double v = static_cast<double>(y) / (kMsH - 1);
            const double r0 = kRg + (kRt - kRg) * v * v;
            const V3 sunL{std::sqrt(std::max(0.0, 1.0 - mu * mu)), mu, 0.0};
            double L1[3] = {0, 0, 0}, fsum[3] = {0, 0, 0};
            for (int d = 0; d < kMsDirs; d++) {
                const double z = 1.0 - 2.0 * (d + 0.5) / kMsDirs;
                const double phi = d * 2.39996323;
                const double rxy = std::sqrt(std::max(0.0, 1.0 - z * z));
                const V3 w{rxy * std::cos(phi), z, rxy * std::sin(phi)};
                const double tMax = hitsGround(r0, w.y) ? sphereRoot(r0, w.y, kRg, false) : sphereRoot(r0, w.y, kRt, true);
                if (tMax <= 0.0) continue;
                double Tv[3] = {1, 1, 1};
                double tPrev = 0.0;
                for (int i = 1; i <= kMsSteps; i++) {
                    const double t1 = tMax * i / kMsSteps;
                    const double ds = t1 - tPrev;
                    const double tm = 0.5 * (t1 + tPrev);
                    const double r = std::sqrt(std::max(0.0, r0 * r0 + 2.0 * r0 * w.y * tm + tm * tm));
                    const double h = r - kRg;
                    const double muS = (r0 * mu + tm * dotv(w, sunL)) / r;
                    double Ts[3];
                    transmittance(r, muS, Ts);
                    for (int c = 0; c < 3; c++) {
                        const double sigE = kBetaR[c] * std::exp(-h / kHR) + betaM(c) * std::exp(-h / kHM);
                        const double stepT = std::exp(-sigE * ds);
                        const double sc = 1.0 - stepT;
                        L1[c] += Tv[c] * sc * kSolarLux * Ts[c];
                        fsum[c] += Tv[c] * sc;
                        Tv[c] *= stepT;
                    }
                    tPrev = t1;
                }
            }
            for (int c = 0; c < 3; c++) {
                const double lmean = L1[c] / (4.0 * kPi * kMsDirs);
                const double fms = std::clamp(fsum[c] / kMsDirs, 0.0, 0.95);
                tab[(static_cast<size_t>(y) * kMsW + x) * 3 + c] = lmean / (1.0 - fms);
            }
        }
    return tab;
}

// Multiple-scattering radiance at radius r and cos zenith mu from the table (bilinear, the same
// lookup as msLookup in common.glsl).
inline double multiScatterLookup(const std::vector<double>& tab, int c, double r, double mu) {
    const double x01 = mu * 0.5 + 0.5;
    const double y01 = std::sqrt(std::max(0.0, (r - kRg) / (kRt - kRg)));
    const double tx = std::clamp(x01 * (kMsW - 1), 0.0, kMsW - 1.0);
    const double ty = std::clamp(y01 * (kMsH - 1), 0.0, kMsH - 1.0);
    const int x0 = std::min(static_cast<int>(tx), kMsW - 2), y0 = std::min(static_cast<int>(ty), kMsH - 2);
    const double fx = tx - x0, fy = ty - y0;
    auto at = [&](int x, int y) { return tab[(static_cast<size_t>(y) * kMsW + x) * 3 + c]; };
    return (1 - fy) * ((1 - fx) * at(x0, y0) + fx * at(x0 + 1, y0)) +
           fy * ((1 - fx) * at(x0, y0 + 1) + fx * at(x0 + 1, y0 + 1));
}

// Single scattering plus isotropic multiple scattering along dir from radius r0 for distance tMax.
inline void scatterAlong(double r0, V3 dir, double tMax, V3 sun, const std::vector<double>& ms, double L[3],
                         double Tv[3]) {
    const double cosT = dotv(dir, sun);
    const double phR = 3.0 / (16.0 * kPi) * (1.0 + cosT * cosT);
    const double g2 = kMieG * kMieG;
    const double phM = (1.0 - g2) / (4.0 * kPi * std::pow(1.0 + g2 - 2.0 * kMieG * cosT, 1.5));
    for (int c = 0; c < 3; c++) { L[c] = 0.0; Tv[c] = 1.0; }
    double tPrev = 0.0;
    for (int i = 1; i <= kViewSteps; i++) {
        const double f = static_cast<double>(i) / kViewSteps;
        const double t1 = tMax * f * f;
        const double ds = t1 - tPrev;
        const double tm = 0.5 * (t1 + tPrev);
        const double r = std::sqrt(std::max(0.0, r0 * r0 + 2.0 * r0 * dir.y * tm + tm * tm));
        const double h = r - kRg;
        const double dR = std::exp(-h / kHR), dM = std::exp(-h / kHM);
        const double muSun = (r0 * sun.y + tm * cosT) / r;
        double Ts[3];
        transmittance(r, muSun, Ts);
        for (int c = 0; c < 3; c++) {
            const double sigE = kBetaR[c] * dR + betaM(c) * dM;
            const double sigS = kBetaR[c] * dR * phR + betaM(c) * dM * phM;
            const double sigIso = kBetaR[c] * dR + betaM(c) * dM;
            const double stepT = std::exp(-sigE * ds);
            const double src = sigS * kSolarLux * Ts[c] + sigIso * multiScatterLookup(ms, c, r, muSun);
            L[c] += Tv[c] * src * (1.0 - stepT) / sigE;
            Tv[c] *= stepT;
        }
        tPrev = t1;
    }
}

// Sky radiance seen from radius r0 along dir (the same branch structure as the GPU).
inline void skyRadiance(double r0, V3 dir, V3 sun, const std::vector<double>& ms, double out[3]) {
    double Tv[3];
    if (hitsGround(r0, dir.y)) {
        const double tG = sphereRoot(r0, dir.y, kRg, false);
        scatterAlong(r0, dir, tG, sun, ms, out, Tv);
        double Tsun[3];
        transmittance(r0, sun.y, Tsun);
        const double horiz = std::max(0.0, sun.y);
        for (int c = 0; c < 3; c++) out[c] += Tv[c] * (0.2 / kPi) * kSolarLux * Tsun[c] * horiz;
        return;
    }
    const double tTop = sphereRoot(r0, dir.y, kRt, true);
    scatterAlong(r0, dir, tTop, sun, ms, out, Tv);
}

}  // namespace atmo
}  // namespace engine
