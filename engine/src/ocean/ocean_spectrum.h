// Tessendorf's FFT ocean: initial spectrum (Tessendorf, "Simulating Ocean Water", SIGGRAPH 2001 course notes).
//   P_h(k) = A exp(-1/(kL)^2) / k^4 |k^.w^|^2 exp(-k^2 l^2),   L = V^2 / g
//   h0(k)  = (1/sqrt 2) (xi_r + i xi_i) sqrt(P_h(k)),            xi ~ N(0, 1)
// GPU (shaders/ocean_evolve.comp, shaders/fft2d_stage.comp):
//   h(k,t) = h0(k) e^{iwt} + conj(h0(-k)) e^{-iwt},   w = sqrt(gk)
//   D(x,t) = sum_k (-i k/|k|) h(k,t) e^{ikx}          (horizontal choppy displacement)
// Header-only. A grid is interleaved (re, im) in natural FFT order: index z * N + x. The wavenumber of index n
// is 2 pi s / L with s = n for n < N/2 and s = n - N otherwise.
#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

namespace engine {
namespace ocean {

// Grid and patch of the renderer. shaders/ocean.vert repeats kOceanN and kOceanL: keep them in sync.
constexpr uint32_t kOceanLog2N = 8;
constexpr uint32_t kOceanN = 1u << kOceanLog2N;  // 256 points: 4 m cells, so waves of 8 m and longer are held
constexpr float kOceanL = 1024.0f;               // patch period (m): the longest wave the grid holds
constexpr double kOceanWindMs = 10.0;            // wind speed V (m/s)
constexpr double kOceanSigmaM = 0.30;            // RMS height (m): calm-to-light chop, so reflections stay clear
// Capillary cascade: a second, small patch that holds the waves the main grid cannot. It covers |k| above the
// main grid's corner wavenumber (sqrt 2 pi N / L, so the two never count the same wave), down to about 0.5 m.
constexpr float kOceanCapL = 96.0f;              // patch period (m): cells of 0.375 m, waves of 0.75 m and shorter
constexpr float kOceanCapGain = 4.0f;            // amplitude gain on the Phillips spectrum: slope RMS about 0.02

struct OceanParams {
    uint32_t N = kOceanN;
    float L = kOceanL;
    float windSpeed = static_cast<float>(kOceanWindMs);
    float windX = 1.0f, windZ = 0.0f;  // wind direction (normalised inside)
    float A = 0.0f;                    // Phillips amplitude, see phillipsAmplitude()
    float g = 9.81f;
    float smallWaveL = 0.001f;         // l: suppresses waves shorter than about a millimetre
    float kMin = 0.0f;                 // waves with |k| below this (rad/m) are dropped: a coarser cascade holds them
    float gain = 1.0f;                 // multiplies A
};

// Per-mode amplitude A that gives an RMS height sigma (m) on an N-point grid of period L at wind V.
// The grid variance is sum_k 2 P_h(k) = (L/2pi)^2 pi A (V^2/g)^2 (1 - eps), where eps = 1/(kc Lw)^2 is the share
// of the spectrum above the grid's Nyquist wavenumber kc = pi N / L (about 1.6 % at N = 256, L = 1024, V = 10).
// Dropping eps:  A = 4 pi sigma^2 / (L^2 (V^2/g)^2).
// Check: a fully developed sea at V = 10 m/s has H_1/3 ~ 0.21 V^2/g ~ 2.1 m (Bretschneider), and sigma = H_1/3 / 4.
inline float phillipsAmplitude(double sigma, double L, double V, double g) {
    const double Lw = V * V / g;
    return static_cast<float>(4.0 * 3.14159265358979323846 * sigma * sigma / (L * L * Lw * Lw));
}

inline OceanParams defaultParams() {
    OceanParams p;
    p.A = phillipsAmplitude(kOceanSigmaM, kOceanL, kOceanWindMs, p.g);
    return p;
}

// The capillary cascade: the same spectrum on a 96 m patch, above the main grid's corner wavenumber.
inline OceanParams defaultCapillaryParams() {
    OceanParams p = defaultParams();
    p.L = kOceanCapL;
    p.kMin = static_cast<float>(std::sqrt(2.0) * 3.14159265358979323846 * kOceanN / kOceanL);
    p.gain = kOceanCapGain;
    return p;
}

// Deterministic Gaussian sample from a 32-bit LCG (Box-Muller). Enough for a reproducible spectrum.
struct Gauss {
    uint32_t s;
    explicit Gauss(uint32_t seed) : s(seed) {}
    double uniform() {
        s = s * 1664525u + 1013904223u;
        return ((s >> 8) + 0.5) / 16777216.0;  // (0, 1)
    }
    double normal() {
        const double u1 = uniform(), u2 = uniform();
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * 3.14159265358979323846 * u2);
    }
};

inline float waveNumber(uint32_t n, uint32_t N, float L) {
    const int s = n < N / 2 ? static_cast<int>(n) : static_cast<int>(n) - static_cast<int>(N);
    return 6.28318530717958647692f * static_cast<float>(s) / L;
}

// Phillips spectrum value at wave vector (kx, kz).
inline double phillips(const OceanParams& p, double kx, double kz) {
    const double k = std::sqrt(kx * kx + kz * kz);
    if (k < 1e-9 || k < p.kMin) return 0.0;
    const double Lw = static_cast<double>(p.windSpeed) * p.windSpeed / p.g;
    const double wl = std::sqrt(static_cast<double>(p.windX) * p.windX + static_cast<double>(p.windZ) * p.windZ);
    const double wx = p.windX / wl, wz = p.windZ / wl;
    const double dot = (kx * wx + kz * wz) / k;  // k^ . w^
    const double l2 = static_cast<double>(p.smallWaveL) * p.smallWaveL;
    return p.gain * p.A * std::exp(-1.0 / ((k * Lw) * (k * Lw))) / (k * k * k * k) * dot * dot * std::exp(-k * k * l2);
}

// Builds h0 for the whole grid. out has 2 * N * N floats.
inline void buildH0(const OceanParams& p, std::vector<float>& out, uint32_t seed) {
    const uint32_t N = p.N;
    out.assign(static_cast<size_t>(2) * N * N, 0.0f);
    Gauss rng(seed);
    for (uint32_t z = 0; z < N; z++) {
        for (uint32_t x = 0; x < N; x++) {
            const double kx = waveNumber(x, N, p.L), kz = waveNumber(z, N, p.L);
            const double amp = std::sqrt(phillips(p, kx, kz)) / std::sqrt(2.0);
            const size_t i = (static_cast<size_t>(z) * N + x) * 2;
            out[i] = static_cast<float>(amp * rng.normal());
            out[i + 1] = static_cast<float>(amp * rng.normal());
        }
    }
}

}  // namespace ocean
}  // namespace engine
