// Renders the open-sea scene through the RHI and render graph, then checks the GPU tables
// against the double-precision CPU reference and the final frame for sanity. Writes
// render.ppm (row 0 at the top). ENGINE_VALIDATION=1 enables the validation layer.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "atmo/atmo_reference.h"
#include "renderer/ocean_renderer.h"

namespace {

using engine::atmo::V3;

float halfToFloat(uint16_t h) {
    const uint32_t s = (h >> 15) & 1u;
    uint32_t e = (h >> 10) & 0x1Fu;
    uint32_t m = h & 0x3FFu;
    uint32_t f;
    if (e == 0) {
        if (m == 0) {
            f = s << 31;
        } else {  // subnormal: normalise
            e = 127 - 15 + 1;
            while (!(m & 0x400u)) { m <<= 1; e--; }
            m &= 0x3FFu;
            f = (s << 31) | (e << 23) | (m << 13);
        }
    } else if (e == 31) {
        f = (s << 31) | (0xFFu << 23) | (m << 13);
    } else {
        f = (s << 31) | ((e + 112) << 23) | (m << 13);
    }
    float r;
    std::memcpy(&r, &f, 4);
    return r;
}

// RGBA16F bytes to floats, 4 per texel.
std::vector<float> halfImageToFloat(const std::vector<uint8_t>& bytes) {
    std::vector<float> out(bytes.size() / 2);
    for (size_t i = 0; i < out.size(); i++) {
        uint16_t h;
        std::memcpy(&h, bytes.data() + 2 * i, 2);
        out[i] = halfToFloat(h);
    }
    return out;
}

// Bilinear fetch of channel c of an RGBA float table (w x h), at normalised (x01, y01).
// Texel i holds x01 = i / (w - 1), matching the GPU layout; clamp addressing.
double bilinear(const std::vector<float>& t, int w, int h, int c, double x01, double y01) {
    const double tx = std::clamp(x01 * (w - 1), 0.0, w - 1.0);
    const double ty = std::clamp(y01 * (h - 1), 0.0, h - 1.0);
    const int x0 = std::min(static_cast<int>(tx), w - 2), y0 = std::min(static_cast<int>(ty), h - 2);
    const double fx = tx - x0, fy = ty - y0;
    auto at = [&](int x, int y) { return static_cast<double>(t[(static_cast<size_t>(y) * w + x) * 4 + c]); };
    return (1 - fy) * ((1 - fx) * at(x0, y0) + fx * at(x0 + 1, y0)) + fy * ((1 - fx) * at(x0, y0 + 1) + fx * at(x0 + 1, y0 + 1));
}

// Sky-view coordinates for a direction, as in skyViewUV() in shaders/common.glsl.
void skyViewParams(V3 d, double& x01, double& y01) {
    double x = std::atan2(d.z, d.x) / (2.0 * engine::atmo::kPi);
    x01 = x - std::floor(x);
    const double e = std::asin(std::clamp(d.y, -1.0, 1.0));
    const double s = (e < 0 ? -1.0 : 1.0) * std::sqrt(std::fabs(e) / (0.5 * engine::atmo::kPi));
    y01 = 0.5 * s + 0.5;
}

// Direction of sky-view texel (i, j), as the GPU table generator builds it.
V3 skyViewTexelDir(int i, int j, int w, int h) {
    const double x01 = static_cast<double>(i) / (w - 1), y01 = static_cast<double>(j) / (h - 1);
    const double s = 2.0 * y01 - 1.0;
    const double e = s * std::fabs(s) * 0.5 * engine::atmo::kPi;
    const double phi = x01 * 2.0 * engine::atmo::kPi;
    return {std::cos(e) * std::cos(phi), std::sin(e), std::cos(e) * std::sin(phi)};
}

struct Stats {
    double median = 0, p95 = 0, p99 = 0, max = 0;
    size_t n = 0;
};

Stats stats(std::vector<double> e) {
    Stats r;
    r.n = e.size();
    if (e.empty()) return r;
    std::sort(e.begin(), e.end());
    auto at = [&](double q) { return e[std::min(e.size() - 1, static_cast<size_t>(q * e.size()))]; };
    r.median = at(0.5);
    r.p95 = at(0.95);
    r.p99 = at(0.99);
    r.max = e.back();
    return r;
}

}  // namespace

int main() {
    try {
        engine::rhi::DeviceOptions opts;
        opts.validation = std::getenv("ENGINE_VALIDATION") != nullptr;
        auto dev = engine::rhi::createDevice(engine::rhi::Backend::Vulkan, opts);
        std::printf("backend %s, device %s\n", dev->backendName(), dev->deviceName().c_str());

        engine::renderer::Settings s;
#if defined(__ANDROID__)
        // Conservative mobile defaults: keep the full HDR/atmosphere pipeline, but cap
        // render resolution and shadow bandwidth for thermally constrained GPUs.
        s.width = 960;
        s.height = 540;
        s.shadowMapSize = 1024;
#endif
        // Runtime overrides let the same binary scale from mobile to desktop.
        auto envU32 = [](const char* name, uint32_t fallback) {
            const char* value = std::getenv(name);
            if (!value || !*value) return fallback;
            char* end = nullptr;
            const unsigned long parsed = std::strtoul(value, &end, 10);
            return (end && *end == '\0' && parsed >= 1 && parsed <= 16384)
                ? static_cast<uint32_t>(parsed) : fallback;
        };
        s.width = envU32("ENGINE_WIDTH", s.width);
        s.height = envU32("ENGINE_HEIGHT", s.height);
        s.shadowMapSize = envU32("ENGINE_SHADOW_SIZE", s.shadowMapSize);
        // ENGINE_STILL=1 renders one 1920x1080 frame with no timing loop, for a quality still.
        const bool still = std::getenv("ENGINE_STILL") != nullptr;
        if (still) {
            s.width = envU32("ENGINE_WIDTH", 1920);
            s.height = envU32("ENGINE_HEIGHT", 1080);
        }
        // Shader and asset directories: compile-time defaults, overridable at run time so a
        // copied build folder (for example on Android storage) finds its files.
        const char* shaderEnv = std::getenv("ENGINE_SHADER_DIR");
        const char* assetEnv = std::getenv("ENGINE_ASSET_DIR");
        const std::string shaderDir = (shaderEnv && *shaderEnv) ? shaderEnv : ENGINE_SHADER_DIR;
        const std::string assetDir = (assetEnv && *assetEnv) ? assetEnv : ENGINE_ASSET_DIR;
        // ENGINE_SUN_ELEV_DEG overrides the sun elevation (keeps the scene's azimuth). The default
        // 1.2 degrees puts sun-blocking clouds about 57 km away, beyond the cloud pass's 30 km draw range.
        if (const char* elev = std::getenv("ENGINE_SUN_ELEV_DEG"); elev && *elev) {
            const double e = std::atof(elev) * engine::atmo::kPi / 180.0;
            const double hx = s.sunDir.x, hz = s.sunDir.z;
            const double hl = std::sqrt(hx * hx + hz * hz);
            s.sunDir = {std::cos(e) * hx / hl, std::sin(e), std::cos(e) * hz / hl};
        }
        if (const char* ev = std::getenv("ENGINE_EV100"); ev && *ev) s.ev100 = std::atof(ev);
        engine::renderer::OceanRenderer r(*dev, shaderDir, assetDir, s);
        std::printf("graph:\n%s", r.describeGraph().c_str());
        const auto sunU = r.sunDirUnit();
        const auto sunL = r.sunIlluminanceLux();
        std::printf("sun: elevation %.2f deg, sea-level sun illuminance %.4g %.4g %.4g lux\n",
                    std::asin(sunU.y) * 180.0 / engine::atmo::kPi, sunL.x, sunL.y, sunL.z);

        // Warm-up, then 120 timed frames. Each frame waits for the GPU, so the time is the
        // wall-clock cost of one frame including command submission and the wait.
        const bool quick = std::getenv("ENGINE_QUICK") != nullptr;
        const int warmFrames = (still || quick) ? 1 : 10;
        const int timedFrames = (still || quick) ? 1 : 120;
        // ENGINE_TIME sets the scene clock (seconds) for the measured frame; clouds drift with it.
        const char* timeEnv = std::getenv("ENGINE_TIME");
        const double timeBase = (timeEnv && *timeEnv) ? std::atof(timeEnv) : 0.33;
        for (int f = 0; f < warmFrames; f++) r.renderFrame(f * 0.033);
        std::vector<double> frameMs;
        std::vector<std::string> passOrder;
        std::map<std::string, std::vector<double>> passMs;
        for (int f = 0; f < timedFrames; f++) {
            r.renderFrame(timeBase + f * 0.033);
            frameMs.push_back(r.lastFrameMs());
            for (const auto& [name, ms] : r.lastPassTimesMs()) {
                if (!passMs.count(name)) passOrder.push_back(name);
                passMs[name].push_back(ms);
            }
        }
        const Stats ft = stats(frameMs);
        std::printf("frame time on %s, %ux%u, %d frames: median %.1f ms, p95 %.1f ms, max %.1f ms\n",
                    dev->deviceName().c_str(), s.width, s.height, timedFrames, ft.median, ft.p95, ft.max);
        // Per-pass GPU times (ENGINE_PROFILE): the median over the timed frames, slowest first.
        if (std::getenv("ENGINE_PROFILE")) {
            if (passOrder.empty()) {
                std::printf("GPU timestamps unavailable on this device\n");
            } else {
                std::vector<std::pair<double, std::string>> rows;
                double total = 0.0;
                for (const auto& name : passOrder) {
                    const double m = stats(passMs[name]).median;
                    rows.emplace_back(m, name);
                    total += m;
                }
                std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
                std::printf("GPU time per pass (median of %d frames):\n", timedFrames);
                for (const auto& row : rows) std::printf("  %-30s %8.3f ms\n", row.second.c_str(), row.first);
                std::printf("  %-30s %8.3f ms\n", "sum of passes", total);
            }
        }

        // ENGINE_BENCH=1: stop after the timing report, so a device run skips the CPU-reference suite.
        if (std::getenv("ENGINE_BENCH")) {
            std::printf("ENGINE_BENCH: timing only, validation suite skipped\n");
            return 0;
        }

        if (quick) {
            // Mobile smoke/render mode: skip the expensive CPU-reference validation suite.
            // The renderer is still fully executed and the display image is read back.
            const auto pixels = r.readImage(r.display(), 4);
            std::FILE* out = std::fopen("render.ppm", "wb");
            if (!out) throw std::runtime_error("cannot write render.ppm");
            std::fprintf(out, "P6\n%u %u\n255\n", s.width, s.height);
            for (size_t i = 0; i + 3 < pixels.size(); i += 4) {
                std::fputc(pixels[i + 0], out);
                std::fputc(pixels[i + 1], out);
                std::fputc(pixels[i + 2], out);
            }
            std::fclose(out);
            std::printf("wrote render.ppm (%ux%u); quick mode skipped CPU-reference diagnostics\n", s.width, s.height);
            return 0;
        }

        const double k = r.preExposure();
        const V3 sun{sunU.x, sunU.y, sunU.z};
        const double r0 = engine::atmo::kRg + s.camPos.y;
        bool ok = true;

        // 1. Transmittance table. Grid texels, then random points (bilinear), vs the reference.
        const int TW = 256, TH = 64;
        const auto trans = halfImageToFloat(r.readImage(r.transmittance(), 8));
        std::vector<double> terr;
        for (int j = 0; j < TH; j++)
            for (int i = 0; i < TW; i++) {
                const double t = 2.0 * i / (TW - 1) - 1.0;
                const double mu = t * std::fabs(t);
                const double v = static_cast<double>(j) / (TH - 1);
                const double rr = engine::atmo::kRg + (engine::atmo::kRt - engine::atmo::kRg) * v * v;
                double ref[3];
                engine::atmo::transmittance(rr, mu, ref);
                if (ref[0] > 1e-3) terr.push_back(std::fabs(trans[(static_cast<size_t>(j) * TW + i) * 4] - ref[0]) / ref[0]);
            }
        unsigned seed = 12345u;
        auto rnd = [&seed]() { seed = seed * 1664525u + 1013904223u; return (seed >> 8) * (1.0 / 16777216.0); };
        for (int n = 0; n < 3000; n++) {
            const double mu = 2.0 * rnd() - 1.0;
            const double rr = engine::atmo::kRg + (engine::atmo::kRt - engine::atmo::kRg) * rnd();
            if (engine::atmo::hitsGround(rr, mu)) continue;
            const double u = 0.5 + 0.5 * (mu < 0 ? -1.0 : 1.0) * std::sqrt(std::fabs(mu));
            const double v = std::sqrt((rr - engine::atmo::kRg) / (engine::atmo::kRt - engine::atmo::kRg));
            double ref[3];
            engine::atmo::transmittance(rr, mu, ref);
            if (ref[0] > 1e-3) terr.push_back(std::fabs(bilinear(trans, TW, TH, 0, u, v) - ref[0]) / ref[0]);
        }
        const Stats te = stats(terr);
        std::printf("transmittance LUT (%zu points): relative error median %.2e, p95 %.2e, p99 %.2e, max %.2e\n", te.n,
                    te.median, te.p95, te.p99, te.max);
        // The max sits on the Earth-shadow boundary, where the table steps from its value to
        // zero; the p99 bound is the meaningful one.
        if (te.p99 > 0.01) { std::printf("  FAIL: transmittance table off\n"); ok = false; }

        // 1b. Multiple-scattering table: every texel finite and non-negative, and close to the
        //     CPU table built with the same integrals.
        const auto msCpu = engine::atmo::multiScatterTable();
        const int MW = 32, MH = 32;
        const auto ms = halfImageToFloat(r.readImage(r.multiScatter(), 8));
        double msMax = 0.0;
        for (double v : msCpu) msMax = std::max(msMax, v);
        std::vector<double> merr;
        size_t msBad = 0;
        for (int y = 0; y < MH; y++)
            for (int x = 0; x < MW; x++)
                for (int c = 0; c < 3; c++) {
                    const double g = ms[(static_cast<size_t>(y) * MW + x) * 4 + c];
                    const double ref = msCpu[(static_cast<size_t>(y) * MW + x) * 3 + c];
                    if (!std::isfinite(g) || g < 0.0) { msBad++; continue; }
                    // Texels below 1% of the table maximum are skipped (relative error not meaningful).
                    if (ref > 0.01 * msMax) merr.push_back(std::fabs(g - ref) / ref);
                }
        const Stats me = stats(merr);
        std::printf("multiple-scattering LUT: %zu non-finite or negative values; relative error median %.2e, p95 %.2e, max %.2e (%zu)\n",
                    msBad, me.median, me.p95, me.max, me.n);
        if (msBad) { std::printf("  FAIL: multiple-scattering table has non-finite or negative values\n"); ok = false; }
        if (me.p95 > 0.05) { std::printf("  FAIL: multiple-scattering table off\n"); ok = false; }

        // 2. Sky-view table at its texel directions, all three channels. Entries below 1% of
        //    the brightest channel value are skipped (their relative error is not meaningful).
        const int SW = 193, SH = 108;
        const auto sky = halfImageToFloat(r.readImage(r.skyView(), 8));
        double maxRef = 0.0;
        for (int j = 0; j < SH; j++)
            for (int i = 0; i < SW; i++) {
                double ref[3];
                engine::atmo::skyRadiance(r0, skyViewTexelDir(i, j, SW, SH), sun, msCpu, ref);
                maxRef = std::max({maxRef, ref[0], ref[1], ref[2]});
            }
        std::vector<double> serr;
        for (int j = 0; j < SH; j++)
            for (int i = 0; i < SW; i++) {
                double ref[3];
                engine::atmo::skyRadiance(r0, skyViewTexelDir(i, j, SW, SH), sun, msCpu, ref);
                for (int c = 0; c < 3; c++) {
                    if (ref[c] < 0.01 * maxRef) continue;
                    const double g = sky[(static_cast<size_t>(j) * SW + i) * 4 + c] / k;
                    serr.push_back(std::fabs(g - ref[c]) / ref[c]);
                }
            }
        const Stats se = stats(serr);
        std::printf("sky-view LUT (texel directions, RGB): relative error median %.2e, p95 %.2e, max %.2e (%zu values)\n",
                    se.median, se.p95, se.max, se.n);
        if (se.p95 > 0.05) { std::printf("  FAIL: sky-view table off\n"); ok = false; }

        // 3. Lookup mapping: random directions, bilinear fetch of the table vs the reference
        //    evaluated directly. Checks the direction-to-texel mapping and the filtering.
        std::vector<double> lerr;
        for (int n = 0; n < 2000; n++) {
            const double phi = 2.0 * engine::atmo::kPi * rnd();
            const double el = (rnd() * 2.0 - 0.3) * engine::atmo::kPi * 0.5;
            const V3 d{std::cos(el) * std::cos(phi), std::sin(el), std::cos(el) * std::sin(phi)};
            double x01, y01;
            skyViewParams(d, x01, y01);
            double ref[3];
            engine::atmo::skyRadiance(r0, d, sun, msCpu, ref);
            for (int c = 0; c < 3; c++) {
                if (ref[c] < 0.01 * maxRef) continue;
                lerr.push_back(std::fabs(bilinear(sky, SW, SH, c, x01, y01) / k - ref[c]) / ref[c]);
            }
        }
        const Stats le = stats(lerr);
        std::printf("sky-view lookup (random directions, bilinear, RGB): relative error median %.2e, p95 %.2e, max %.2e (%zu)\n",
                    le.median, le.p95, le.max, le.n);
        if (le.p95 > 0.05) { std::printf("  FAIL: sky-view lookup mapping off\n"); ok = false; }

        // 4. Final HDR frame: every value finite.
        const auto hdr = halfImageToFloat(r.readImage(r.hdr(), 8));
        size_t bad = 0;
        float hdrMax = 0.0f;
        for (float v : hdr) {
            if (!std::isfinite(v)) bad++;
            else hdrMax = std::max(hdrMax, v);
        }
        std::printf("HDR frame: %zu non-finite values, max pre-exposed radiance %.4g\n", bad, hdrMax);
        // God rays: the removed in-scatter is never positive, and is non-zero where clouds shade the air.
        const auto shaft = halfImageToFloat(r.readImage(r.godRays(), 8));
        size_t shaftPositive = 0, shaftNonZero = 0;
        double shaftMin = 0.0, shaftMax = -1e30;
        for (size_t i = 0; i + 3 < shaft.size(); i += 4) {
            for (int c = 0; c < 3; c++) {
                const double v = shaft[i + c];
                if (!std::isfinite(v)) { shaftPositive++; continue; }
                if (v > 0.0) shaftPositive++;
                if (std::fabs(v) > 1e-9) shaftNonZero++;
                shaftMin = std::min(shaftMin, v);
                shaftMax = std::max(shaftMax, v);
            }
        }
        std::printf("god rays (quarter res): min %.4g, max %.4g, %zu values non-zero, %zu positive (must be 0)\n",
                    shaftMin, shaftMax, shaftNonZero, shaftPositive);
        if (std::getenv("ENGINE_DUMP_SHAFT")) {
            // Debug: removed in-scatter as a grey image, 0 black .. 2 white (linear).
            const uint32_t gw = r.godRays().width(), gh = r.godRays().height();
            std::FILE* g = std::fopen("godrays.ppm", "wb");
            if (g) {
                std::fprintf(g, "P6\n%u %u\n255\n", gw, gh);
                for (size_t i = 0; i + 3 < shaft.size(); i += 4) {
                    const double v = std::min(1.0, -shaft[i] / 2.0);
                    const unsigned char b = static_cast<unsigned char>(v * 255.0 + 0.5);
                    const unsigned char rgb[3] = {b, b, b};
                    std::fwrite(rgb, 1, 3, g);
                }
                std::fclose(g);
                std::printf("wrote godrays.ppm (%ux%u)\n", gw, gh);
            }
        }
        if (bad) { std::printf("  FAIL: non-finite radiance\n"); ok = false; }
        const auto bloom = halfImageToFloat(r.readImage(r.bloom(), 8));
        size_t bloomBad = 0;
        for (float v : bloom)
            if (!std::isfinite(v) || v < 0.0f) bloomBad++;
        std::printf("bloom (half res): %zu non-finite or negative values\n", bloomBad);
        if (bloomBad) { std::printf("  FAIL: bloom has non-finite or negative values\n"); ok = false; }

        // Sun shadow map (stored as 1 - depth): the objects must have written texels, and every
        // value must lie in [0, 1]. The cleared background is 0.
        const auto shadowBytes = r.readImage(r.shadowMap(), 4);
        size_t occluded = 0, shadowBad = 0;
        for (size_t i = 0; i + 4 <= shadowBytes.size(); i += 4) {
            float d;
            std::memcpy(&d, shadowBytes.data() + i, 4);
            if (!std::isfinite(d) || d < 0.0f || d > 1.0f) shadowBad++;
            else if (d > 0.0f) occluded++;
        }
        std::printf("sun shadow map: %zu texels occluded by objects, %zu out of range\n", occluded, shadowBad);
        if (occluded == 0) { std::printf("  FAIL: the objects cast no shadow\n"); ok = false; }
        if (shadowBad) { std::printf("  FAIL: shadow map has values outside [0, 1]\n"); ok = false; }

        // 5. Display frame: PPM output and a layout check (sky on top, sea below).
        const auto ldr = r.readImage(r.display(), 4);
        const uint32_t W = s.width, H = s.height;
        auto px = [&](uint32_t x, uint32_t y) { return ldr.data() + (static_cast<size_t>(y) * W + x) * 4; };
        std::FILE* f = std::fopen("render.ppm", "wb");
        if (!f) throw std::runtime_error("cannot write render.ppm");
        std::fprintf(f, "P6\n%u %u\n255\n", W, H);
        for (uint32_t y = 0; y < H; y++)
            for (uint32_t x = 0; x < W; x++) {
                const uint8_t* p = px(x, y);
                std::fputc(p[0], f);
                std::fputc(p[1], f);
                std::fputc(p[2], f);
            }
        std::fclose(f);
        // Sunset bay: colours are printed for inspection only; no reference photograph is available.
        const uint8_t* centre = px(W / 2, H / 2);
        std::printf("centre pixel %u %u %u\n", centre[0], centre[1], centre[2]);
        std::printf("wrote render.ppm (%ux%u)\n", W, H);

        if (dev->validationErrorCount() != 0) {
            std::printf("FAIL: validation reported %u errors\n", dev->validationErrorCount());
            ok = false;
        }
        std::printf("%s\n", ok ? "PASS" : "FAIL");
        return ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "engine_render: %s\n", e.what());
        return 1;
    }
}
