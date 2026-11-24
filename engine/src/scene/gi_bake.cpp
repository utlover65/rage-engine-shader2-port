#include "scene/gi_bake.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>

#include "scene/landscape.h"

namespace engine {
namespace scene {

namespace {

constexpr int kDirs = 256;            // rays per probe (Fibonacci sphere)
constexpr float kPi = 3.14159265358979f;
constexpr float kSeaAlbedo = 0.02f;   // water is dark from above (Fresnel and absorption)
constexpr float kTerrainAlbedo = 0.25f;

float halfToFloat(uint16_t h) {
    const uint32_t sign = (h >> 15) & 1u, exp = (h >> 10) & 0x1fu, man = h & 0x3ffu;
    float v;
    if (exp == 0) v = std::ldexp(static_cast<float>(man), -24);
    else if (exp == 31) v = man ? std::numeric_limits<float>::quiet_NaN() : std::numeric_limits<float>::infinity();
    else v = std::ldexp(static_cast<float>(man | 0x400u), static_cast<int>(exp) - 25);
    return sign ? -v : v;
}

struct Tri {
    float v0[3], e1[3], e2[3], n[3], alb[3];
};

// Sky radiance along a unit direction, bilinear in the sky-view LUT (same parametrisation as
// skyViewUV in shaders/common.glsl).
std::array<float, 3> skySample(const SkyTable& s, float dx, float dy, float dz) {
    float x01 = std::atan2(dz, dx) / (2.0f * kPi);
    x01 -= std::floor(x01);
    const float e = std::asin(std::clamp(dy, -1.0f, 1.0f));
    const float sgn = e < 0.0f ? -1.0f : 1.0f;
    const float y01 = 0.5f * sgn * std::sqrt(std::fabs(e) / (0.5f * kPi)) + 0.5f;
    const float W = static_cast<float>(s.width), H = static_cast<float>(s.height);
    const float u = x01 * (W - 1.0f) / W + 0.5f / W, v = y01 * (H - 1.0f) / H + 0.5f / H;
    const float fx = u * W - 0.5f, fy = v * H - 0.5f;
    const int x0 = static_cast<int>(std::floor(fx)), y0 = static_cast<int>(std::floor(fy));
    const float tx = fx - x0, ty = fy - y0;
    auto texel = [&](int x, int y, int c) {
        x = std::clamp(x, 0, s.width - 1);
        y = std::clamp(y, 0, s.height - 1);
        return s.rgba[(static_cast<size_t>(y) * s.width + x) * 4 + c];
    };
    std::array<float, 3> out{};
    for (int c = 0; c < 3; c++) {
        const float a = texel(x0, y0, c) + (texel(x0 + 1, y0, c) - texel(x0, y0, c)) * tx;
        const float b = texel(x0, y0 + 1, c) + (texel(x0 + 1, y0 + 1, c) - texel(x0, y0 + 1, c)) * tx;
        out[c] = a + (b - a) * ty;
    }
    return out;
}

// Möller-Trumbore. Returns the hit distance, or a negative value for a miss.
float rayTriangle(const float* o, const float* d, const Tri& t) {
    const float px = d[1] * t.e2[2] - d[2] * t.e2[1];
    const float py = d[2] * t.e2[0] - d[0] * t.e2[2];
    const float pz = d[0] * t.e2[1] - d[1] * t.e2[0];
    const float det = t.e1[0] * px + t.e1[1] * py + t.e1[2] * pz;
    if (std::fabs(det) < 1e-9f) return -1.0f;
    const float inv = 1.0f / det;
    const float sx = o[0] - t.v0[0], sy = o[1] - t.v0[1], sz = o[2] - t.v0[2];
    const float u = (sx * px + sy * py + sz * pz) * inv;
    if (u < 0.0f || u > 1.0f) return -1.0f;
    const float qx = sy * t.e1[2] - sz * t.e1[1];
    const float qy = sz * t.e1[0] - sx * t.e1[2];
    const float qz = sx * t.e1[1] - sy * t.e1[0];
    const float v = (d[0] * qx + d[1] * qy + d[2] * qz) * inv;
    if (v < 0.0f || u + v > 1.0f) return -1.0f;
    const float tt = (t.e2[0] * qx + t.e2[1] * qy + t.e2[2] * qz) * inv;
    return tt > 1e-3f ? tt : -1.0f;
}

// Ray march of the headland heightfield. Returns the distance, or a negative value.
float rayTerrain(const float* o, const float* d, float tMax) {
    float prevAbove = o[1] - landHeight(o[0], o[2]);
    for (float t = 0.5f; t < tMax; t += 1.0f) {
        const float x = o[0] + d[0] * t, y = o[1] + d[1] * t, z = o[2] + d[2] * t;
        const float above = y - landHeight(x, z);
        if (above < 0.0f && prevAbove >= 0.0f) return t;
        prevAbove = above;
    }
    return -1.0f;
}

void terrainNormal(float x, float z, float* n) {
    const float e = 0.5f;
    const float gx = (landHeight(x + e, z) - landHeight(x - e, z)) / (2.0f * e);
    const float gz = (landHeight(x, z + e) - landHeight(x, z - e)) / (2.0f * e);
    const float len = std::sqrt(gx * gx + 1.0f + gz * gz);
    n[0] = -gx / len; n[1] = 1.0f / len; n[2] = -gz / len;
}

}  // namespace

std::vector<float> halfRgbaToFloat(const std::vector<uint8_t>& bytes) {
    std::vector<float> out(bytes.size() / 2);
    for (size_t i = 0; i < out.size(); i++) {
        uint16_t h;
        std::memcpy(&h, bytes.data() + 2 * i, 2);
        out[i] = halfToFloat(h);
    }
    return out;
}

ProbeSet bakeAmbientProbes(const std::vector<float>& objects, const SkyTable& sky) {
    ProbeSet ps;
    ps.dims[0] = ps.dims[1] = ps.dims[2] = 4;
    // Probe grid over the beach and the near shelf: x -120..120, z -110..90, heights 1.5..16 m.
    ps.origin[0] = -120.0f; ps.origin[1] = 1.5f; ps.origin[2] = -110.0f;
    ps.step[0] = 240.0f / 3.0f; ps.step[1] = (16.0f - 1.5f) / 3.0f; ps.step[2] = 200.0f / 3.0f;

    // Scene triangles, precomputed for the ray tests.
    std::vector<Tri> tris;
    tris.reserve(objects.size() / 30);
    for (size_t i = 0; i + 29 < objects.size(); i += 30) {
        Tri t;
        const float* a = &objects[i];
        const float* b = &objects[i + 10];
        const float* c = &objects[i + 20];
        for (int k = 0; k < 3; k++) {
            t.v0[k] = a[k];
            t.e1[k] = b[k] - a[k];
            t.e2[k] = c[k] - a[k];
            t.n[k] = a[3 + k];
            t.alb[k] = a[6 + k];
        }
        tris.push_back(t);
    }

    // Fibonacci sphere directions, shared by every probe.
    std::vector<std::array<float, 3>> dirs(kDirs);
    for (int i = 0; i < kDirs; i++) {
        const float z = 1.0f - (2.0f * i + 1.0f) / kDirs;
        const float r = std::sqrt(std::max(0.0f, 1.0f - z * z));
        const float phi = i * 2.39996323f;
        dirs[i] = {r * std::cos(phi), z, r * std::sin(phi)};
    }

    const int count = ps.probeCount();
    ps.faces.assign(static_cast<size_t>(count) * 6 * 4, 0.0f);
    const float faceAxis[6][3] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};

    for (int pz = 0; pz < ps.dims[2]; pz++) {
        for (int py = 0; py < ps.dims[1]; py++) {
            for (int px = 0; px < ps.dims[0]; px++) {
                const int index = px + ps.dims[0] * (py + ps.dims[1] * pz);
                const float o[3] = {ps.origin[0] + px * ps.step[0], ps.origin[1] + py * ps.step[1],
                                    ps.origin[2] + pz * ps.step[2]};
                float sum[6][3] = {}, wsum[6] = {};
                for (const auto& dir : dirs) {
                    const float* d = dir.data();
                    float bestT = std::numeric_limits<float>::infinity();
                    std::array<float, 3> alb{}, n{};
                    bool hit = false;
                    for (const Tri& t : tris) {
                        const float th = rayTriangle(o, d, t);
                        if (th > 0.0f && th < bestT) {
                            bestT = th;
                            hit = true;
                            alb = {t.alb[0], t.alb[1], t.alb[2]};
                            n = {t.n[0], t.n[1], t.n[2]};
                        }
                    }
                    const float tt = rayTerrain(o, d, std::min(bestT, 400.0f));
                    if (tt > 0.0f && tt < bestT) {
                        bestT = tt;
                        hit = true;
                        const float x = o[0] + d[0] * tt, z = o[2] + d[2] * tt;
                        float nn[3];
                        terrainNormal(x, z, nn);
                        n = {nn[0], nn[1], nn[2]};
                        alb = {kTerrainAlbedo, kTerrainAlbedo, kTerrainAlbedo};
                    }
                    if (d[1] < 0.0f) {
                        const float ts = -o[1] / d[1];
                        if (ts > 0.0f && ts < bestT) {
                            bestT = ts;
                            hit = true;
                            n = {0.0f, 1.0f, 0.0f};
                            alb = {kSeaAlbedo, kSeaAlbedo, kSeaAlbedo};
                        }
                    }
                    // Radiance: sky seen directly, or one bounce of sky light off the surface hit.
                    std::array<float, 3> L;
                    if (hit) {
                        const auto skyAtSurface = skySample(sky, n[0], n[1], n[2]);
                        L = {alb[0] * skyAtSurface[0], alb[1] * skyAtSurface[1], alb[2] * skyAtSurface[2]};
                    } else {
                        L = skySample(sky, d[0], d[1], d[2]);
                    }
                    for (int f = 0; f < 6; f++) {
                        const float w = std::max(0.0f, d[0] * faceAxis[f][0] + d[1] * faceAxis[f][1] + d[2] * faceAxis[f][2]);
                        if (w <= 0.0f) continue;
                        for (int c = 0; c < 3; c++) sum[f][c] += L[c] * w;
                        wsum[f] += w;
                    }
                }
                for (int f = 0; f < 6; f++) {
                    for (int c = 0; c < 3; c++) {
                        ps.faces[(static_cast<size_t>(index) * 6 + f) * 4 + c] =
                            wsum[f] > 0.0f ? sum[f][c] / wsum[f] : 0.0f;
                    }
                }
            }
        }
    }
    return ps;
}

}  // namespace scene
}  // namespace engine
