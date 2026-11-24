#include "scene/landscape.h"

#include <algorithm>
#include <cmath>

#include "scene/objects.h"

namespace engine {
namespace scene {

namespace {

float smooth01(float t) {
    t = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
    return t * t * (3.0f - 2.0f * t);
}

}  // namespace

// Mirrored exactly in shaders/ocean.frag (shoreZ, seabedHeight). Keep the constants in sync.
float shoreZ(float x) {
    return -34.0f + 12.0f * std::sin(0.011f * x + 0.4f) + 5.0f * std::sin(0.029f * x - 1.1f);
}

float landHeight(float x, float z) {
    const float d = z - shoreZ(x);  // > 0 on land, < 0 at sea
    if (d >= 0.0f) {
        // Dry sand: a dune ramp rising inland, with a little undulation near the shore.
        const float s = smooth01(d / 28.0f);
        const float w = smooth01(d / 12.0f);
        return 3.5f * s + w * (0.5f * std::sin(0.09f * x + 0.04f * z) +
                               0.4f * std::sin(0.053f * z - 0.031f * x + 1.7f));
    }
    // Shelf: gentle seaward slope, a steeper drop-off beyond 45 m, and sand bars near the shore.
    const float u = -d;
    float depth = 0.085f * std::min(u, 45.0f) + 0.3f * std::max(u - 45.0f, 0.0f);
    depth = std::min(depth, 40.0f);
    depth += 0.35f * std::sin(0.21f * x + 0.37f * z) * smooth01(u / 10.0f);
    return -std::max(depth, 0.0f);
}

std::vector<float> buildLandscape(float cellSize) {
    std::vector<float> v;
    const int nx = static_cast<int>(std::ceil((kLandX1 - kLandX0) / cellSize));
    const int nz = static_cast<int>(std::ceil((kLandZ1 - kLandZ0) / cellSize));
    const float cx = (kLandX1 - kLandX0) / static_cast<float>(nx);
    const float cz = (kLandZ1 - kLandZ0) / static_cast<float>(nz);
    const int nxv = nx + 1, nzv = nz + 1;
    std::vector<float> hgt(static_cast<size_t>(nxv) * nzv), nrm(static_cast<size_t>(nxv) * nzv * 3);
    auto X = [&](int i) { return kLandX0 + i * cx; };
    auto Z = [&](int j) { return kLandZ0 + j * cz; };
    for (int j = 0; j < nzv; j++) {
        for (int i = 0; i < nxv; i++) {
            hgt[static_cast<size_t>(j) * nxv + i] = landHeight(X(i), Z(j));
        }
    }
    for (int j = 0; j < nzv; j++) {
        for (int i = 0; i < nxv; i++) {
            const float x = X(i), z = Z(j);
            const float e = 0.5f * cellSize;
            const float gx = (landHeight(x + e, z) - landHeight(x - e, z)) / (2.0f * e);
            const float gz = (landHeight(x, z + e) - landHeight(x, z - e)) / (2.0f * e);
            const float len = std::sqrt(gx * gx + 1.0f + gz * gz);
            const size_t k = (static_cast<size_t>(j) * nxv + i) * 3;
            nrm[k] = -gx / len; nrm[k + 1] = 1.0f / len; nrm[k + 2] = -gz / len;
        }
    }
    const float sand[3] = {0.66f, 0.57f, 0.43f};
    const float rock[3] = {0.31f, 0.30f, 0.27f};
    auto vert = [&](int i, int j, float material, const float* alb) {
        const float p[3] = {X(i), hgt[static_cast<size_t>(j) * nxv + i], Z(j)};
        const size_t k = (static_cast<size_t>(j) * nxv + i) * 3;
        appendVertex(v, p, &nrm[k], alb, material);
    };
    for (int j = 0; j < nz; j++) {
        for (int i = 0; i < nx; i++) {
            const float h00 = hgt[static_cast<size_t>(j) * nxv + i], h10 = hgt[static_cast<size_t>(j) * nxv + i + 1];
            const float h01 = hgt[static_cast<size_t>(j + 1) * nxv + i], h11 = hgt[static_cast<size_t>(j + 1) * nxv + i + 1];
            const float avg = 0.25f * (h00 + h10 + h01 + h11);
            // Sand on the shelf, the shore and the dune; rock only on any ground well above the dune.
            const bool isSand = avg < 6.0f;
            const float mat = isSand ? static_cast<float>(Sand) : static_cast<float>(Rock);
            const float* alb = isSand ? sand : rock;
            vert(i, j, mat, alb); vert(i + 1, j, mat, alb); vert(i + 1, j + 1, mat, alb);
            vert(i, j, mat, alb); vert(i + 1, j + 1, mat, alb); vert(i, j + 1, mat, alb);
        }
    }
    return v;
}

}  // namespace scene
}  // namespace engine
