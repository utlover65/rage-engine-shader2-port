#include <cmath>
#include <cstdio>
#include <vector>
#include "scene/objects.h"

int main() {
    const auto vertices = engine::scene::buildSceneObjects();
    if (vertices.empty() || vertices.size() % 10 != 0) {
        std::fprintf(stderr, "invalid scene vertex layout\n");
        return 2;
    }
    size_t nonFinite = 0;
    size_t rockVertices = 0, sandVertices = 0;
    double normalError = 0.0;
    for (size_t i = 0; i < vertices.size(); i += 10) {
        for (size_t j = 0; j < 10; ++j)
            if (!std::isfinite(vertices[i + j])) ++nonFinite;
        const double nx = vertices[i + 3], ny = vertices[i + 4], nz = vertices[i + 5];
        normalError += std::fabs(std::sqrt(nx * nx + ny * ny + nz * nz) - 1.0);
        const int material = static_cast<int>(vertices[i + 9] + 0.5f);
        if (material == engine::scene::Rock) ++rockVertices;
        if (material == engine::scene::Sand) ++sandVertices;
    }
    const size_t count = vertices.size() / 10;
    const double meanError = normalError / static_cast<double>(count);
    std::printf("vertices=%zu floats=%zu non_finite=%zu mean_normal_error=%.8g rock=%zu sand=%zu\n",
                count, vertices.size(), nonFinite, meanError, rockVertices, sandVertices);
    return nonFinite == 0 && meanError < 1e-4 && rockVertices > 0 && sandVertices > 0 ? 0 : 3;
}
