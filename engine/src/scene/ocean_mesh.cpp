#include "scene/ocean_mesh.h"

#include <cmath>

namespace engine {
namespace scene {

OceanMesh buildOceanMesh() {
    // Ring radii: fine ratio where waves are resolved (to 1.5 km), then coarse out to 100 km
    // so the mesh reaches past the horizon. Beyond 1.5 km the waves are fully faded (LOD), so
    // the far rings only carry the flat sea.
    std::vector<double> radii;
    for (double r = 0.3; r < 1500.0; r *= 1.04) radii.push_back(r);
    for (double r = radii.empty() ? 0.3 : radii.back() * 1.25; r < 1.0e5; r *= 1.25) radii.push_back(r);
    const uint32_t rings = static_cast<uint32_t>(radii.size());
    const uint32_t M = kOceanSegments;

    OceanMesh mesh;
    mesh.rest.reserve((1 + static_cast<size_t>(rings) * M) * 2);
    mesh.rest.push_back(0.0f);  // centre vertex
    mesh.rest.push_back(0.0f);
    for (uint32_t j = 0; j < rings; j++)
        for (uint32_t k = 0; k < M; k++) {
            const double a = 2.0 * M_PI * k / M;
            mesh.rest.push_back(static_cast<float>(radii[j] * std::cos(a)));
            mesh.rest.push_back(static_cast<float>(radii[j] * std::sin(a)));
        }

    auto ringVertex = [&](uint32_t j, uint32_t k) -> uint32_t { return 1 + j * M + (k % M); };
    mesh.indices.reserve(static_cast<size_t>(rings) * M * 6 + M * 3);
    // Centre fan to the first ring.
    for (uint32_t k = 0; k < M; k++) {
        mesh.indices.push_back(0);
        mesh.indices.push_back(ringVertex(0, k));
        mesh.indices.push_back(ringVertex(0, k + 1));
    }
    // Quads between consecutive rings.
    for (uint32_t j = 0; j + 1 < rings; j++)
        for (uint32_t k = 0; k < M; k++) {
            const uint32_t a = ringVertex(j, k), b = ringVertex(j, k + 1);
            const uint32_t c = ringVertex(j + 1, k), d = ringVertex(j + 1, k + 1);
            mesh.indices.insert(mesh.indices.end(), {a, c, d, a, d, b});
        }
    return mesh;
}

}  // namespace scene
}  // namespace engine
