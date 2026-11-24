// Beach and shallows: a sand shore running roughly east-west, a dry dune rising gently inland, and
// a shelf that shelves seaward to deep water. The same analytic height function drives the mesh
// (geometric displacement), the baked lighting ray march, and the seabed shading in ocean.frag, so
// all three agree. Keep the GLSL port (ocean.frag, beachHeight) identical to landHeight.
#pragma once

#include <vector>

namespace engine {
namespace scene {

// Terrain mesh extent (m). Covers the camera's view of the beach with margin.
constexpr float kLandX0 = -260.0f, kLandX1 = 260.0f;
constexpr float kLandZ0 = -240.0f, kLandZ1 = 160.0f;

// Shoreline z (m) at world x: the waterline, sea towards -z, land towards +z.
float shoreZ(float x);

// Ground height (m, sea level at 0) at world (x, z).
float landHeight(float x, float z);

// Triangle list (10 floats per vertex, same layout as buildSceneObjects) over the terrain extent,
// with square cells of cellSize metres. Sand everywhere on the beach and the shelf.
std::vector<float> buildLandscape(float cellSize = 2.5f);

}  // namespace scene
}  // namespace engine
