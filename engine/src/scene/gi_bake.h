// Baked ambient lighting: a grid of probes, each holding six "ambient cube" faces (average
// radiance arriving from +x, -x, +y, -y, +z, -z). Built on the CPU by ray casting the scene and
// the beach, so it captures sky occlusion (contact darkening under the pier and lighthouse)
// and one bounce of sky light with colour bleeding from the surfaces the rays hit.
#pragma once

#include <cstdint>
#include <vector>

namespace engine {
namespace scene {

// The sky-view LUT read back from the GPU: RGBA, linear, pre-exposed (the same values the
// shaders see), in the LUT's own parametrisation.
struct SkyTable {
    int width = 0, height = 0;
    std::vector<float> rgba;
};

struct ProbeSet {
    int dims[3] = {0, 0, 0};
    float origin[3] = {0, 0, 0};  // world position of probe (0, 0, 0)
    float step[3] = {1, 1, 1};    // spacing along x, y, z (m)
    // Six faces per probe, each a vec4 (rgb, unused w), probe index x + dx*(y + dy*z).
    std::vector<float> faces;
    int probeCount() const { return dims[0] * dims[1] * dims[2]; }
};

// Decodes RGBA16F texels (8 bytes each, little-endian) to floats.
std::vector<float> halfRgbaToFloat(const std::vector<uint8_t>& bytes);

// Bakes the probe grid. `objects` is the scene triangle list (10 floats per vertex, triangles of
// three consecutive vertices). The terrain is ray-marched through scene::landHeight.
ProbeSet bakeAmbientProbes(const std::vector<float>& objects, const SkyTable& sky);

}  // namespace scene
}  // namespace engine
