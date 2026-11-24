// Polar ring mesh of the sea surface, centred on the camera. Rings grow geometrically, so
// vertices are dense near the camera and sparse toward the horizon. The rest-plane
// coordinates are offsets from the camera; the vertex shader adds the camera position and
// applies the waves.
#pragma once

#include <cstdint>
#include <vector>

namespace engine {
namespace scene {

struct OceanMesh {
    std::vector<float> rest;       // x, y per vertex: rest-plane offset from the camera (m)
    std::vector<uint32_t> indices; // triangle list
    uint32_t vertexCount() const { return static_cast<uint32_t>(rest.size() / 2); }
    uint32_t indexCount() const { return static_cast<uint32_t>(indices.size()); }
};

// Angular segments per ring (about 0.18 degrees each).
constexpr uint32_t kOceanSegments = 2048;

OceanMesh buildOceanMesh();

}  // namespace scene
}  // namespace engine
