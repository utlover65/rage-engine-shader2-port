// Loads raw RGBA8 texture files (see tools/bake_materials.py) as sampled, mip-mapped images.
#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "rhi/rhi.h"

namespace engine {
namespace assets {

// Reads width*height*4 bytes from `path`, builds the full mip chain on the CPU (2x2 box filter),
// uploads every level and leaves the image in the ShaderRead state. Throws rhi::Error on failure.
std::unique_ptr<rhi::Image> loadRawTexture(rhi::Device& dev, const std::string& path, uint32_t width,
                                           uint32_t height, const std::string& name);

}  // namespace assets
}  // namespace engine
