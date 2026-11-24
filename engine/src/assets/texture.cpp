#include "assets/texture.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iterator>
#include <vector>

namespace engine {
namespace assets {

namespace {

std::vector<uint8_t> readFile(const std::string& path, size_t expectedBytes) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw rhi::Error("cannot open texture '" + path + "'");
    std::vector<uint8_t> data((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (data.size() != expectedBytes)
        throw rhi::Error("texture '" + path + "' has " + std::to_string(data.size()) + " bytes, expected " +
                         std::to_string(expectedBytes));
    return data;
}

uint32_t mipCount(uint32_t w, uint32_t h) {
    uint32_t n = 1;
    uint32_t m = std::max(w, h);
    while (m > 1) {
        m >>= 1;
        n++;
    }
    return n;
}

// Packs all levels, level 0 first, each level max(1, w >> k) by max(1, h >> k) RGBA8 texels.
std::vector<uint8_t> buildMipChain(const std::vector<uint8_t>& base, uint32_t w, uint32_t h, uint32_t levels) {
    std::vector<uint8_t> chain(base);
    uint32_t sw = w, sh = h;
    for (uint32_t level = 1; level < levels; level++) {
        const uint32_t dw = std::max(1u, sw >> 1), dh = std::max(1u, sh >> 1);
        const size_t old = chain.size();
        chain.resize(old + static_cast<size_t>(dw) * dh * 4);
        uint8_t* dst = chain.data() + old;
        const uint8_t* s = chain.data() + (old - static_cast<size_t>(sw) * sh * 4);
        for (uint32_t y = 0; y < dh; y++) {
            const uint32_t y0 = std::min(2 * y, sh - 1), y1 = std::min(2 * y + 1, sh - 1);
            for (uint32_t x = 0; x < dw; x++) {
                const uint32_t x0 = std::min(2 * x, sw - 1), x1 = std::min(2 * x + 1, sw - 1);
                const uint8_t* p00 = s + 4 * (static_cast<size_t>(y0) * sw + x0);
                const uint8_t* p10 = s + 4 * (static_cast<size_t>(y0) * sw + x1);
                const uint8_t* p01 = s + 4 * (static_cast<size_t>(y1) * sw + x0);
                const uint8_t* p11 = s + 4 * (static_cast<size_t>(y1) * sw + x1);
                for (int c = 0; c < 4; c++) {
                    const int sum = p00[c] + p10[c] + p01[c] + p11[c];
                    dst[4 * (static_cast<size_t>(y) * dw + x) + c] = static_cast<uint8_t>((sum + 2) / 4);
                }
            }
        }
        sw = dw;
        sh = dh;
    }
    return chain;
}

}  // namespace

std::unique_ptr<rhi::Image> loadRawTexture(rhi::Device& dev, const std::string& path, uint32_t width,
                                           uint32_t height, const std::string& name) {
    const size_t bytes = static_cast<size_t>(width) * height * 4;
    const std::vector<uint8_t> base = readFile(path, bytes);
    const uint32_t levels = mipCount(width, height);
    const std::vector<uint8_t> chain = buildMipChain(base, width, height, levels);

    rhi::ImageDesc desc;
    desc.width = width;
    desc.height = height;
    desc.format = rhi::Format::RGBA8_UNORM;
    desc.sampled = true;
    desc.name = name;
    desc.mipLevels = levels;
    auto image = dev.createImage(desc);

    auto staging = dev.createBuffer(chain.size(), rhi::BufferUsage::Staging);
    std::memcpy(staging->data(), chain.data(), chain.size());

    auto cmd = dev.createCommandList();
    cmd->begin();
    cmd->transition(*image, rhi::ImageState::TransferDst);
    cmd->copyBufferToImage(*staging, *image);
    cmd->transition(*image, rhi::ImageState::ShaderRead);
    cmd->end();
    dev.submitAndWait(*cmd);
    return image;
}

}  // namespace assets
}  // namespace engine
