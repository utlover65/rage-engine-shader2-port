// RHI smoke test: draws one triangle into an offscreen target, reads it back, and checks
// the pixels. Set ENGINE_VALIDATION=1 to run with the Khronos validation layer.
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "rhi/rhi.h"

int main(int argc, char** argv) {
    using namespace engine::rhi;
    const uint32_t W = 256, H = 256;
    const std::string outPath = argc > 1 ? argv[1] : "triangle.ppm";
    const std::string shaderDir = ENGINE_SHADER_DIR;

    try {
        DeviceOptions opts;
        opts.validation = std::getenv("ENGINE_VALIDATION") != nullptr;
        auto dev = createDevice(Backend::Vulkan, opts);
        std::printf("backend: %s, device: %s\n", dev->backendName(), dev->deviceName().c_str());

        auto target = dev->createImage({W, H, Format::RGBA8_UNORM, true, false, false, true, "triangle target"});
        auto readback = dev->createBuffer(static_cast<size_t>(W) * H * 4, BufferUsage::Readback);
        auto vs = dev->createShader(shaderDir + "/tri.vert.spv");
        auto fs = dev->createShader(shaderDir + "/tri.frag.spv");

        PipelineDesc pd;
        pd.name = "triangle";
        pd.vertex = vs.get();
        pd.fragment = fs.get();
        pd.colorFormat = Format::RGBA8_UNORM;
        auto pipe = dev->createPipeline(pd);

        auto cmd = dev->createCommandList();
        cmd->begin();
        cmd->transition(*target, ImageState::ColorTarget);
        cmd->beginRendering(*target, LoadOp::Clear, {0.05f, 0.05f, 0.10f, 1.0f}, nullptr);
        cmd->bindPipeline(*pipe);
        cmd->draw(3);
        cmd->endRendering();
        cmd->transition(*target, ImageState::TransferSrc);
        cmd->copyImageToBuffer(*target, *readback);
        cmd->end();
        dev->submitAndWait(*cmd);

        const uint8_t* px = static_cast<const uint8_t*>(readback->data());
        FILE* f = std::fopen(outPath.c_str(), "wb");
        if (!f) throw Error("cannot write " + outPath);
        std::fprintf(f, "P6\n%u %u\n255\n", W, H);
        for (uint32_t i = 0; i < W * H; i++) {
            std::fputc(px[i * 4 + 0], f);
            std::fputc(px[i * 4 + 1], f);
            std::fputc(px[i * 4 + 2], f);
        }
        std::fclose(f);

        auto at = [&](uint32_t x, uint32_t y) { return px + (y * W + x) * 4; };
        const uint8_t* centre = at(W / 2, H / 2 + 20);  // inside the triangle
        const uint8_t* corner = at(2, 2);               // outside it
        const bool centreDrawn = centre[0] != 13 || centre[1] != 13 || centre[2] != 26;
        const bool cornerClear = corner[0] == 13 && corner[1] == 13 && corner[2] == 26;
        std::printf("centre %d %d %d, corner %d %d %d, validation errors %u\n", centre[0], centre[1], centre[2],
                    corner[0], corner[1], corner[2], dev->validationErrorCount());
        if (!centreDrawn || !cornerClear) {
            std::fprintf(stderr, "pixel check failed\n");
            return 1;
        }
        if (dev->validationErrorCount() != 0) {
            std::fprintf(stderr, "validation reported errors\n");
            return 1;
        }
        std::printf("PASS: triangle rendered and read back through the RHI\n");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "engine_triangle: %s\n", e.what());
        return 1;
    }
}
