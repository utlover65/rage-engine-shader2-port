// Headless compute test: dispatches compute_fill.comp into a storage buffer, reads it back and checks
// every element. Exit 0 on pass. Usage: engine_compute_test <path to compute_fill.comp.spv>
#include <cstdint>
#include <cstdio>

#include "rhi/rhi.h"

using namespace engine::rhi;

int main(int argc, char** argv) {
    if (argc < 2) { std::printf("usage: engine_compute_test <compute_fill.comp.spv>\n"); return 2; }
    DeviceOptions opts;
    opts.validation = true;
    auto dev = createDevice(Backend::Vulkan, opts);
    auto shader = dev->createShader(argv[1]);
    ComputePipelineDesc desc;
    desc.name = "fill";
    desc.compute = shader.get();
    desc.pushConstantBytes = 4;
    auto pipe = dev->createComputePipeline(desc);

    const uint32_t n = 256, k = 7;
    auto buf = dev->createBuffer(n * sizeof(uint32_t), BufferUsage::Storage);
    auto set = dev->createDescriptorSet();
    dev->setStorageBuffer(*set, *buf, 0);

    auto cl = dev->createCommandList();
    cl->begin();
    cl->bindComputePipeline(*pipe);
    cl->bindDescriptorSet(*set);
    cl->pushConstants(&k, sizeof(k));
    cl->dispatch(n / 64, 1, 1);
    cl->end();
    dev->submitAndWait(*cl);

    const auto* out = static_cast<const uint32_t*>(buf->data());
    uint32_t bad = 0;
    for (uint32_t i = 0; i < n; i++) if (out[i] != 3u * i + k) bad++;
    const uint32_t valErr = dev->validationErrorCount();
    std::printf("compute: %u of %u elements wrong, validation errors %u\n", bad, n, valErr);
    const bool ok = bad == 0 && valErr == 0;
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
