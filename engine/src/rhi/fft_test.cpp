// Headless FFT test: a 256-point complex FFT run as eight Stockham compute passes on the GPU,
// checked against a double-precision naive DFT. Exit 0 on pass.
// Usage: engine_fft_test <path to fft_stage.comp.spv>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "rhi/rhi.h"

using namespace engine::rhi;
using cd = std::complex<double>;

int main(int argc, char** argv) {
    if (argc < 2) { std::printf("usage: engine_fft_test <fft_stage.comp.spv>\n"); return 2; }
    const uint32_t N = 256;
    DeviceOptions opts;
    opts.validation = true;
    auto dev = createDevice(Backend::Vulkan, opts);
    auto shader = dev->createShader(argv[1]);
    ComputePipelineDesc desc;
    desc.name = "fft_stage";
    desc.compute = shader.get();
    desc.pushConstantBytes = 8;
    auto pipe = dev->createComputePipeline(desc);

    auto bufA = dev->createBuffer(N * sizeof(float) * 2, BufferUsage::Storage);
    auto bufB = dev->createBuffer(N * sizeof(float) * 2, BufferUsage::Storage);
    // Ping-pong sets: even passes read A and write B, odd passes read B and write A.
    auto setAB = dev->createDescriptorSet();
    auto setBA = dev->createDescriptorSet();
    dev->setStorageBuffer(*setAB, *bufA, 0);
    dev->setStorageBuffer(*setAB, *bufB, 1);
    dev->setStorageBuffer(*setBA, *bufB, 0);
    dev->setStorageBuffer(*setBA, *bufA, 1);

    // Deterministic input in [-1, 1].
    std::vector<cd> x(N);
    uint32_t seed = 12345u;
    auto rnd = [&seed]() {
        seed = seed * 1664525u + 1013904223u;
        return static_cast<double>(seed >> 8) / static_cast<double>(1u << 24) * 2.0 - 1.0;
    };
    float* in = static_cast<float*>(bufA->data());
    for (uint32_t i = 0; i < N; i++) {
        x[i] = {rnd(), rnd()};
        in[2 * i] = static_cast<float>(x[i].real());
        in[2 * i + 1] = static_cast<float>(x[i].imag());
    }

    auto cl = dev->createCommandList();
    cl->begin();
    cl->bindComputePipeline(*pipe);
    uint32_t pass = 0;
    for (uint32_t ns = 1; ns < N; ns <<= 1, pass++) {
        cl->bindDescriptorSet(pass % 2 == 0 ? *setAB : *setBA);
        const uint32_t push[2] = {ns, N};
        cl->pushConstants(push, sizeof(push));
        cl->dispatch((N / 2 + 63) / 64, 1, 1);
        cl->computeBarrier();
    }
    cl->end();
    dev->submitAndWait(*cl);

    // Eight passes (even count) leave the result in bufA.
    const float* out = static_cast<const float*>(bufA->data());
    double maxErr = 0.0, maxMag = 0.0;
    for (uint32_t m = 0; m < N; m++) {
        cd ref = 0;
        for (uint32_t n = 0; n < N; n++) ref += x[n] * std::polar(1.0, -2.0 * M_PI * double(n) * double(m) / N);
        const cd got(out[2 * m], out[2 * m + 1]);
        maxErr = std::max(maxErr, std::abs(got - ref));
        maxMag = std::max(maxMag, std::abs(ref));
    }
    const double rel = maxErr / maxMag;
    const uint32_t valErr = dev->validationErrorCount();
    std::printf("fft256: max error %.3e (relative %.3e), validation errors %u\n", maxErr, rel, valErr);
    const bool ok = rel < 1e-4 && valErr == 0;
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
