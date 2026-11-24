// Headless ocean test on the renderer's grid (N = 256, L = 1024 m, V = 10 m/s):
//   1. the GPU height and horizontal displacements at a 16 x 16 lattice of grid points, against a direct
//      double-precision sum of the same equations (Tessendorf, 2001);
//   2. the RMS height of the whole grid (Parseval) against the target sigma the amplitude was set for;
//   3. zero validation errors.
// Usage: engine_ocean_test <ocean_evolve.comp.spv> <fft2d_stage.comp.spv>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <utility>
#include <vector>

#include "ocean/ocean_spectrum.h"
#include "rhi/rhi.h"

using namespace engine::rhi;
using engine::ocean::OceanParams;
using cd = std::complex<double>;

int main(int argc, char** argv) {
    if (argc < 3) {
        std::printf("usage: engine_ocean_test <ocean_evolve.spv> <fft2d_stage.spv>\n");
        return 2;
    }
    const OceanParams p = engine::ocean::defaultParams();
    const uint32_t N = p.N;
    const uint32_t log2N = engine::ocean::kOceanLog2N;
    const float t = 1.7f;

    std::vector<float> h0;
    engine::ocean::buildH0(p, h0, 1234u);

    DeviceOptions opts;
    opts.validation = true;
    auto dev = createDevice(Backend::Vulkan, opts);
    auto evShader = dev->createShader(argv[1]);
    auto fftShader = dev->createShader(argv[2]);
    ComputePipelineDesc evDesc;
    evDesc.name = "ocean evolve";
    evDesc.compute = evShader.get();
    evDesc.pushConstantBytes = 20;  // t, L, g, N, mode
    auto evPipe = dev->createComputePipeline(evDesc);
    ComputePipelineDesc fftDesc;
    fftDesc.name = "fft2d";
    fftDesc.compute = fftShader.get();
    fftDesc.pushConstantBytes = 12;  // Ns, N, mode
    auto fftPipe = dev->createComputePipeline(fftDesc);

    const size_t bytes = static_cast<size_t>(N) * N * 2 * sizeof(float);
    auto bufH0 = dev->createBuffer(bytes, BufferUsage::Storage);
    auto bufA = dev->createBuffer(bytes, BufferUsage::Storage);
    auto bufB = dev->createBuffer(bytes, BufferUsage::Storage);
    for (size_t i = 0; i < h0.size(); i++) static_cast<float*>(bufH0->data())[i] = h0[i];

    auto setEvolve = dev->createDescriptorSet();  // in = h0, out = A
    auto setAB = dev->createDescriptorSet();      // in = A, out = B
    auto setBA = dev->createDescriptorSet();      // in = B, out = A
    dev->setStorageBuffer(*setEvolve, *bufH0, 0);
    dev->setStorageBuffer(*setEvolve, *bufA, 1);
    dev->setStorageBuffer(*setAB, *bufA, 0);
    dev->setStorageBuffer(*setAB, *bufB, 1);
    dev->setStorageBuffer(*setBA, *bufB, 0);
    dev->setStorageBuffer(*setBA, *bufA, 1);

    // GPU result per mode: real part, natural order (z * N + x).
    std::vector<std::vector<double>> gpu(3, std::vector<double>(static_cast<size_t>(N) * N));
    for (uint32_t mode = 0; mode < 3; mode++) {
        auto cl = dev->createCommandList();
        cl->begin();
        cl->bindComputePipeline(*evPipe);
        cl->bindDescriptorSet(*setEvolve);
        const struct { float t, L, g; uint32_t N, mode; } ev{t, p.L, p.g, N, mode};
        cl->pushConstants(&ev, sizeof(ev));
        cl->dispatch(N / 8, N / 8, 1);
        cl->computeBarrier();
        cl->bindComputePipeline(*fftPipe);
        for (uint32_t pass = 0; pass < 2 * log2N; pass++) {
            const uint32_t fftMode = pass < log2N ? 0u : 1u;
            const uint32_t Ns = 1u << (pass % log2N);
            cl->bindDescriptorSet(pass % 2 == 0 ? *setAB : *setBA);
            const uint32_t push[3] = {Ns, N, fftMode};
            cl->pushConstants(push, sizeof(push));
            cl->dispatch(N / 2 / 32, N, 1);
            cl->computeBarrier();
        }
        cl->end();
        dev->submitAndWait(*cl);
        const float* out = static_cast<const float*>(bufA->data());
        for (size_t i = 0; i < static_cast<size_t>(N) * N; i++) gpu[mode][i] = out[2 * i];
    }

    // Reference: h(k,t) on the whole grid, then a direct sum at a lattice of points.
    std::vector<cd> hk(static_cast<size_t>(N) * N);
    auto h0c = [&](uint32_t zi, uint32_t xi) {
        const size_t i = (static_cast<size_t>(zi) * N + xi) * 2;
        return cd(h0[i], h0[i + 1]);
    };
    for (uint32_t mz = 0; mz < N; mz++)
        for (uint32_t mx = 0; mx < N; mx++) {
            const double kx = engine::ocean::waveNumber(mx, N, p.L), kz = engine::ocean::waveNumber(mz, N, p.L);
            const double k = std::sqrt(kx * kx + kz * kz);
            const double w = std::sqrt(static_cast<double>(p.g) * k);
            const cd e(std::cos(w * t), std::sin(w * t));
            const cd hp = h0c(mz, mx) * e;
            const cd hm = std::conj(h0c((N - mz) % N, (N - mx) % N)) * std::conj(e);
            hk[static_cast<size_t>(mz) * N + mx] = (k < 1e-9) ? cd(0, 0) : hp + hm;
        }

    const uint32_t step = 16;
    double maxErr = 0.0, maxRef = 0.0;
    for (uint32_t mode = 0; mode < 3; mode++)
        for (uint32_t z = 0; z < N; z += step)
            for (uint32_t x = 0; x < N; x += step) {
                const double X = static_cast<double>(x) * p.L / N, Z = static_cast<double>(z) * p.L / N;
                cd sum = 0;
                for (uint32_t mz = 0; mz < N; mz++)
                    for (uint32_t mx = 0; mx < N; mx++) {
                        const double kx = engine::ocean::waveNumber(mx, N, p.L), kz = engine::ocean::waveNumber(mz, N, p.L);
                        const double k = std::sqrt(kx * kx + kz * kz);
                        if (k < 1e-9) continue;
                        cd v = hk[static_cast<size_t>(mz) * N + mx];
                        if (mode == 1) v *= cd(0, -kx / k);
                        if (mode == 2) v *= cd(0, -kz / k);
                        sum += v * std::polar(1.0, kx * X + kz * Z);
                    }
                const double ref = sum.real();
                const double got = gpu[mode][static_cast<size_t>(z) * N + x];
                maxErr = std::max(maxErr, std::fabs(got - ref));
                maxRef = std::max(maxRef, std::fabs(ref));
            }

    double ss = 0.0;
    for (double v : gpu[0]) ss += v * v;
    const double rms = std::sqrt(ss / static_cast<double>(gpu[0].size()));
    const double target = engine::ocean::kOceanSigmaM;
    const double ratio = rms / target;
    const double rel = maxErr / maxRef;
    const uint32_t valErr = dev->validationErrorCount();
    std::printf("ocean N=%u L=%.0f m at t=%.2f s: lattice max error %.3e, peak %.3e (relative %.3e)\n", N, p.L, t,
                maxErr, maxRef, rel);
    std::printf("RMS height %.4f m vs target %.4f m (ratio %.4f), A = %.4e, validation errors %u\n", rms, target,
                ratio, p.A, valErr);
    const bool ok = maxRef > 1e-6 && rel < 2e-3 && std::fabs(ratio - 1.0) < 0.06 && valErr == 0;
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
