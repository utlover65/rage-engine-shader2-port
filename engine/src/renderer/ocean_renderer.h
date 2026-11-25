// Real-time open-sea renderer built on the RHI and the render graph. Passes, in order:
//   1. transmittance and multiple-scattering LUTs (once)
//   2. sky-view LUT (per frame)
//   3. cloud slab and cloud-shadow field (half resolution)
//   4. god rays: in-scatter removed by cloud shadows (quarter resolution)
//   5. sky composite -> HDR
//   6. sun shadow map (scene objects and headland from the sun)
//   7. planar reflection: objects and headland above the sea, from a mirrored camera (half res)
//   8. ocean surface and scene objects (depth tested, shadowed, probe-lit) -> HDR
//   9. bloom, blur and filmic tone map -> display image
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "math/vec_mat.h"
#include "scene/gi_bake.h"
#include "rg/render_graph.h"
#include "ocean/ocean_spectrum.h"
#include "rhi/rhi.h"

namespace engine {
namespace renderer {

struct Settings {
    uint32_t width = 960;
    uint32_t height = 540;
    uint32_t shadowMapSize = 2048;
    // Exposure value. 15 ("Sunny 16") suits a midday sun. The default sun is 5 degrees above the
    // horizon (about 15,000 lux at sea level, red-shifted; see sunIlluminanceLux), so the default
    // exposure is for a low sunset sun. Higher values are darker; the sky must stay below the ACES
    // shoulder. At 5 degrees, sun-blocking clouds sit about 14 km away, inside the cloud pass's range.
    // The earlier 1.2-degree look is ENGINE_SUN_ELEV_DEG=1.2 ENGINE_EV100=10.5.
    double ev100 = 13.5;  // darker than the earlier 11.5: the sea and sky keep their tonal range
    double fovYDeg = 50.0;
    math::Vec3 camPos{0.0, 2.2, -120.0};
    math::Vec3 camTarget{0.0, 1.2, -400.0};
    math::Vec3 sunDir{0.05, 0.21, -0.97};  // unnormalised; about 12 degrees elevation, clear of the skyline
};

class OceanRenderer {
public:
    // shaderDir holds the compiled .spv files; assetDir holds the raw texture files.
    OceanRenderer(rhi::Device& device, const std::string& shaderDir, const std::string& assetDir,
                  const Settings& settings);
    ~OceanRenderer();
    OceanRenderer(const OceanRenderer&) = delete;
    OceanRenderer& operator=(const OceanRenderer&) = delete;

    // Runs the whole graph for the current settings and waits for the GPU.
    void renderFrame(double timeSeconds);

    // Copies an image to host memory: RGBA8 is 4 bytes per pixel, RGBA16F is 8.
    std::vector<uint8_t> readImage(rhi::Image& image, size_t bytesPerPixel);

    rhi::Image& display() { return graph_.image(hLdr_); }
    rhi::Image& hdr() { return graph_.image(hHdr_); }
    rhi::Image& transmittance() { return graph_.image(hTrans_); }
    rhi::Image& skyView() { return graph_.image(hSky_); }
    rhi::Image& multiScatter() { return graph_.image(hMs_); }
    rhi::Image& bloom() { return graph_.image(hBloomA_); }
    rhi::Image& shadowMap() { return graph_.image(hShadow_); }
    rhi::Image& godRays() { return graph_.image(hShaft_); }
    rhi::Image& skyPrefilter() { return graph_.image(hPre_); }
    rhi::Image& aerialInScatter() { return graph_.image(hAerialL_); }
    rhi::Image& aerialTransmittance() { return graph_.image(hAerialT_); }
    rhi::Image& planarReflection() { return graph_.image(hRefl_); }

    const Settings& settings() const { return settings_; }
    double preExposure() const { return preExposure_; }
    std::string describeGraph() const { return graph_.describe(); }
    // Sun illuminance on a sun-facing surface at sea level (lux, per channel), before exposure.
    math::Vec3 sunIlluminanceLux() const { return sunLux_; }
    math::Vec3 sunDirUnit() const { return sunUnit_; }

private:
    struct FrameData;  // std140 uniform block, defined in the .cpp
    void bakeProbes();
    void updateDynamicProbes();  // DDGI: refresh dynamic probe irradiance each frame

    rhi::Device& dev_;
    Settings settings_;
    double preExposure_ = 1.0;
    math::Vec3 sunUnit_{};
    math::Vec3 sunLux_{};
    math::Mat4 prevViewProj_{};  // last frame's unjittered view-projection
    uint32_t frameIndex_ = 0;
    bool historyValid_ = false;

    // Shaders and pipelines.
    std::unique_ptr<rhi::Shader> fullscreenVs_, oceanVs_;
    std::unique_ptr<rhi::Shader> transFs_, msFs_, skyViewFs_, cloudFs_, cloudShadowFs_, skyFs_, oceanFs_;
    std::unique_ptr<rhi::Shader> brightFs_, blurFs_, toneFs_, shaftFs_;
    std::unique_ptr<rhi::Shader> dofPrepareFs_, dofBlurFs_, dofCompositeFs_, lumDownsampleFs_, lumReduceFs_;
    std::unique_ptr<rhi::Shader> meshVs_, meshFs_, shadowVs_, shadowFs_;
    std::unique_ptr<rhi::Pipeline> transPipe_, msPipe_, skyViewPipe_, cloudPipe_, cloudShadowPipe_, skyPipe_, oceanPipe_;
    std::unique_ptr<rhi::Pipeline> brightPipe_, blurPipe_, tonePipe_, shaftPipe_, prefilterPipe_;
    std::unique_ptr<rhi::Pipeline> dofPreparePipe_, dofBlurPipe_, dofCompositePipe_, lumDownsamplePipe_, lumReducePipe_;
    std::unique_ptr<rhi::Shader> prefilterFs_, aerialFs_;
    std::unique_ptr<rhi::Pipeline> aerialPipe_;
    std::unique_ptr<rhi::Shader> depthFs_, gtaoFs_, aoBlurFs_, taaFs_, copyFs_;
    std::unique_ptr<rhi::Pipeline> taaPipe_, copyPipe_;
    std::unique_ptr<rhi::Pipeline> depthMeshPipe_, depthOceanPipe_, gtaoPipe_, aoBlurPipe_;
    std::unique_ptr<rhi::Pipeline> meshPipe_, shadowPipe_;

    // Descriptor sets, one per pass (each pass reads a fixed set of images).
    std::unique_ptr<rhi::DescriptorSet> dTrans_, dMs_, dSkyView_, dCloud_, dCloudShadow_, dSky_, dOcean_;
    std::unique_ptr<rhi::DescriptorSet> dBright_, dBlurH_, dBlurV_, dTone_, dMesh_, dMeshRefl_, dShadow_, dShaft_, dPrefilter_, dAerial_, dShadowNear_, dShadowFar_, dDepthPre_, dGtao_, dAoBlur_, dTaa_, dCopy_;
    std::unique_ptr<rhi::DescriptorSet> dDofPrepare_, dDofBlurH_, dDofBlurV_, dDofComposite_, dLumDownsample_, dLumReduce_;

    // Scanned materials (wood and rock diffuse, with derived normal and height maps).
    std::unique_ptr<rhi::Image> woodAlbedo_, woodNormal_, rockAlbedo_, rockNormal_;

    // Buffers.
    std::unique_ptr<rhi::Buffer> ubo_, uboRefl_, uboShadowNear_, uboShadowFar_, oceanVb_, oceanIb_;
    uint32_t oceanIndexCount_ = 0;
    std::unique_ptr<rhi::Buffer> objVb_;   // scene objects, triangle list (see scene/objects.h)
    uint32_t objVertexCount_ = 0;
    std::unique_ptr<rhi::Buffer> terrVb_;  // headland (see scene/landscape.h)
    uint32_t terrVertexCount_ = 0;
    scene::ProbeSet probes_;               // baked ambient cubes: 64 probes x 6 faces x vec4
    float dynamicProbes_[64 * 8] = {};     // DDGI: 64 probes x 2 vec4s (position+validity, irradiance+pad)

    rg::Graph graph_;
    rg::Handle hTrans_ = 0, hMs_ = 0, hSky_ = 0, hCloud_ = 0, hCloudShadow_ = 0;
    rg::Handle hHdr_ = 0, hDepth_ = 0, hLdr_ = 0;
    rg::Handle hBloomA_ = 0, hBloomB_ = 0;
    rg::Handle hDofPrep_ = 0, hDofH_ = 0, hDofV_ = 0, hDofOut_ = 0;  // adaptive DOF chain
    rg::Handle hLumA_ = 0, hLumB_ = 0;  // auto-exposure luminance reduction
    rg::Handle hShadow_ = 0, hShadowDepth_ = 0;
    rg::Handle hShaft_ = 0;  // god rays, quarter resolution
    rg::Handle hPre_ = 0;    // prefiltered sky: three stacked rows of the sky-view LUT
    rg::Handle hAerialL_ = 0, hAerialT_ = 0;  // aerial-perspective froxel atlases (in-scatter, transmittance)
    rg::Handle hResolved_ = 0, hHist_ = 0;  // TAA: resolved colour, history (imported, see histImage_)
    std::unique_ptr<rhi::Image> histImage_;
    rg::Handle hLinDepth_ = 0, hDepthPre_ = 0, hAO_ = 0, hAOB_ = 0;  // linear depth (prepass), AO (raw, blurred)
    rg::Handle hShadowNear_ = 0, hShadowNearDepth_ = 0, hShadowFar_ = 0, hShadowFarDepth_ = 0;  // cascades
    rg::Handle hRefl_ = 0, hReflDepth_ = 0;  // planar reflection, half resolution

    // FFT ocean. A compute pass before the depth prepass evolves the spectrum and runs the 2-D FFT for the
    // height and the two horizontal displacements; the ocean vertex stage reads them as storage slots 0..2.
    std::unique_ptr<rhi::Shader> oceanEvolveCs_, fftCs_;
    std::unique_ptr<rhi::Pipeline> oceanEvolvePipe_, fftPipe_;
    std::unique_ptr<rhi::Buffer> oceanH0_, oceanPing_;
    std::unique_ptr<rhi::Buffer> oceanSpec_[3];  // height, Dx, Dz after the FFT
    std::unique_ptr<rhi::DescriptorSet> dOceanEvolve_[3], dOceanFftAB_[3], dOceanFftBA_[3];
    double oceanTime_ = 0.0;
    void recordOceanFft(rhi::CommandList& c);
    // Capillary cascade: a second 96 m patch of short waves (see ocean_spectrum.h). Its height field is read in the
    // ocean fragment stage as storage slot 3, and its slopes perturb the normal near the camera.
    std::unique_ptr<rhi::Buffer> oceanH0Cap_, oceanCap_;
    std::unique_ptr<rhi::DescriptorSet> dOceanEvolveCap_, dOceanFftABCap_, dOceanFftBACap_;
    bool capOn_ = true;          // ENGINE_SEA_CAP=0 turns the capillary cascade off (the Pass 33 sea)
    bool profile_ = false;
    std::vector<std::pair<std::string, double>> passMs_;
    // Auto-exposure: smoothed gain applied in the tone map (uExposure.w). 1.0 when disabled.
    bool autoExposureOn_ = true;
    float autoExposure_ = 1.0f;
    // Adaptive DOF lens parameters (focal length mm, f-number, focus distance m, sensor height mm).
    float dofFocalMm_ = 35.0f;
    float dofAperture_ = 2.8f;
    float dofFocusM_ = 80.0f;
    float dofSensorH_ = 24.0f;
    bool cloudLayersOn_ = true;  // ENGINE_CLOUD_LAYERS=0 turns the stratus and altostratus sun shadows off

    std::unique_ptr<rhi::CommandList> cmd_;
    double lastFrameMs_ = 0.0;

public:
    double lastFrameMs() const { return lastFrameMs_; }
    // GPU time of each graph pass in the last frame (ms), in graph order. Filled only with ENGINE_PROFILE set on a
    // device with timestamps; otherwise empty.
    std::vector<std::pair<std::string, double>> lastPassTimesMs() const { return passMs_; }
};

}  // namespace renderer
}  // namespace engine
