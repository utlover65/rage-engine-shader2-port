#include "renderer/ocean_renderer.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <utility>

#include "assets/texture.h"
#include "atmo/atmo_reference.h"
#include "scene/gi_bake.h"
#include "scene/landscape.h"
#include "scene/objects.h"
#include "scene/ocean_mesh.h"

namespace engine {
namespace renderer {

// Layout must match FrameData in shaders/common.glsl (std140: mat4 then vec4s).
struct alignas(16) OceanRenderer::FrameData {
    float viewProj[16];
    float invViewProj[16];
    float camPos[4];
    float sunDir[4];
    float sunIlluminance[4];
    float sunRadiance[4];
    float exposure[4];
    float time[4];
    float sunViewProj[16];
    float reflViewProj[16];
    float probeOrigin[4];
    float probeStep[4];
    float probeDims[4];
    float probes[384 * 4];  // 64 probes x 6 faces x vec4 (uProbes[384] in common.glsl)
    float mainViewProj[16];
    float sunNearViewProj[16];
    float sunFarViewProj[16];
    float prevViewProj[16];
    float taa[4];
    float dof[4];  // focal length (mm), f-number, focus distance (m), sensor height (mm)
    float uWeather[4];  // weather: x=rain, y=wetness, z=flood, w=foam
    float dynamicProbes[64 * 8];  // DDGI: 64 probes x 2 vec4s (position+validity, irradiance+pad)
};

namespace {

constexpr uint32_t kTransW = 256, kTransH = 64;
constexpr uint32_t kMsSize = 32;  // multiple-scattering LUT is kMsSize x kMsSize (MS_W, MS_H)
constexpr uint32_t kSkyW = 193, kSkyH = 108;
constexpr uint32_t kAerialW = 640, kAerialH = 24;  // 16 slices of 40 x 24 texels
// Angular radius of the sun (mean solar disk).
constexpr double kSunRadiusDeg = 0.2665;
constexpr uint32_t kObjStride = 10 * sizeof(float);  // position, normal, albedo, material

// ENGINE_* switches: "0" turns a feature off; anything else, or no value, keeps the default.
bool envFlag(const char* name, bool fallback) {
    const char* v = std::getenv(name);
    return v ? std::strcmp(v, "0") != 0 : fallback;
}

using TexSet = std::array<const rhi::Image*, rhi::kMaxSampledImages>;

// Builds a texture set from (binding, image) pairs. Binding b is slot b - 1 (see rhi.h).
TexSet bindings(std::initializer_list<std::pair<uint32_t, const rhi::Image*>> list) {
    TexSet t{};
    for (const auto& entry : list) t[entry.first - 1] = entry.second;
    return t;
}

float toF(double v) { return static_cast<float>(v); }

void storeMat(float* dst, const math::Mat4& m) {
    for (int i = 0; i < 16; i++) dst[i] = toF(m.m[i]);
}

// Full-screen pass: the vertex stage generates the triangle. The push constant holds the target
// size (x, y) and a pass flag (z), read by the fragment stage as uPush.uTargetSize.
void drawFullscreen(rhi::CommandList& cmd, const rhi::Pipeline& pipe, const rhi::DescriptorSet& set,
                    rhi::Image& target, float flag = 0.0f) {
    cmd.bindPipeline(pipe);
    cmd.bindDescriptorSet(set);
    const float size[4] = {toF(target.width()), toF(target.height()), flag, 0.0f};
    cmd.pushConstants(size, sizeof(size));
    cmd.draw(3);
}

}  // namespace

OceanRenderer::OceanRenderer(rhi::Device& device, const std::string& shaderDir, const std::string& assetDir,
                             const Settings& settings)
    : dev_(device), settings_(settings) {
    sunUnit_ = math::normalize(settings.sunDir);
    preExposure_ = 1.0 / (1.2 * std::pow(2.0, settings.ev100));
    double T[3];
    atmo::transmittance(atmo::kRg, sunUnit_.y, T);
    sunLux_ = {T[0] * atmo::kSolarLux, T[1] * atmo::kSolarLux, T[2] * atmo::kSolarLux};

    // Shaders.
    fullscreenVs_ = dev_.createShader(shaderDir + "/fullscreen.vert.spv");
    oceanVs_ = dev_.createShader(shaderDir + "/ocean.vert.spv");
    transFs_ = dev_.createShader(shaderDir + "/transmittance.frag.spv");
    msFs_ = dev_.createShader(shaderDir + "/ms.frag.spv");
    skyViewFs_ = dev_.createShader(shaderDir + "/skyview.frag.spv");
    cloudFs_ = dev_.createShader(shaderDir + "/clouds.frag.spv");
    cloudShadowFs_ = dev_.createShader(shaderDir + "/cloud_shadow.frag.spv");
    shaftFs_ = dev_.createShader(shaderDir + "/shafts.frag.spv");
    prefilterFs_ = dev_.createShader(shaderDir + "/sky_prefilter.frag.spv");
    aerialFs_ = dev_.createShader(shaderDir + "/aerial.frag.spv");
    depthFs_ = dev_.createShader(shaderDir + "/depth_only.frag.spv");
    gtaoFs_ = dev_.createShader(shaderDir + "/gtao.frag.spv");
    aoBlurFs_ = dev_.createShader(shaderDir + "/ao_blur.frag.spv");
    taaFs_ = dev_.createShader(shaderDir + "/taa.frag.spv");
    copyFs_ = dev_.createShader(shaderDir + "/copy.frag.spv");
    skyFs_ = dev_.createShader(shaderDir + "/sky.frag.spv");
    oceanFs_ = dev_.createShader(shaderDir + "/ocean.frag.spv");
    oceanEvolveCs_ = dev_.createShader(shaderDir + "/ocean_evolve.comp.spv");
    fftCs_ = dev_.createShader(shaderDir + "/fft2d_stage.comp.spv");
    brightFs_ = dev_.createShader(shaderDir + "/bloom_bright.frag.spv");
    blurFs_ = dev_.createShader(shaderDir + "/bloom_blur.frag.spv");
    toneFs_ = dev_.createShader(shaderDir + "/tonemap.frag.spv");
    dofPrepareFs_ = dev_.createShader(shaderDir + "/dof_prepare.frag.spv");
    dofBlurFs_ = dev_.createShader(shaderDir + "/dof_blur.frag.spv");
    dofCompositeFs_ = dev_.createShader(shaderDir + "/dof_composite.frag.spv");
    lumDownsampleFs_ = dev_.createShader(shaderDir + "/exposure_downsample.frag.spv");
    lumReduceFs_ = dev_.createShader(shaderDir + "/exposure_reduce.frag.spv");
    meshVs_ = dev_.createShader(shaderDir + "/mesh.vert.spv");
    meshFs_ = dev_.createShader(shaderDir + "/mesh.frag.spv");
    shadowVs_ = dev_.createShader(shaderDir + "/mesh_shadow.vert.spv");
    shadowFs_ = dev_.createShader(shaderDir + "/mesh_shadow.frag.spv");

    // Pipelines. Full-screen passes declare a 16-byte push constant (target size and flag).
    auto fullscreenPipe = [&](const char* name, const rhi::Shader& fs, rhi::Format fmt) {
        rhi::PipelineDesc d;
        d.name = name;
        d.vertex = fullscreenVs_.get();
        d.fragment = &fs;
        d.colorFormat = fmt;
        d.pushConstantBytes = 16;
        return dev_.createPipeline(d);
    };
    transPipe_ = fullscreenPipe("transmittance", *transFs_, rhi::Format::RGBA16_FLOAT);
    msPipe_ = fullscreenPipe("multiple scattering", *msFs_, rhi::Format::RGBA16_FLOAT);
    skyViewPipe_ = fullscreenPipe("sky-view", *skyViewFs_, rhi::Format::RGBA16_FLOAT);
    cloudPipe_ = fullscreenPipe("clouds", *cloudFs_, rhi::Format::RGBA16_FLOAT);
    cloudShadowPipe_ = fullscreenPipe("cloud shadows", *cloudShadowFs_, rhi::Format::R32_FLOAT);
    shaftPipe_ = fullscreenPipe("god rays", *shaftFs_, rhi::Format::RGBA16_FLOAT);
    prefilterPipe_ = fullscreenPipe("sky prefilter", *prefilterFs_, rhi::Format::RGBA16_FLOAT);
    aerialPipe_ = fullscreenPipe("aerial froxels", *aerialFs_, rhi::Format::RGBA16_FLOAT);
    gtaoPipe_ = fullscreenPipe("ambient occlusion", *gtaoFs_, rhi::Format::R32_FLOAT);
    aoBlurPipe_ = fullscreenPipe("ambient occlusion blur", *aoBlurFs_, rhi::Format::R32_FLOAT);
    taaPipe_ = fullscreenPipe("TAA resolve", *taaFs_, rhi::Format::RGBA16_FLOAT);
    copyPipe_ = fullscreenPipe("TAA history copy", *copyFs_, rhi::Format::RGBA16_FLOAT);
    skyPipe_ = fullscreenPipe("sky composite", *skyFs_, rhi::Format::RGBA16_FLOAT);
    brightPipe_ = fullscreenPipe("bloom bright", *brightFs_, rhi::Format::RGBA16_FLOAT);
    blurPipe_ = fullscreenPipe("bloom blur", *blurFs_, rhi::Format::RGBA16_FLOAT);
    tonePipe_ = fullscreenPipe("tone map", *toneFs_, rhi::Format::RGBA8_UNORM);
    dofPreparePipe_ = fullscreenPipe("DOF prepare", *dofPrepareFs_, rhi::Format::RGBA16_FLOAT);
    dofBlurPipe_ = fullscreenPipe("DOF blur", *dofBlurFs_, rhi::Format::RGBA16_FLOAT);
    dofCompositePipe_ = fullscreenPipe("DOF composite", *dofCompositeFs_, rhi::Format::RGBA16_FLOAT);
    lumDownsamplePipe_ = fullscreenPipe("exposure downsample", *lumDownsampleFs_, rhi::Format::R32_FLOAT);
    lumReducePipe_ = fullscreenPipe("exposure reduce", *lumReduceFs_, rhi::Format::R32_FLOAT);
    {
        rhi::ComputePipelineDesc d;
        d.name = "ocean evolve";
        d.compute = oceanEvolveCs_.get();
        d.pushConstantBytes = 20;  // t, L, g, N, mode
        oceanEvolvePipe_ = dev_.createComputePipeline(d);
        d.name = "ocean fft";
        d.compute = fftCs_.get();
        d.pushConstantBytes = 12;  // Ns, N, mode
        fftPipe_ = dev_.createComputePipeline(d);
    }
    {
        rhi::PipelineDesc d;
        d.name = "ocean";
        d.vertex = oceanVs_.get();
        d.fragment = oceanFs_.get();
        d.colorFormat = rhi::Format::RGBA16_FLOAT;
        d.hasDepth = true;
        d.depthTest = true;
        d.depthWrite = true;
        d.vertexStride = 2 * sizeof(float);
        d.vertexAttributes = {{0, 0, rhi::VertexFormat::Float2}};
        d.cullBack = false;  // the surface is viewed from above; folded waves must not vanish
        oceanPipe_ = dev_.createPipeline(d);
    }
    // Depth prepass (linear depth to a colour target, depth to its own attachment).
    {
        rhi::PipelineDesc d;
        d.name = "depth prepass: scene objects";
        d.vertex = meshVs_.get();
        d.fragment = depthFs_.get();
        d.colorFormat = rhi::Format::R32_FLOAT;
        d.hasDepth = true;
        d.depthTest = true;
        d.depthWrite = true;
        d.vertexStride = 10 * sizeof(float);
        d.vertexAttributes = {{0, 0, rhi::VertexFormat::Float3}, {1, 12, rhi::VertexFormat::Float3},
                              {2, 24, rhi::VertexFormat::Float3}, {3, 36, rhi::VertexFormat::Float1}};
        d.cullBack = false;
        depthMeshPipe_ = dev_.createPipeline(d);
    }
    {
        rhi::PipelineDesc d;
        d.name = "depth prepass: sea";
        d.vertex = oceanVs_.get();
        d.fragment = depthFs_.get();
        d.colorFormat = rhi::Format::R32_FLOAT;
        d.hasDepth = true;
        d.depthTest = true;
        d.depthWrite = true;
        d.vertexStride = 2 * sizeof(float);
        d.vertexAttributes = {{0, 0, rhi::VertexFormat::Float2}};
        d.cullBack = false;
        depthOceanPipe_ = dev_.createPipeline(d);
    }
    // Scene objects and headland: position, normal, albedo, material class (10 floats per vertex).
    auto objectAttrs = std::vector<rhi::VertexAttribute>{
        {0, 0, rhi::VertexFormat::Float3}, {1, 12, rhi::VertexFormat::Float3},
        {2, 24, rhi::VertexFormat::Float3}, {3, 36, rhi::VertexFormat::Float1}};
    const std::vector<rhi::VertexAttribute> shadowAttrs = {
        {0, 0, rhi::VertexFormat::Float3}, {1, 12, rhi::VertexFormat::Float3}, {2, 24, rhi::VertexFormat::Float3}};
    {
        rhi::PipelineDesc d;
        d.name = "scene objects";
        d.vertex = meshVs_.get();
        d.fragment = meshFs_.get();
        d.colorFormat = rhi::Format::RGBA16_FLOAT;
        d.hasDepth = true;
        d.depthTest = true;
        d.depthWrite = true;
        d.vertexStride = kObjStride;
        d.vertexAttributes = objectAttrs;
        d.cullBack = false;
        meshPipe_ = dev_.createPipeline(d);
    }
    {
        rhi::PipelineDesc d;
        d.name = "sun shadow";
        d.vertex = shadowVs_.get();
        d.fragment = shadowFs_.get();
        d.colorFormat = rhi::Format::R32_FLOAT;
        d.hasDepth = true;
        d.depthTest = true;
        d.depthWrite = true;
        d.vertexStride = kObjStride;
        d.vertexAttributes = shadowAttrs;
        d.cullBack = false;
        shadowPipe_ = dev_.createPipeline(d);
    }

    // Images. Every intermediate is sampled, so each is usable by later passes.
    const uint32_t W = settings.width, H = settings.height;
    const uint32_t halfW = std::max(1u, W / 2), halfH = std::max(1u, H / 2);
    hTrans_ = graph_.create(dev_, {kTransW, kTransH, rhi::Format::RGBA16_FLOAT, true, false, true, true, "transmittance LUT"});
    hMs_ = graph_.create(dev_, {kMsSize, kMsSize, rhi::Format::RGBA16_FLOAT, true, false, true, true, "multiple-scattering LUT"});
    hSky_ = graph_.create(dev_, {kSkyW, kSkyH, rhi::Format::RGBA16_FLOAT, true, false, true, true, "sky-view LUT"});
    hCloud_ = graph_.create(dev_, {halfW, halfH, rhi::Format::RGBA16_FLOAT, true, false, true, true, "cloud (half res)"});
    hCloudShadow_ = graph_.create(dev_, {halfW, halfH, rhi::Format::R32_FLOAT, true, false, true, false, "cloud shadow (half res)"});
    hPre_ = graph_.create(dev_, {kSkyW, 3 * kSkyH, rhi::Format::RGBA16_FLOAT, true, false, true, true, "sky prefilter (3 rows)"});
    hLinDepth_ = graph_.create(dev_, {W, H, rhi::Format::R32_FLOAT, true, false, true, true, "linear depth (prepass)"});
    hDepthPre_ = graph_.create(dev_, {W, H, rhi::Format::D32_FLOAT, false, true, false, false, "depth prepass"});
    hAO_ = graph_.create(dev_, {halfW, halfH, rhi::Format::R32_FLOAT, true, false, true, true, "ambient occlusion (half res)"});
    hAOB_ = graph_.create(dev_, {halfW, halfH, rhi::Format::R32_FLOAT, true, false, true, true, "ambient occlusion blurred"});
    hResolved_ = graph_.create(dev_, {W, H, rhi::Format::RGBA16_FLOAT, true, false, true, true, "TAA resolved HDR"});
    // The history persists across frames, so it is owned here and imported into the graph; the
    // graph then treats it as available to passes that read last frame's colour.
    histImage_ = dev_.createImage({W, H, rhi::Format::RGBA16_FLOAT, true, false, true, true, "TAA history"});
    hHist_ = graph_.import(*histImage_, "TAA history");
    hAerialL_ = graph_.create(dev_, {kAerialW, kAerialH, rhi::Format::RGBA16_FLOAT, true, false, true, true, "aerial in-scatter froxels"});
    hAerialT_ = graph_.create(dev_, {kAerialW, kAerialH, rhi::Format::RGBA16_FLOAT, true, false, true, true, "aerial transmittance froxels"});
    hShaft_ = graph_.create(dev_, {std::max(1u, W / 4), std::max(1u, H / 4), rhi::Format::RGBA16_FLOAT, true, false, true, true, "god rays (quarter res)"});
    hHdr_ = graph_.create(dev_, {W, H, rhi::Format::RGBA16_FLOAT, true, false, true, true, "HDR"});
    hDepth_ = graph_.create(dev_, {W, H, rhi::Format::D32_FLOAT, false, true, false, false, "depth"});
    hShadow_ = graph_.create(dev_, {settings.shadowMapSize, settings.shadowMapSize, rhi::Format::R32_FLOAT, true, false, true, true, "sun shadow map"});
    hShadowNear_ = graph_.create(dev_, {1024, 1024, rhi::Format::R32_FLOAT, true, false, true, true, "sun shadow near"});
    hShadowNearDepth_ = graph_.create(dev_, {1024, 1024, rhi::Format::D32_FLOAT, false, true, false, false, "sun shadow near depth"});
    hShadowFar_ = graph_.create(dev_, {1024, 1024, rhi::Format::R32_FLOAT, true, false, true, true, "sun shadow far"});
    hShadowFarDepth_ = graph_.create(dev_, {1024, 1024, rhi::Format::D32_FLOAT, false, true, false, false, "sun shadow far depth"});
    hShadowDepth_ = graph_.create(dev_, {settings.shadowMapSize, settings.shadowMapSize, rhi::Format::D32_FLOAT, false, true, false, false, "shadow depth"});
    hRefl_ = graph_.create(dev_, {halfW, halfH, rhi::Format::RGBA16_FLOAT, true, false, true, true, "planar reflection (half res)"});
    hReflDepth_ = graph_.create(dev_, {halfW, halfH, rhi::Format::D32_FLOAT, false, true, false, false, "planar reflection depth"});
    hBloomA_ = graph_.create(dev_, {halfW, halfH, rhi::Format::RGBA16_FLOAT, true, false, true, true, "bloom A (half res)"});
    hBloomB_ = graph_.create(dev_, {halfW, halfH, rhi::Format::RGBA16_FLOAT, true, false, true, true, "bloom B (half res)"});
    hLdr_ = graph_.create(dev_, {W, H, rhi::Format::RGBA8_UNORM, true, false, false, true, "display"});
    // Adaptive depth-of-field chain (full resolution HDR): prepared (colour + CoC), horizontal and
    // vertical bokeh blur, and the final composite.
    hDofPrep_ = graph_.create(dev_, {W, H, rhi::Format::RGBA16_FLOAT, true, false, true, true, "DOF prepared"});
    hDofH_ = graph_.create(dev_, {W, H, rhi::Format::RGBA16_FLOAT, true, false, true, true, "DOF blur H"});
    hDofV_ = graph_.create(dev_, {W, H, rhi::Format::RGBA16_FLOAT, true, false, true, true, "DOF blur V"});
    hDofOut_ = graph_.create(dev_, {W, H, rhi::Format::RGBA16_FLOAT, true, false, true, true, "DOF composite"});
    // Auto-exposure luminance reduction: an 8x8 grid, then a single final value.
    hLumA_ = graph_.create(dev_, {8, 8, rhi::Format::R32_FLOAT, true, false, true, true, "luminance grid"});
    hLumB_ = graph_.create(dev_, {1, 1, rhi::Format::R32_FLOAT, true, false, true, true, "luminance average"});

    // Buffers: frame uniforms (main and mirrored camera), the ocean mesh, scene objects, headland.
    ubo_ = dev_.createBuffer(sizeof(FrameData), rhi::BufferUsage::Uniform);
    uboRefl_ = dev_.createBuffer(sizeof(FrameData), rhi::BufferUsage::Uniform);
    uboShadowNear_ = dev_.createBuffer(sizeof(FrameData), rhi::BufferUsage::Uniform);
    uboShadowFar_ = dev_.createBuffer(sizeof(FrameData), rhi::BufferUsage::Uniform);
    const scene::OceanMesh mesh = scene::buildOceanMesh();
    oceanVb_ = dev_.createBuffer(mesh.rest.size() * sizeof(float), rhi::BufferUsage::Vertex);
    std::memcpy(oceanVb_->data(), mesh.rest.data(), mesh.rest.size() * sizeof(float));
    oceanIb_ = dev_.createBuffer(mesh.indices.size() * sizeof(uint32_t), rhi::BufferUsage::Index);
    std::memcpy(oceanIb_->data(), mesh.indices.data(), mesh.indices.size() * sizeof(uint32_t));
    oceanIndexCount_ = mesh.indexCount();
    capOn_ = envFlag("ENGINE_SEA_CAP", true);
    cloudLayersOn_ = envFlag("ENGINE_CLOUD_LAYERS", true);
    profile_ = envFlag("ENGINE_PROFILE", false);
    autoExposureOn_ = envFlag("ENGINE_AUTO_EXPOSURE", true);
    if (const char* v = std::getenv("ENGINE_DOF_FOCUS"); v && *v) dofFocusM_ = std::atof(v);
    if (const char* v = std::getenv("ENGINE_DOF_APERTURE"); v && *v) dofAperture_ = std::atof(v);
    graph_.setProfiling(profile_);
    // FFT ocean buffers: the initial spectrum h0 (built once, on the CPU), three evolved spectra and a ping buffer.
    {
        const auto params = ocean::defaultParams();
        std::vector<float> h0;
        ocean::buildH0(params, h0, 1234u);
        const size_t bytes = static_cast<size_t>(ocean::kOceanN) * ocean::kOceanN * 2 * sizeof(float);
        oceanH0_ = dev_.createBuffer(bytes, rhi::BufferUsage::Storage);
        std::memcpy(oceanH0_->data(), h0.data(), bytes);
        for (auto& b : oceanSpec_) b = dev_.createBuffer(bytes, rhi::BufferUsage::Storage);
        oceanPing_ = dev_.createBuffer(bytes, rhi::BufferUsage::Storage);
        // Capillary cascade: its own h0 and height field. It shares the ping buffer (its passes are ordered after the
        // main cascade by compute barriers).
        std::vector<float> h0cap;
        ocean::buildH0(ocean::defaultCapillaryParams(), h0cap, 4321u);
        oceanH0Cap_ = dev_.createBuffer(bytes, rhi::BufferUsage::Storage);
        std::memcpy(oceanH0Cap_->data(), h0cap.data(), bytes);
        oceanCap_ = dev_.createBuffer(bytes, rhi::BufferUsage::Storage);
    }
    const std::vector<float> objects = scene::buildSceneObjects();
    objVertexCount_ = static_cast<uint32_t>(objects.size() / 10);
    // An empty object list (the offshore structures are gated off) still needs a non-zero buffer;
    // the draw uses objVertexCount_ == 0 and submits nothing.
    objVb_ = dev_.createBuffer(std::max<size_t>(objects.size(), 10) * sizeof(float), rhi::BufferUsage::Vertex);
    if (!objects.empty()) std::memcpy(objVb_->data(), objects.data(), objects.size() * sizeof(float));
    const std::vector<float> terrain = scene::buildLandscape();
    terrVertexCount_ = static_cast<uint32_t>(terrain.size() / 10);
    terrVb_ = dev_.createBuffer(terrain.size() * sizeof(float), rhi::BufferUsage::Vertex);
    std::memcpy(terrVb_->data(), terrain.data(), terrain.size() * sizeof(float));

    // Scanned materials: raw RGBA8 files from tools/bake_materials.py, mip-mapped on load.
    woodAlbedo_ = assets::loadRawTexture(dev_, assetDir + "/wood_albedo_512.rgba", 512, 512, "wood albedo");
    woodNormal_ = assets::loadRawTexture(dev_, assetDir + "/wood_normal_512.rgba", 512, 512, "wood normal");
    rockAlbedo_ = assets::loadRawTexture(dev_, assetDir + "/rock_albedo_512.rgba", 512, 512, "rock albedo");
    rockNormal_ = assets::loadRawTexture(dev_, assetDir + "/rock_normal_512.rgba", 512, 512, "rock normal");

    // Descriptor sets. Binding b of the fixed layout is slot b - 1 of a TexSet (see rhi.h):
    //   1 transmittance, 2 sky-view, 3 cloud, 4 HDR, 5 multiple scattering, 6 bloom, 7 sun shadow,
    //   8 wood albedo, 9 cloud shadow, 10 wood normal, 11 rock albedo, 12 rock normal, 13 god rays,
    //   14 planar reflection.
    // A pass binds only the images it reads; a pass's target is never bound as a texture.
    auto makeSet = [&](rhi::Buffer* ubo, const TexSet& tex) {
        auto set = dev_.createDescriptorSet();
        dev_.updateDescriptorSet(*set, ubo, tex);
        return set;
    };
    rhi::Image& trans = graph_.image(hTrans_);
    rhi::Image& ms = graph_.image(hMs_);
    rhi::Image& sky = graph_.image(hSky_);
    rhi::Image& cloud = graph_.image(hCloud_);
    rhi::Image& cloudShadow = graph_.image(hCloudShadow_);
    rhi::Image& hdr = graph_.image(hHdr_);
    rhi::Image& bloomA = graph_.image(hBloomA_);
    rhi::Image& bloomB = graph_.image(hBloomB_);
    rhi::Image& shadow = graph_.image(hShadow_);
    rhi::Image& shaft = graph_.image(hShaft_);
    rhi::Image& refl = graph_.image(hRefl_);
    rhi::Image& pre = graph_.image(hPre_);
    rhi::Image& shadowNear = graph_.image(hShadowNear_);
    rhi::Image& shadowFar = graph_.image(hShadowFar_);
    rhi::Image& aerialL = graph_.image(hAerialL_);
    rhi::Image& linDepth = graph_.image(hLinDepth_);
    rhi::Image& aoRaw = graph_.image(hAO_);
    rhi::Image& aoBlur = graph_.image(hAOB_);
    rhi::Image& resolved = graph_.image(hResolved_);
    rhi::Image& hist = graph_.image(hHist_);
    rhi::Image& aerialT = graph_.image(hAerialT_);
    rhi::Image& dofPrep = graph_.image(hDofPrep_);
    rhi::Image& dofH = graph_.image(hDofH_);
    rhi::Image& dofV = graph_.image(hDofV_);
    rhi::Image& dofOut = graph_.image(hDofOut_);
    rhi::Image& lumA = graph_.image(hLumA_);
    // Scene objects, as drawn in the main view and in the mirrored reflection view.
    const TexSet meshTex = bindings({{1, &trans}, {2, &sky}, {3, &cloud}, {5, &ms}, {7, &shadow},
                                     {9, &cloudShadow}, {8, woodAlbedo_.get()}, {10, woodNormal_.get()},
                                     {11, rockAlbedo_.get()}, {12, rockNormal_.get()}, {15, &pre}, {16, &aerialL}, {17, &aerialT}, {18, &shadowNear}, {19, &shadowFar}, {20, &linDepth}, {21, &aoBlur}, {22, &hist}, {23, &hist}});
    dTrans_ = makeSet(ubo_.get(), bindings({}));
    dMs_ = makeSet(ubo_.get(), bindings({{1, &trans}}));
    dSkyView_ = makeSet(ubo_.get(), bindings({{1, &trans}, {5, &ms}}));
    dCloud_ = makeSet(ubo_.get(), bindings({{2, &sky}}));
    dCloudShadow_ = makeSet(ubo_.get(), bindings({}));
    dShaft_ = makeSet(ubo_.get(), bindings({{1, &trans}}));
    dSky_ = makeSet(ubo_.get(), bindings({{2, &sky}, {3, &cloud}, {13, &shaft}}));
    dOcean_ = makeSet(ubo_.get(), bindings({{1, &trans}, {2, &sky}, {3, &cloud}, {5, &ms}, {7, &shadow},
                                       {9, &cloudShadow}, {13, &shaft}, {14, &refl}, {15, &pre}, {16, &aerialL}, {17, &aerialT}, {18, &shadowNear}, {19, &shadowFar}, {21, &aoBlur}, {22, &hist}}));
    dAerial_ = makeSet(ubo_.get(), bindings({{1, &trans}, {5, &ms}}));
    dShadowNear_ = makeSet(uboShadowNear_.get(), bindings({}));
    dShadowFar_ = makeSet(uboShadowFar_.get(), bindings({}));
    dDepthPre_ = makeSet(ubo_.get(), bindings({}));
    dGtao_ = makeSet(ubo_.get(), bindings({{20, &linDepth}}));
    dAoBlur_ = makeSet(ubo_.get(), bindings({{20, &linDepth}, {21, &aoRaw}}));
    dTaa_ = makeSet(ubo_.get(), bindings({{4, &hdr}, {20, &linDepth}, {22, &hist}}));
    dCopy_ = makeSet(ubo_.get(), bindings({{4, &resolved}}));
    // FFT ocean: evolve (h0 -> spectrum m) and the two ping-pong sets of the 2-D FFT. The ocean sets expose
    // the three finished spectra to the vertex stage (depth prepass and ocean surface both use oceanVs_).
    for (uint32_t m = 0; m < 3; m++) {
        dOceanEvolve_[m] = dev_.createDescriptorSet();
        dev_.setStorageBuffer(*dOceanEvolve_[m], *oceanH0_, 0);
        dev_.setStorageBuffer(*dOceanEvolve_[m], *oceanSpec_[m], 1);
        dOceanFftAB_[m] = dev_.createDescriptorSet();
        dev_.setStorageBuffer(*dOceanFftAB_[m], *oceanSpec_[m], 0);
        dev_.setStorageBuffer(*dOceanFftAB_[m], *oceanPing_, 1);
        dOceanFftBA_[m] = dev_.createDescriptorSet();
        dev_.setStorageBuffer(*dOceanFftBA_[m], *oceanPing_, 0);
        dev_.setStorageBuffer(*dOceanFftBA_[m], *oceanSpec_[m], 1);
        dev_.setStorageBuffer(*dOcean_, *oceanSpec_[m], m);
        dev_.setStorageBuffer(*dDepthPre_, *oceanSpec_[m], m);
    }
    // Capillary cascade sets: evolve, and the two ping-pong passes around the shared ping buffer. Slot 3 is the
    // capillary height field, read by the ocean fragment stage.
    dOceanEvolveCap_ = dev_.createDescriptorSet();
    dev_.setStorageBuffer(*dOceanEvolveCap_, *oceanH0Cap_, 0);
    dev_.setStorageBuffer(*dOceanEvolveCap_, *oceanCap_, 1);
    dOceanFftABCap_ = dev_.createDescriptorSet();
    dev_.setStorageBuffer(*dOceanFftABCap_, *oceanCap_, 0);
    dev_.setStorageBuffer(*dOceanFftABCap_, *oceanPing_, 1);
    dOceanFftBACap_ = dev_.createDescriptorSet();
    dev_.setStorageBuffer(*dOceanFftBACap_, *oceanPing_, 0);
    dev_.setStorageBuffer(*dOceanFftBACap_, *oceanCap_, 1);
    dev_.setStorageBuffer(*dOcean_, *oceanCap_, 3);
    dev_.setStorageBuffer(*dDepthPre_, *oceanCap_, 3);
    dPrefilter_ = makeSet(ubo_.get(), bindings({{2, &sky}}));
    dMesh_ = makeSet(ubo_.get(), meshTex);
    dMeshRefl_ = makeSet(uboRefl_.get(), meshTex);
    dShadow_ = makeSet(ubo_.get(), bindings({}));
    dBright_ = makeSet(ubo_.get(), bindings({{4, &dofOut}}));
    dBlurH_ = makeSet(ubo_.get(), bindings({{4, &bloomA}}));
    dBlurV_ = makeSet(ubo_.get(), bindings({{4, &bloomB}}));
    dTone_ = makeSet(ubo_.get(), bindings({{4, &dofOut}, {6, &bloomA}}));
    // Adaptive DOF and auto-exposure sets. These passes reuse the fixed sampler slots; each pass
    // binds only the images it reads (slot 4 = "source", slot 6 = "second source" in these passes).
    dDofPrepare_ = makeSet(ubo_.get(), bindings({{4, &resolved}, {20, &linDepth}}));
    dDofBlurH_ = makeSet(ubo_.get(), bindings({{4, &dofPrep}}));
    dDofBlurV_ = makeSet(ubo_.get(), bindings({{4, &dofH}}));
    dDofComposite_ = makeSet(ubo_.get(), bindings({{4, &dofPrep}, {6, &dofV}}));
    dLumDownsample_ = makeSet(ubo_.get(), bindings({{4, &resolved}}));
    dLumReduce_ = makeSet(ubo_.get(), bindings({{4, &lumA}}));

    // Passes, in the order they run.
    graph_.addPass("transmittance LUT")
        .colorTarget(hTrans_, rhi::LoadOp::Clear)
        .runOnce()
        .execute([this](rhi::CommandList& c) { drawFullscreen(c, *transPipe_, *dTrans_, graph_.image(hTrans_)); });
    graph_.addPass("multiple-scattering LUT")
        .read(hTrans_)
        .colorTarget(hMs_, rhi::LoadOp::Clear)
        .runOnce()
        .execute([this](rhi::CommandList& c) { drawFullscreen(c, *msPipe_, *dMs_, graph_.image(hMs_)); });
    graph_.addPass("sky-view LUT")
        .read(hTrans_)
        .read(hMs_)
        .colorTarget(hSky_, rhi::LoadOp::Clear)
        .execute([this](rhi::CommandList& c) { drawFullscreen(c, *skyViewPipe_, *dSkyView_, graph_.image(hSky_)); });
    graph_.addPass("aerial froxels")
        .read(hTrans_)
        .read(hMs_)
        .colorTarget(hAerialL_, rhi::LoadOp::Clear)
        .execute([this](rhi::CommandList& c) { drawFullscreen(c, *aerialPipe_, *dAerial_, graph_.image(hAerialL_), 0.0f); });
    graph_.addPass("aerial transmittance")
        .read(hTrans_)
        .read(hMs_)
        .colorTarget(hAerialT_, rhi::LoadOp::Clear)
        .execute([this](rhi::CommandList& c) { drawFullscreen(c, *aerialPipe_, *dAerial_, graph_.image(hAerialT_), 1.0f); });
    graph_.addPass("sky prefilter")
        .read(hSky_)
        .colorTarget(hPre_, rhi::LoadOp::Clear)
        .execute([this](rhi::CommandList& c) { drawFullscreen(c, *prefilterPipe_, *dPrefilter_, graph_.image(hPre_)); });
    graph_.addPass("cloud slab")
        .read(hSky_)
        .colorTarget(hCloud_, rhi::LoadOp::Clear)
        .execute([this](rhi::CommandList& c) { drawFullscreen(c, *cloudPipe_, *dCloud_, graph_.image(hCloud_)); });
    graph_.addPass("cloud shadow field")
        .colorTarget(hCloudShadow_, rhi::LoadOp::Clear, {1.0f, 1.0f, 1.0f, 1.0f})
        .execute([this](rhi::CommandList& c) {
            drawFullscreen(c, *cloudShadowPipe_, *dCloudShadow_, graph_.image(hCloudShadow_));
        });
    graph_.addPass("god rays")
        .read(hTrans_)
        .colorTarget(hShaft_, rhi::LoadOp::Clear)
        .execute([this](rhi::CommandList& c) { drawFullscreen(c, *shaftPipe_, *dShaft_, graph_.image(hShaft_)); });
    graph_.addPass("sky composite")
        .read(hSky_)
        .read(hCloud_)
        .read(hShaft_)
        .colorTarget(hHdr_, rhi::LoadOp::Clear)
        .execute([this](rhi::CommandList& c) { drawFullscreen(c, *skyPipe_, *dSky_, graph_.image(hHdr_)); });
    graph_.addPass("ocean FFT")
        .execute([this](rhi::CommandList& c) { recordOceanFft(c); });
    graph_.addPass("depth prepass")
        .colorTarget(hLinDepth_, rhi::LoadOp::Clear)
        .depthTarget(hDepthPre_)
        .execute([this](rhi::CommandList& c) {
            c.bindPipeline(*depthMeshPipe_);
            c.bindDescriptorSet(*dDepthPre_);
            c.bindVertexBuffer(*objVb_);
            c.draw(objVertexCount_);
            c.bindVertexBuffer(*terrVb_);
            c.draw(terrVertexCount_);
            c.bindPipeline(*depthOceanPipe_);
            c.bindDescriptorSet(*dDepthPre_);
            c.bindVertexBuffer(*oceanVb_);
            c.bindIndexBuffer(*oceanIb_);
            c.drawIndexed(oceanIndexCount_);
        });
    graph_.addPass("ambient occlusion")
        .read(hLinDepth_)
        .colorTarget(hAO_, rhi::LoadOp::Clear)
        .execute([this](rhi::CommandList& c) { drawFullscreen(c, *gtaoPipe_, *dGtao_, graph_.image(hAO_)); });
    graph_.addPass("ambient occlusion blur")
        .read(hLinDepth_)
        .read(hAO_)
        .colorTarget(hAOB_, rhi::LoadOp::Clear)
        .execute([this](rhi::CommandList& c) { drawFullscreen(c, *aoBlurPipe_, *dAoBlur_, graph_.image(hAOB_)); });
    graph_.addPass("sun shadow map")
        .colorTarget(hShadow_, rhi::LoadOp::Clear)
        .depthTarget(hShadowDepth_)
        .execute([this](rhi::CommandList& c) {
            c.bindPipeline(*shadowPipe_);
            c.bindDescriptorSet(*dShadow_);
            c.bindVertexBuffer(*objVb_);
            c.draw(objVertexCount_);
            c.bindVertexBuffer(*terrVb_);
            c.draw(terrVertexCount_);
        });
    // Planar reflection: the scene above the sea, seen from the mirrored camera (see renderFrame).
    // Half resolution; alpha is coverage, so the sky shows where nothing is reflected.
    graph_.addPass("sun shadow near map")
        .colorTarget(hShadowNear_, rhi::LoadOp::Clear)
        .depthTarget(hShadowNearDepth_)
        .execute([this](rhi::CommandList& c) {
            c.bindPipeline(*shadowPipe_);
            c.bindDescriptorSet(*dShadowNear_);
            c.bindVertexBuffer(*objVb_);
            c.draw(objVertexCount_);
            c.bindVertexBuffer(*terrVb_);
            c.draw(terrVertexCount_);
        });
    graph_.addPass("sun shadow far map")
        .colorTarget(hShadowFar_, rhi::LoadOp::Clear)
        .depthTarget(hShadowFarDepth_)
        .execute([this](rhi::CommandList& c) {
            c.bindPipeline(*shadowPipe_);
            c.bindDescriptorSet(*dShadowFar_);
            c.bindVertexBuffer(*objVb_);
            c.draw(objVertexCount_);
            c.bindVertexBuffer(*terrVb_);
            c.draw(terrVertexCount_);
        });
    graph_.addPass("planar reflection")
        .read(hTrans_)
        .read(hSky_)
        .read(hCloud_)
        .read(hMs_)
        .read(hShadow_)
        .read(hCloudShadow_)
        .read(hPre_)
        .read(hAerialL_)
        .read(hAerialT_)
        .read(hShadowNear_)
        .read(hShadowFar_)
        .read(hAOB_)
        .read(hLinDepth_)
        .read(hHist_)
        .colorTarget(hRefl_, rhi::LoadOp::Clear)
        .depthTarget(hReflDepth_)
        .execute([this](rhi::CommandList& c) {
            c.bindPipeline(*meshPipe_);
            c.bindDescriptorSet(*dMeshRefl_);
            c.bindVertexBuffer(*objVb_);
            c.draw(objVertexCount_);
            c.bindVertexBuffer(*terrVb_);
            c.draw(terrVertexCount_);
        });
    graph_.addPass("ocean surface and scene objects")
        .read(hTrans_)
        .read(hSky_)
        .read(hCloud_)
        .read(hMs_)
        .read(hShadow_)
        .read(hCloudShadow_)
        .read(hShaft_)
        .read(hRefl_)
        .read(hPre_)
        .read(hAerialL_)
        .read(hAerialT_)
        .read(hShadowNear_)
        .read(hShadowFar_)
        .read(hAOB_)
        .read(hLinDepth_)
        .read(hHist_)
        .colorTarget(hHdr_, rhi::LoadOp::Load)
        .depthTarget(hDepth_)
        .execute([this](rhi::CommandList& c) {
            c.bindPipeline(*oceanPipe_);
            c.bindDescriptorSet(*dOcean_);
            c.bindVertexBuffer(*oceanVb_);
            c.bindIndexBuffer(*oceanIb_);
            c.drawIndexed(oceanIndexCount_);
            c.bindPipeline(*meshPipe_);
            c.bindDescriptorSet(*dMesh_);
            c.bindVertexBuffer(*objVb_);
            c.draw(objVertexCount_);
            c.bindVertexBuffer(*terrVb_);
            c.draw(terrVertexCount_);
        });
    graph_.addPass("TAA resolve")
        .read(hHdr_)
        .read(hLinDepth_)
        .read(hHist_)
        .colorTarget(hResolved_, rhi::LoadOp::Clear)
        .execute([this](rhi::CommandList& c) { drawFullscreen(c, *taaPipe_, *dTaa_, graph_.image(hResolved_)); });
    graph_.addPass("TAA history copy")
        .read(hResolved_)
        .colorTarget(hHist_, rhi::LoadOp::Clear)
        .execute([this](rhi::CommandList& c) { drawFullscreen(c, *copyPipe_, *dCopy_, graph_.image(hHist_)); });
    // Adaptive depth-of-field (ported from the RAGE adaptiveDof bank): prepare colour + circle of
    // confusion, a separable bokeh blur, then a CoC-weighted composite.
    graph_.addPass("DOF prepare")
        .read(hResolved_)
        .read(hLinDepth_)
        .colorTarget(hDofPrep_, rhi::LoadOp::Clear)
        .execute([this](rhi::CommandList& c) { drawFullscreen(c, *dofPreparePipe_, *dDofPrepare_, graph_.image(hDofPrep_)); });
    graph_.addPass("DOF blur horizontal")
        .read(hDofPrep_)
        .colorTarget(hDofH_, rhi::LoadOp::Clear)
        .execute([this](rhi::CommandList& c) { drawFullscreen(c, *dofBlurPipe_, *dDofBlurH_, graph_.image(hDofH_), 0.0f); });
    graph_.addPass("DOF blur vertical")
        .read(hDofH_)
        .colorTarget(hDofV_, rhi::LoadOp::Clear)
        .execute([this](rhi::CommandList& c) { drawFullscreen(c, *dofBlurPipe_, *dDofBlurV_, graph_.image(hDofV_), 1.0f); });
    graph_.addPass("DOF composite")
        .read(hDofPrep_)
        .read(hDofV_)
        .colorTarget(hDofOut_, rhi::LoadOp::Clear)
        .execute([this](rhi::CommandList& c) { drawFullscreen(c, *dofCompositePipe_, *dDofComposite_, graph_.image(hDofOut_)); });
    // Auto-exposure (ported from the RAGE exposureGPU / avgLuminanceCompute bank): reduce the
    // resolved HDR to a luminance grid, then to a single value read back on the host.
    graph_.addPass("exposure downsample")
        .read(hResolved_)
        .colorTarget(hLumA_, rhi::LoadOp::Clear)
        .execute([this](rhi::CommandList& c) { drawFullscreen(c, *lumDownsamplePipe_, *dLumDownsample_, graph_.image(hLumA_)); });
    graph_.addPass("exposure reduce")
        .read(hLumA_)
        .colorTarget(hLumB_, rhi::LoadOp::Clear)
        .execute([this](rhi::CommandList& c) { drawFullscreen(c, *lumReducePipe_, *dLumReduce_, graph_.image(hLumB_)); });
    graph_.addPass("bloom bright")
        .read(hDofOut_)
        .colorTarget(hBloomA_, rhi::LoadOp::Clear)
        .execute([this](rhi::CommandList& c) { drawFullscreen(c, *brightPipe_, *dBright_, graph_.image(hBloomA_)); });
    graph_.addPass("bloom blur horizontal")
        .read(hBloomA_)
        .colorTarget(hBloomB_, rhi::LoadOp::Clear)
        .execute([this](rhi::CommandList& c) {
            drawFullscreen(c, *blurPipe_, *dBlurH_, graph_.image(hBloomB_), 1.0f);
        });
    graph_.addPass("bloom blur vertical")
        .read(hBloomB_)
        .colorTarget(hBloomA_, rhi::LoadOp::Clear)
        .execute([this](rhi::CommandList& c) {
            drawFullscreen(c, *blurPipe_, *dBlurV_, graph_.image(hBloomA_), 0.0f);
        });
    graph_.addPass("tone map")
        .read(hResolved_)
        .read(hBloomA_)
        .colorTarget(hLdr_, rhi::LoadOp::Clear)
        .execute([this](rhi::CommandList& c) { drawFullscreen(c, *tonePipe_, *dTone_, graph_.image(hLdr_)); });

    graph_.compile();
    cmd_ = dev_.createCommandList();

    // Ambient probes: the sky-view LUT is independent of time, so one frame gives the sky the
    // probes need. The probes are empty during this frame (dims 0), so the sky fallback is used.
    renderFrame(0.33);
    scene::SkyTable sky32;
    sky32.width = static_cast<int>(kSkyW);
    sky32.height = static_cast<int>(kSkyH);
    sky32.rgba = scene::halfRgbaToFloat(readImage(graph_.image(hSky_), 8));
    probes_ = scene::bakeAmbientProbes(objects, sky32);
}

// Evolves the spectrum to oceanTime_ and runs the 2-D FFT for height, Dx and Dz. Every dispatch is followed by a
// compute barrier, which orders the ping-pong passes and the later vertex reads.
void OceanRenderer::recordOceanFft(rhi::CommandList& c) {
    const uint32_t N = ocean::kOceanN;
    const uint32_t groups = N / 2 / 32;  // the FFT stage runs 32 butterflies per workgroup
    const float t = static_cast<float>(oceanTime_);
    const float g = 9.81f;
    // One cascade: evolve the spectrum to t, then the 2-D FFT as log2 N row passes and log2 N column passes. The
    // passes alternate between the two buffers, so an even count leaves the result in the spectrum buffer.
    auto runCascade = [&](rhi::DescriptorSet& evolve, rhi::DescriptorSet& ab, rhi::DescriptorSet& ba, float L,
                          uint32_t field) {
        c.bindComputePipeline(*oceanEvolvePipe_);
        c.bindDescriptorSet(evolve);
        const struct { float t, L, g; uint32_t N, mode; } ev{t, L, g, N, field};
        c.pushConstants(&ev, sizeof(ev));
        c.dispatch(N / 8, N / 8, 1);
        c.computeBarrier();
        c.bindComputePipeline(*fftPipe_);
        for (uint32_t pass = 0; pass < 2 * ocean::kOceanLog2N; pass++) {
            c.bindDescriptorSet(pass % 2 == 0 ? ab : ba);
            const uint32_t push[3] = {1u << (pass % ocean::kOceanLog2N), N, pass < ocean::kOceanLog2N ? 0u : 1u};
            c.pushConstants(push, sizeof(push));
            c.dispatch(groups, N, 1);
            c.computeBarrier();
        }
    };
    for (uint32_t m = 0; m < 3; m++)
        runCascade(*dOceanEvolve_[m], *dOceanFftAB_[m], *dOceanFftBA_[m], ocean::kOceanL, m);
    if (capOn_) runCascade(*dOceanEvolveCap_, *dOceanFftABCap_, *dOceanFftBACap_, ocean::kOceanCapL, 0);
}

OceanRenderer::~OceanRenderer() = default;

void OceanRenderer::renderFrame(double timeSeconds) {
    using clock = std::chrono::steady_clock;

    // Main camera.
    const double aspect = static_cast<double>(settings_.width) / settings_.height;
    const math::Vec3 up{0.0, 1.0, 0.0};
    const math::Mat4 view = math::lookAt(settings_.camPos, settings_.camTarget, up);
    const math::Mat4 proj = math::perspectiveVulkan(settings_.fovYDeg * M_PI / 180.0, aspect, 0.1, 2.0e5);
    // Sub-pixel jitter (Halton 2,3 over 8 frames) applied to the projection rows that set clip x and y.
    auto halton = [](uint32_t i, uint32_t base) {
        double f = 1.0, r = 0.0;
        while (i > 0) { f /= base; r += f * (i % base); i /= base; }
        return r;
    };
    const uint32_t jIndex = frameIndex_ % 8 + 1;
    const double jx = halton(jIndex, 2) - 0.5, jy = halton(jIndex, 3) - 0.5;  // pixels
    math::Mat4 projJ = proj;
    for (int c = 0; c < 4; c++) {
        projJ.at(0, c) += 2.0 * jx / settings_.width * proj.at(3, c);
        projJ.at(1, c) += 2.0 * jy / settings_.height * proj.at(3, c);
    }
    const math::Mat4 unjitteredVP = proj * view;
    const math::Mat4 viewProj = projJ * view;
    const math::Mat4 invViewProj = math::inverse(viewProj);
    frameIndex_++;

    // Mirrored camera: the reflection of the main camera in the sea plane. A point on the sea
    // seen by the main camera is seen by this one along the same line, so sampling the mirrored
    // render at the sea point's projection gives the reflection. The render keeps only geometry
    // above the sea (discard in mesh.frag).
    const math::Vec3 mirEye{settings_.camPos.x, -settings_.camPos.y, settings_.camPos.z};
    const math::Vec3 mirTarget{settings_.camTarget.x, -settings_.camTarget.y, settings_.camTarget.z};
    const math::Mat4 mirViewProj = proj * math::lookAt(mirEye, mirTarget, up);

    // Sun shadow: an orthographic view from the sun, centred on the objects. The box is 320 m
    // across and 1200 m deep; the eye sits on the sun side so the nearest occluder has the
    // smallest depth.
    const math::Vec3 shadowCentre{0.0, 0.0, -120.0};  // the bay between the camera and the skyline
    const math::Vec3 shadowEye{shadowCentre.x + sunUnit_.x * 600.0, shadowCentre.y + sunUnit_.y * 600.0,
                               shadowCentre.z + sunUnit_.z * 600.0};
    const math::Mat4 sunView = math::lookAt(shadowEye, shadowCentre, up);
    const math::Mat4 sunProj = math::orthoVulkan(-230.0, 230.0, -230.0, 230.0, 0.1, 2400.0);  // reaches the far skyline
    const math::Mat4 sunViewProj = sunProj * sunView;

    // Sun disk radiance = illuminance / solid angle of the disk.
    const double cosR = std::cos(kSunRadiusDeg * M_PI / 180.0);
    const double omega = 2.0 * M_PI * (1.0 - cosR);

    FrameData fd{};
    storeMat(fd.viewProj, viewProj);
    storeMat(fd.invViewProj, invViewProj);
    storeMat(fd.sunViewProj, sunViewProj);
    storeMat(fd.reflViewProj, mirViewProj);
    storeMat(fd.mainViewProj, viewProj);
    storeMat(fd.prevViewProj, prevViewProj_);
    fd.taa[2] = historyValid_ ? 1.0f : 0.0f;
    fd.taa[3] = capOn_ ? 1.0f : 0.0f;  // read by ocean.frag
    fd.camPos[0] = toF(settings_.camPos.x);
    fd.camPos[1] = toF(settings_.camPos.y);
    fd.camPos[2] = toF(settings_.camPos.z);
    fd.sunDir[0] = toF(sunUnit_.x);
    fd.sunDir[1] = toF(sunUnit_.y);
    fd.sunDir[2] = toF(sunUnit_.z);
    fd.sunDir[3] = toF(cosR);
    for (int c = 0; c < 3; c++) {
        const double lux = (c == 0 ? sunLux_.x : c == 1 ? sunLux_.y : sunLux_.z);
        fd.sunIlluminance[c] = toF(lux * preExposure_);
        fd.sunRadiance[c] = toF(lux / omega * preExposure_);
    }
    fd.exposure[0] = toF(preExposure_);
    // Wave LOD footprint: no finer than the pixel, and no finer than the mesh cell.
    const double pixelAngle = settings_.fovYDeg * M_PI / 180.0 / settings_.height;
    const double cellAngle = 2.0 * M_PI / scene::kOceanSegments;
    fd.exposure[1] = toF(std::max(pixelAngle, cellAngle));
    fd.exposure[2] = cloudLayersOn_ ? 1.0f : 0.0f;  // read by cloudShadowTransmittance in common.glsl
    fd.exposure[3] = autoExposure_;  // auto-exposure gain, read by tonemap.frag
    fd.time[0] = toF(timeSeconds);
    // Adaptive DOF lens parameters (see dof_prepare.frag).
    fd.dof[0] = dofFocalMm_;
    fd.dof[1] = dofAperture_;
    fd.dof[2] = dofFocusM_;
    fd.dof[3] = dofSensorH_;
    // Weather: rain intensity (0..1), wetness factor (0..1), flood level (metres above shore), foam intensity (0..1)
    // uWeather is a vec4 member of FrameData (common.glsl); indices x/y/z/w map to rain/wetness/flood/foam
    fd.uWeather[0] = 0.0f;  // rain intensity
    fd.uWeather[1] = 0.0f;  // wetness factor
    fd.uWeather[2] = 0.0f;  // flood level
    fd.uWeather[3] = 0.0f;  // foam intensity
    // Probe grid (dims 0 before the bake, which makes the shaders fall back to the sky).
    for (int i = 0; i < 3; i++) {
        fd.probeOrigin[i] = probes_.origin[i];
        fd.probeStep[i] = probes_.step[i];
        fd.probeDims[i] = static_cast<float>(probes_.dims[i]);
    }
    const size_t probeFloats = std::min(probes_.faces.size(), sizeof(fd.probes) / sizeof(float));
    std::memcpy(fd.probes, probes_.faces.data(), probeFloats * sizeof(float));
    // Cascade matrices. The near and far shadow passes read the same frame block with their own
    // sun matrix, so their vertex stage stays unchanged.
    const math::Mat4 sunNearVP = math::orthoVulkan(-40.0, 40.0, -40.0, 40.0, 0.1, 1200.0) * sunView;
    const math::Mat4 sunFarVP = math::orthoVulkan(-500.0, 500.0, -500.0, 500.0, 0.1, 1200.0) * sunView;
    storeMat(fd.sunNearViewProj, sunNearVP);
    storeMat(fd.sunFarViewProj, sunFarVP);
    // DDGI: refresh dynamic probe irradiance each frame so GI responds to time-of-day/weather.
    updateDynamicProbes();
    std::memcpy(fd.dynamicProbes, dynamicProbes_, sizeof(fd.dynamicProbes));
    std::memcpy(ubo_->data(), &fd, sizeof(fd));
    FrameData fdNear = fd;
    storeMat(fdNear.sunViewProj, sunNearVP);
    std::memcpy(uboShadowNear_->data(), &fdNear, sizeof(fdNear));
    FrameData fdFar = fd;
    storeMat(fdFar.sunViewProj, sunFarVP);
    std::memcpy(uboShadowFar_->data(), &fdFar, sizeof(fdFar));

    // The planar reflection's frame: the mirrored camera, clipping geometry below the sea.
    FrameData fr = fd;
    storeMat(fr.viewProj, mirViewProj);
    storeMat(fr.invViewProj, math::inverse(mirViewProj));
    fr.camPos[0] = toF(mirEye.x);
    fr.camPos[1] = toF(mirEye.y);
    fr.camPos[2] = toF(mirEye.z);
    fr.probeDims[3] = 1.0f;
    std::memcpy(uboRefl_->data(), &fr, sizeof(fr));

    oceanTime_ = timeSeconds;
    const auto t0 = clock::now();
    cmd_->begin();
    graph_.execute(*cmd_);
    cmd_->end();
    dev_.submitAndWait(*cmd_);
    // Auto-exposure: read back the single average-luminance texel and adapt the gain for next
    // frame (one frame of latency, standard for exposure adaptation).
    if (autoExposureOn_) {
        const auto lumBytes = readImage(graph_.image(hLumB_), 4);
        float avgLum;
        std::memcpy(&avgLum, lumBytes.data(), 4);
        if (std::isfinite(avgLum) && avgLum > 0.0f) {
            const float targetGain = std::clamp(0.18f / avgLum, 0.25f, 8.0f);
            autoExposure_ += (targetGain - autoExposure_) * 0.15f;
        }
    }
    if (profile_ && dev_.timestampsSupported()) {
        const auto names = graph_.passNames();
        const auto ticks = dev_.readTimestamps(static_cast<uint32_t>(2 * names.size()));
        const double nsPerTick = dev_.timestampPeriodNs();
        passMs_.clear();
        for (size_t i = 0; i < names.size(); i++) {
            const double ms = static_cast<double>(ticks[2 * i + 1] - ticks[2 * i]) * nsPerTick * 1e-6;
            passMs_.emplace_back(names[i], ms);
        }
    }
    lastFrameMs_ = std::chrono::duration<double, std::milli>(clock::now() - t0).count();
    prevViewProj_ = unjitteredVP;
    historyValid_ = true;
}

std::vector<uint8_t> OceanRenderer::readImage(rhi::Image& image, size_t bytesPerPixel) {
    const size_t bytes = static_cast<size_t>(image.width()) * image.height() * bytesPerPixel;
    auto buf = dev_.createBuffer(bytes, rhi::BufferUsage::Readback);
    auto cmd = dev_.createCommandList();
    cmd->begin();
    cmd->transition(image, rhi::ImageState::TransferSrc);
    cmd->copyImageToBuffer(image, *buf);
    cmd->end();
    dev_.submitAndWait(*cmd);
    std::vector<uint8_t> out(bytes);
    std::memcpy(out.data(), buf->data(), bytes);
    return out;
}

void OceanRenderer::updateDynamicProbes() {
    // DDGI dynamic probe grid: 4x4x4 probes over the beach/shelf region, matching the baked
    // probe grid's extent. Each probe stores world position + irradiance sampled from the
    // current sky/sun/atmosphere, so GI responds to time-of-day and weather changes.
    const int nx = 4, ny = 4, nz = 4;
    const float ox = -120.0f, oy = 1.5f, oz = -110.0f;
    const float sx = 240.0f / 3.0f, sy = (16.0f - 1.5f) / 3.0f, sz = 200.0f / 3.0f;

    for (int iz = 0; iz < nz; iz++) {
        for (int iy = 0; iy < ny; iy++) {
            for (int ix = 0; ix < nx; ix++) {
                const int index = ix + nx * (iy + ny * iz);
                const math::Vec3 pos{ox + ix * sx, oy + iy * sy, oz + iz * sz};

                // Probe validity: above sea level and inside the probe grid.
                const float valid = pos.y > 0.0f ? 1.0f : 0.0f;

                // Sample sky radiance along the probe's normal (up) for irradiance.
                // Uses the same atmosphere model as skyRadiance in common.glsl.
                math::Vec3 irradiance{0.0, 0.0, 0.0};
                if (valid > 0.5f) {
                    // Approximate hemispherical irradiance from the sky: sample 6 cardinal
                    // directions and average, weighted by cosine.
                    const math::Vec3 dirs[6] = {
                        {1.0, 0.0, 0.0}, {-1.0, 0.0, 0.0},
                        {0.0, 1.0, 0.0}, {0.0, -1.0, 0.0},
                        {0.0, 0.0, 1.0}, {0.0, 0.0, -1.0}};
                    for (int d = 0; d < 6; d++) {
                        const math::Vec3 dir = dirs[d];
                        const double cosT = std::max(0.0, dir.y);
                        if (cosT <= 0.0) continue;
                        // Sky radiance: use sun direction and transmittance for a rough estimate.
                        const double muSun = dir.y * sunUnit_.y + dir.x * sunUnit_.x + dir.z * sunUnit_.z;
                        const double Tsun = std::max(0.1, 1.0 - 0.5 * std::max(0.0, -muSun));
                        irradiance = irradiance + sunUnit_ * static_cast<double>(Tsun) * cosT * 0.15;
                    }
                    irradiance = irradiance * 0.5 + static_cast<double>(sunLux_) * 0.0005 * std::max(0.0, static_cast<double>(sunUnit_.y));
                }

                dynamicProbes_[index * 8 + 0] = static_cast<float>(pos.x);
                dynamicProbes_[index * 8 + 1] = static_cast<float>(pos.y);
                dynamicProbes_[index * 8 + 2] = static_cast<float>(pos.z);
                dynamicProbes_[index * 8 + 3] = valid;  // validity
                dynamicProbes_[index * 8 + 4] = static_cast<float>(irradiance.x);
                dynamicProbes_[index * 8 + 5] = static_cast<float>(irradiance.y);
                dynamicProbes_[index * 8 + 6] = static_cast<float>(irradiance.z);
                dynamicProbes_[index * 8 + 7] = 0.0f;  // pad
            }
        }
    }
}

}  // namespace renderer
}  // namespace engine
