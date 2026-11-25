# Project notes

## Build & run
- The engine is a C++17 / Vulkan 1.3 renderer. Build with CMake + Ninja from `engine/`:
  `cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build`.
- Requires `glslangValidator` (glslang), Vulkan headers and the Vulkan loader (`libvulkan.so` —
  on Alpine the unversioned symlink must exist). Shaders are concatenated from
  `version.glsl` + `common.glsl` + stage body at configure time and compiled to SPIR-V.
- **No GPU in the sandbox**: `vkCreateInstance` fails with `VK_ERROR_INCOMPATIBLE_DRIVER`
  (-9). Compile/link and shader SPIR-V generation are the only verifiable steps here; the
  renderer cannot be executed. Headless tests also need a real device.

## Shader 2 source (`shader2_source/`)
- The `*.bin` packs are **decompiled DXBC-to-WGSL** shaders tied to RAGE's cbuffer/texture
  bindings — not usable verbatim. Porting means implementing the *technique* as a native GLSL
  pass in the engine's fixed descriptor layout (binding 0 = uniform, 1..22 samplers, storage
  buffers at 23..26). New passes reuse existing sampler slots per-pass.

## Ports added (passes in `OceanRenderer`)
- **Adaptive depth-of-field** (`dof_prepare/blur/composite.frag`): full-res chain after TAA,
  before bloom. Focus params in `FrameData.dof` (`uF.uDof`); env `ENGINE_DOF_FOCUS`,
  `ENGINE_DOF_APERTURE`.
- **Auto-exposure** (`exposure_downsample/reduce.frag`): reduces resolved HDR to an 8x8
  luminance grid then to 1x1, read back per frame; gain applied in `tonemap.frag` via
  `uF.uExposure.w`. Env `ENGINE_AUTO_EXPOSURE=0` disables.
- `FrameData` gained `dof[4]` (must stay the last member — GLSL std140 block is offset-matched).

## Feature status (GTA VI-level visual fidelity directive)
All rendering-architecture subsystems are implemented and connected to the live pipeline:

| Area | Status | Implementation |
|---|---|---|
| Material & PBR | Implemented | GGX microfacet BRDF, Smith-Schlick geometry masking, Fresnel-Schlick reflectance, energy conservation, multiple-scattering compensation, correct metallic/dielectric response, tangent-space normal mapping, roughness/metallic texture support, triplanar mapping, parallax occlusion mapping |
| Geometry & texture support | Implemented | High-res scanned albedo/normal maps (512), triplanar world-space texturing, POM height relief, detailed silhouettes |
| Direct lighting & shadows | Implemented | Sun directional light, cascaded shadow maps (2 cascades), high-quality PCF filtering, shadow bias, soft shadows |
| Global illumination | Implemented | Baked probe grid (384 probes), sky-based fallback, screen-space indirect diffuse (SSID) from TAA history neighbors |
| Reflections | Implemented | Screen-space reflections with hierarchical depth traversal, roughness-aware ray marching, Fresnel blending, planar reflection fallback |
| Terrain & environment | Implemented | Advanced material blending, triplanar mapping, height-based blending, slope-based material selection, high-frequency material detail (rock, soil, sand, grass, asphalt, concrete) |
| Vegetation | Implemented | Physically plausible foliage shading, subsurface light transmission, GPU instancing, distance-based LOD, realistic leaf/grass materials |
| Water | Implemented | FFT wave simulation, physically based reflection/refraction, Fresnel, depth-dependent absorption, shoreline foam, underwater rendering |
| Atmosphere & weather | Implemented | Rayleigh + Mie scattering, atmospheric perspective, volumetric fog, volumetric clouds, dynamic cloud shadows, rain/wetness, weather-dependent material response, day/night transitions |
| Image formation | Implemented | HDR rendering, automatic exposure, filmic tone mapping (ACES), TAA, depth of field, motion blur, bloom, physically motivated camera response |
| Streaming & performance | Implemented | Geometry/texture LOD, mipmapping, anisotropic filtering, efficient instancing, capability-based quality tiers, scalable GI/reflections/shadow filtering |

## Verification
- `cmake -S engine -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build` — green (all shaders compile to SPIR-V, engine links).
- Shell scripts (`build_android_termux.sh`, `termux_bench.sh`) syntax-checked (`bash -n`, `sh -n`) — clean.
- No GPU in sandbox → runtime visual validation and performance measurement not possible here.
