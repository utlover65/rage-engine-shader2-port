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
