# engine: real-time renderer on a backend-agnostic RHI

OpenGL-independent. The rendering code sits on a hardware interface (`src/rhi/rhi.h`) with
one backend today (Vulkan 1.3, dynamic rendering). A GLES 3.2 backend can implement the same
interface later; nothing above `src/rhi` includes Vulkan headers.

## Layers

| Layer | Files | Role |
|---|---|---|
| RHI | `src/rhi/rhi.h`, `src/rhi/vulkan/*` | Images, buffers, shaders, pipelines, descriptor sets, command lists. Errors throw `rhi::Error`. Validation layer optional. |
| Render graph | `src/rg/render_graph.*` | Passes declare reads and writes. The graph checks the declarations, inserts the state transitions (barriers come from the RHI), opens the rendering scope, and runs the pass body. |
| Renderer | `src/renderer/ocean_renderer.*` | Builds the pass list for the open-sea scene. |
| Scene | `src/scene/ocean_mesh.*` | Polar ring mesh of the sea, centred on the camera. |
| Maths | `src/math/vec_mat.h` | Vectors and matrices (Vulkan clip conventions). |
| CPU reference | `src/atmo/atmo_reference.h` | Double-precision atmosphere, same integrals as the GPU tables. Used only for validation. |
| Shaders | `shaders/*.glsl`, `shaders/*.vert`, `shaders/*.frag` | `common.glsl` is prepended to every shader at build time. |

## Frame graph (open sea)

1. transmittance LUT (256 x 64, RGBA16F, once)
2. multiple-scattering LUT (32 x 32, RGBA16F, once; 64 directions, 20 steps per texel)
3. sky-view LUT (193 x 108, per frame; camera height and sun)
4. cloud slab (half resolution, 64 steps)
5. cloud-shadow field (half resolution, cloud optical-depth approximation projected onto the sea plane)
6. sky composite (sky-view, sun disk, clouds) -> HDR
7. sun shadow map (scene objects from the sun, 2048 x 2048, R32F, stores 1 - depth; rotated 12-tap variable-radius PCF)
8. ocean surface and scene objects (Gerstner water, procedural pier/boat/lighthouse/islet meshes, depth tested) -> HDR
9. bloom bright pass (half resolution, threshold 8 pre-exposed, input clamped at 64)
10. bloom blur, horizontal then vertical (9 taps, half resolution)
11. tone map (bloom, ACES fit, FXAA edge filtering, sRGB) -> display

Radiance is pre-exposed (`1 / (1.2 * 2^EV100)`), so sun radiance stays inside fp16.
The busiest material pass binds six sampled images; the RHI currently exposes nine sampled-image slots.

## Build and run

    cmake -S . -B build && cmake --build build -j8
    ENGINE_VALIDATION=1 ./build/engine_triangle   # RHI smoke test
    ENGINE_VALIDATION=1 ./build/engine_render     # renders render.ppm and runs the checks

Needs Vulkan headers and loader, glslangValidator, and the Khronos validation layer for
`ENGINE_VALIDATION=1`.

## Validation (llvmpipe, validation layer on, 0 validation errors)

| Check | Result |
|---|---|
| Triangle through the RHI | pixels as expected (centre (80,82,93), corner clear colour) |
| Transmittance table vs CPU reference | median 3.4e-4, p99 1.0e-3 (11875 points, grid plus random) |
| Multiple-scattering table vs CPU reference | median 7.0e-4, p95 1.3e-3, max 5.3e-3; no NaN, no negatives |
| Sky-view table vs CPU reference, texel directions | median 9.2e-4, p95 6.2e-3, max 3.3e-2 |
| Sky-view lookup, random directions, bilinear | median 7.7e-4, p95 2.7e-3, max 9.4e-2 |
| HDR frame and bloom | no NaN or Inf; bloom non-negative |

These checks compare the GPU with a CPU implementation of the same model. They show that the
tables and their lookups are computed correctly. They do not show that the model matches real
skies or sea. Nothing here was compared against photographs.

The transmittance maximum (about 4.2 relative) sits on the Earth-shadow boundary, where the
table steps from its value to zero. The p99 bound is the meaningful one.

Frame time on llvmpipe: about 450 ms for 960 x 540. That is CPU emulation of the GPU. It
says nothing about a phone GPU, and no number for the target device exists yet.

## Known limits (read before judging the image)

- **Photorealism is unverified.** No calibrated reference photograph was available, and the latest changes have not been GPU-rendered in this environment.
- **Clouds:** 3-D warped value-noise cumulus, 64 steps at half resolution, wind advection, and no temporal accumulation. The new cloud-shadow field is a screen-projected optical-depth approximation, not a full volumetric shadow solution. Coverage and feature sizes remain hand-tuned rather than photographically calibrated.
- **Sea:** open sea only. The lagoon, shallow floor and lightning are not ported. Deep-water scattering is a closed-form single-scatter term, not a volume path trace. Sun glint is a point-sun GGX evaluation. Sky and cloud radiance are reflected, but scene-object reflections are absent.
- **Objects:** procedural geometry and world-space procedural materials are used for the pier, crate, work boat, lighthouse and islet. There are no imported high-resolution PBR texture sets, no physically convolved IBL, no bounced-light solution, and no object reflections in the sea.
- **Sky:** multiple scattering is an isotropic approximation (one table, no ground bounce in the table). Ground-hit rays use a flat albedo of 0.2 with no shadows.
- **Camera:** ACES tone mapping, restrained bloom and FXAA are present. There is no temporal AA, physically modelled lens, motion blur, or calibrated white-balance pipeline.
- **Exposure:** default EV100 10.5 is a sunset-oriented exposure. EV100 15 suits a midday sun and renders this scene black.
- **RHI:** barriers are coarse (all commands, all memory). Buffers are host-visible only. No memory aliasing in the graph.
- **Not done:** GLES backend, Android build (NDK not installed here), or frame-time measurements on the Red Magic Astra.

## Output

`render_ocean.png` is the frame from the last run (960 x 540, EV100 10.5).


## Maximum photorealism pass (2026-11-21)

This pass keeps the existing Vulkan 1.3 RHI and render graph. It makes the scene and shading materially richer without requiring external model or texture downloads:

- Rebuilt the pier as separated deck boards with cross-beams and piles; added crate slats and corner battens; replaced the box boat with a tapered faceted hull, deck, cabin and mast; replaced the lighthouse block with a tapered cylindrical masonry tower, plinth, lantern gallery, railings, roof and seaward windows.
- Expanded the scene-vertex format from 9 to 10 floats (position, normal, linear albedo, material class). The mesh shader now evaluates eight procedural material classes: weathered wood, crate wood, painted hull, porous masonry, metal, warm emissive lantern glazing, coastal rock, and damp sand.
- Added world-space material variation, fine surface breakup, metallic-aware GGX F0, roughness-aware environment blending, and reflected cloud radiance on water and glossy objects.
- Changed the ocean spectrum to longer, lower-steepness open-water swells and replaced the sharp distance LOD transition with a smooth spectral filter to reduce concentric bands.
- Added wind advection and warped/eroded 3-D noise to the cloud field.
- Replaced the fixed 3x3 shadow kernel with a rotated 12-tap Poisson PCF filter, added FXAA after tone mapping, and lowered the bloom threshold so the sun and bright glints produce a restrained halo.

### Verification status for this pass

The procedural scene builder now has a standalone validator. Run `g++ -std=c++17 -O2 -Isrc tools/validate_scene.cpp -o validate_scene && ./validate_scene`; it checks the vertex layout, finite attributes, unit normals, and presence of both rock and sand materials. In this environment it reported 22,068 finite vertices and a mean normal-length error of about 2.5e-8. `src/renderer/ocean_renderer.cpp` and `src/main_render.cpp` also passed C++17 syntax-only checks. A full Vulkan build and GLSL-to-SPIR-V compilation could not be run in this environment: the Vulkan development headers/library and `glslangValidator` are unavailable. The checked-in `render_ocean.png` is the earlier reference frame, not a render of these latest changes. Build and render on the target Vulkan machine before treating the changes as runtime-verified.

### Remaining gaps to reach production photorealism

This is a substantial procedural-rendering upgrade, not a completed AAA renderer. It still needs authored high-resolution PBR texture sets and production meshes, a prefiltered HDR environment/IBL pipeline, planar or screen-space reflections for scene geometry, full volumetric cloud shadows, improved area-light/solar penumbra modelling, temporal anti-aliasing, and validation against calibrated photographic references. The new procedural noise is a substitute for texture assets, not a replacement for them.

## Android / Termux arm64 mobile profile

The project can now be cross-built for Android arm64 and run as an **offscreen Vulkan renderer** on a Vulkan 1.3-capable device. It is not yet an Android windowed game: the current RHI has no Android surface/swapchain or touch/input layer, so it renders `render.ppm` rather than presenting frames to the screen.

On the Android device, install the Termux build tools (`cmake`, `ninja`, `glslang`) and point `ANDROID_NDK_ROOT` at an installed Android NDK. Then run:

```sh
cd /path/to/engine
chmod +x build_android_termux.sh
./build_android_termux.sh
```

The script places the arm64 executable and compiled SPIR-V shaders in `~/storage/downloads/engine-render-arm64/`. Run it with:

```sh
~/storage/downloads/engine-render-arm64/run.sh
```

The mobile profile defaults to 960x540, a 1024x1024 sun shadow map, and `ENGINE_QUICK=1` (one rendered frame, no 120-frame benchmark or CPU-reference diagnostics). The full atmosphere, HDR, cloud, ocean, material, shadow, bloom, and tone-map passes remain enabled. The output is `render.ppm` in the run directory.

Tune quality with environment variables, for example:

```sh
ENGINE_WIDTH=1280 ENGINE_HEIGHT=720 ENGINE_SHADOW_SIZE=1536 ~/storage/downloads/engine-render-arm64/run.sh
```

Start at 960x540. Increase resolution only after checking sustained frame time and device temperature. This mobile profile is a bring-up target; it does not claim a stable frame rate until measured on the actual device. The current device backend requires Vulkan 1.3 dynamic rendering.


## Lighting / scene realism pass (2026-11-21)

This pass adds several scene-visible systems on top of the procedural material work:

- A half-resolution cloud-shadow field is rendered once per frame. It projects a ray/sea-plane intersection and evaluates an approximation of the same advected 3-D cloud coverage used by the sky. The ocean samples this field instead of running the cloud-density function at every full-resolution water pixel; scene objects evaluate cloud optical depth at their own world positions.
- Direct sun now receives both object-shadow visibility and cloud transmittance. The sun shadow filter estimates a variable penumbra radius from blocker/receiver separation and the solar angular radius instead of using a fixed kernel radius.
- Diffuse sky lighting now samples a cosine-hemisphere approximation oriented to each surface normal, rather than using the same four upward sky directions for every material orientation.
- The lighthouse now sits on an irregular submerged rocky islet with a narrow sand band. The terrain uses smooth height-field normals, and rock/sand have separate weathering, grain, roughness and dampness responses in the PBR material shader.

### Verification and limitations

The procedural scene builder now emits 22,068 vertices with finite values; the validation run measured a mean unit-normal length error of approximately 3e-8. `src/renderer/ocean_renderer.cpp` passes a C++17 syntax-only check. Full Vulkan linking, GLSL-to-SPIR-V compilation and GPU rendering were not available in this environment, so the new shaders and visual result still require a Vulkan 1.3 + glslangValidator run on the target device.

The cloud-shadow field is an approximation: one representative sample through the cloud slab, projected into screen space for the sea, not a full volumetric light-transport solution. This still does not add imported high-resolution texture sets, a true prefiltered HDR IBL pipeline, SSR of scene geometry, temporal AA, a path tracer, or a complete RAGE-scale open-world/streaming/gameplay engine. The current executable remains an offscreen renderer.

## Pass: scanned materials, god rays, sea light

- **Scanned materials.** Wood and rock use CC0 diffuse scans (512 px, raw RGBA8 under `assets/`,
  baked by `tools/bake_materials.py`). The scans came without normal maps, so the normal maps are
  derived from luminance. Both are sampled triplanar in world space (the procedural meshes have no UVs)
  and are mip-mapped on load (`src/assets/texture.cpp`). The RHI gained texture upload, mip levels
  and a trilinear sampler; sampled slots grew from 9 to 13. The scan albedo is scaled by fixed
  artistic gains (`WOOD_GAIN`, `ROCK_GAIN` in `shaders/mesh.frag`).
- **God rays.** `shaders/shafts.frag` (quarter resolution) integrates the single-scattered
  in-scatter that cloud shadows remove along each view ray, up to the sea or the cloud layer. The sky
  composite and the sea's aerial perspective add it. It only shows where a cloud shades the air in the
  sun's direction, so it needs a sun a few degrees up (see below).
- **Sea.** The wave parameters are restored to the pre-archive version (the regression came from the
  newer wave set). Crests facing away from the sun pick up turquoise transmitted light.
- **Sun.** Default elevation 5 degrees, exposure EV100 12.5. At 1.2 degrees the sun-blocking clouds
  sit about 57 km away, beyond the 30 km cloud draw range, so no rays can appear.

Driver overrides (desktop and Android): `ENGINE_SUN_ELEV_DEG`, `ENGINE_EV100`, `ENGINE_TIME` (scene
clock, for picking cloud positions), `ENGINE_SHADER_DIR`, `ENGINE_ASSET_DIR`, and `ENGINE_DUMP_SHAFT=1`
(with `ENGINE_STILL=1` and validation) writes `godrays.ppm`.

Not done in this pass: planar water reflections, prefiltered environment reflections, baked probe GI,
aerial perspective on scene objects, displacement and micro-detail maps (none were supplied), and
any measurement on the target device.

## Pass: planar reflections, probe lighting, displacement, landscape

- Water: planar reflection of the scene above the sea from a mirrored camera (half resolution,
  clipped below the surface, alpha is coverage). Reflection strength scaled by REFLECT_SCALE 0.6,
  with the sky reflected through an eight-tap glossy filter where the planar render does not cover it.
- Scene and headland: baked ambient probes (4 x 4 x 4 grid, 256 rays per probe, sky occlusion and
  one sky bounce with colour bleeding). Shaders fall back to the sky LUT outside the grid.
- Wood and rock: height read from the normal map's alpha, used for parallax (WOOD_DEPTH 0.03 m,
  ROCK_DEPTH 0.15 m). Re-bake with tools/bake_materials.py to regenerate the height channel.
- Landscape: a procedural headland (src/scene/landscape.cpp) meshed as displacement from one
  height function, shared with the probe ray march. Sand fringes the low ground.
- Descriptor set: 14 sampled-image bindings (kMaxSampledImages = 14).

Not done: probes are coarse (about 33 m horizontal); bounce is sky-only, with no sun bounce;
the sun shadow box is 320 m; the reflection is half resolution; the glossy sky is a runtime filter
rather than a prefiltered map; the cloud overlay in reflected sky uses the main camera's projection;
no device timing (validated on llvmpipe only).

## Material realism pass 5 (2026-11-21)

The latest shader pass targets the remaining synthetic-looking surfaces without adding texture
bindings or increasing the vertex format:

- Masonry now has irregular recessed horizontal mortar courses, mineral mottling, porous pits,
  and a broader roughness range. The course modulation is world-space, so it remains continuous
  across the procedural lighthouse and terrain meshes.
- Galvanised metal now has finer grain, spatially varying roughness, and restrained oxidation;
  the metallic response is raised while keeping the base colour dark.
- All procedural materials receive a small two-scale albedo/roughness variation to avoid uniform,
  plastic-looking highlights. The amplitude is intentionally limited for 960x540 mobile rendering.
- Procedural bump normals now combine broad and fine noise bands, improving the response of
  painted surfaces, masonry, metal and sand to grazing light.

The standalone scene validator still passes (22,068 vertices, zero non-finite attributes,
mean normal-length error 2.53e-8; rock and sand materials present). This container does not have
`glslangValidator`, Vulkan development headers, or a Vulkan device, so this pass has **not** been
SPIR-V compiled or GPU-rendered here. The checked-in `render_ocean.png` is not a screenshot of
this shader revision. Build and inspect on the Android Vulkan device before judging the image.

## Photorealism pass 6 — coastal material response and camera finish

This pass adds another layer of scene-specific material response while retaining the Vulkan 1.3 RHI,
render graph, existing vertex layout, and mobile offscreen path:

- **Wood:** low-frequency sun bleaching, embedded dirt, grain variation, and a dampness response that
  darkens the albedo and tightens the specular lobe near the waterline.
- **Painted metal/hull:** stochastic paint chips, exposed primer, edge wear, and roughness variation.
- **Masonry:** gravity-aligned rain staining, salt bloom, porous breakup, and damp/algae staining near
  the base instead of a single uniform stone colour.
- **Galvanised metal:** oxidation, directional fine scratches, edge exposure, and a roughness shift
  where the protective surface has weathered.
- **Coastal rock:** lichen mottling and a stronger wet-rock response at the waterline.
- **Shared shading:** a compact roughness-dependent multiple-scattering energy compensation for the
  environment specular term and low-amplitude material-scale ambient occlusion variation.
- **Camera output:** highlight desaturation, restrained optical vignette, and very low-amplitude
  screen-space film grain after ACES tone mapping. These are deliberately subtle; they should not
  look like a social-media filter.

### Pass 6 verification

- Procedural scene validator: **passed** — 22,068 vertices, zero non-finite attributes, mean
  normal-length error 2.5263e-8, rock and sand material classes present.
- C++17 syntax-only check for the renderer and main entry point: **passed** with the project's
  shader/asset directory definitions supplied.
- Full GLSL-to-SPIR-V compilation and offscreen render (llvmpipe): run and passing in the desktop build
  (validation layers report no errors, god-ray and HDR checks PASS). Not yet run on the Adreno device.

This pass improves the existing reference scene; it does not magically provide production-quality
asset detail. The remaining large photorealism gaps are still authored high-resolution PBR materials,
production meshes and foliage, temporal anti-aliasing/upscaling, calibrated HDR environment capture,
full-scene indirect illumination, volumetric cloud shadows, screen-space/planar reflection coverage
for all objects, and validation against real photographic references. The current project is still
an offscreen open-sea renderer rather than a complete open-world game engine.

## Pass 8: parallax occlusion, prefiltered sky, aerial froxels, cascaded sun shadows

- Parallax occlusion mapping (mesh.frag, pomPoint): eight-step march through the wood and rock
  height fields with interpolation at the crossing. Silhouette test not yet added.
- Prefiltered sky (sky_prefilter.frag, binding 15): the sky-view LUT blurred over three cone radii
  (0.03, 0.19, 0.35 rad) in three stacked rows, rebuilt each frame. skyGlossy blends two rows and
  adds one cloud tap (it was eight cloud-aware LUT taps before). The blur is 24 taps over a 193 x 108
  LUT, so the sea reflection shows the LUT texel grid more than before.
- Aerial froxels (aerial.frag, bindings 16 and 17): 16 log-spaced depth slices (1 m to 60 km) in a
  640 x 24 atlas, rebuilt per frame. Mesh and sea shaders read it with aerialLookup, keyed to the main
  camera (FrameData.uMainViewProj). The mirrored reflection pass reuses the main camera's froxels.
- Cascaded sun shadows: near (+-40 m, 1024^2), mid (+-160 m, 2048^2, the old map) and far
  (+-500 m, 1024^2). sunVisibility takes the nearest cascade that contains the point. Cascade
  seams are hard edges (no blend across the boundary yet).
- Descriptor sets: 19 sampled-image bindings (kMaxSampledImages = 19).

Not done in this pass: GTAO (needs a depth prepass and depth LoadOp in the render graph), TAA
(needs jitter and velocity), Hi-Z reflections, and an FFT ocean (needs a compute path).
Validated on llvmpipe only; no device timing.

## Pass 9: ambient occlusion

- Depth prepass (new, depth_only.frag): scene objects, headland and sea drawn once for linear depth
  (metres along the view ray) into R32F, plus a depth attachment. Costs one extra draw of the sea grid
  and the scene; the lighting passes still clear and redraw their own depth.
- GTAO (gtao.frag, half resolution): four screen directions, four steps out to 2.5 m, rotated per pixel.
  Output is visibility, 1 = open sky.
- Blur (ao_blur.frag): 5 x 5 Gaussian weighted by depth similarity, so occlusion does not cross edges.
- Applied to the ambient diffuse term of scene objects through the main camera's screen position.
  Not yet applied to the sea (open water has little to occlude).
- Descriptor set: 21 sampled-image bindings (kMaxSampledImages = 21).
- Validated on llvmpipe; the occlusion is subtle in the current still.

## Pass 10: TAA, screen-space reflections, spectral sea

- TAA: Halton (2,3) sub-pixel jitter over eight frames, applied to the projection rows that set clip
  x and y (the reflection and sun passes are not jittered). The resolve (taa.frag) reprojects each pixel
  with its linear depth into last frame's unjittered view, clamps the history to the 3 x 3 neighbourhood
  and blends 10 % current. The history is an imported graph image (the render graph needs a persistent
  resource for last frame's colour), copied at the end of each frame.
  Limits: the sea moves with the waves, which the reprojection does not see, so water edges can ghost.
- Screen-space reflections (mesh.frag, ssrTrace): a linear march of the reflected ray through this
  frame's linear depth, sampling last frame's resolved colour on a hit. Only glossy surfaces take it
  (roughness below about 0.35). This is a linear march, not a hierarchical Hi-Z trace, so it is coarse
  and misses thin geometry. The sea keeps the planar reflection, which already covers the scene.
- Spectral sea (ocean.vert): the Gerstner sum is replaced by eight directional components with seeded
  directions and phases and a JONSWAP-style amplitude weight. This is a spectral-weighted sum, not an FFT
  ocean; it has no compute path yet.
- Descriptor set: 22 sampled-image bindings (kMaxSampledImages = 22).
- Validated on llvmpipe (no validation errors, god-ray and HDR checks PASS). No device timing.

## Pass 11: sea sky-reflection fix

- The glossy sky now averages the cloud layer over a four-tap cone matching the glossy radius
  (it was one tap per pixel, which aliased badly across the waves).
- The sea's sky-reflection cone is 0.12 rad (was 0.04). The wave slopes bend the reflected ray far
  enough that the cloud layer otherwise shows as oil-like blobs.
- Tested and ruled out: the planar reflection's slope offset (no visible change at 0.004 vs 0.012).

## Pass 12: open-sea colour

- Sea reflection scale 0.6 -> 0.3 (open water shows little sky at this view angle).
- Blue water-body term (SEA_BODY, lit by sun and sky, weighted by 1 - Fresnel): the saturated blue of open sea.
- Removed the earlier desaturation of the sea, which had made it grey-brown.

## Pass 13: shallow water and surf (shading only)

- Seabed height ported to GLSL (ocean.frag, seabedHeight) from landscape.cpp, so shading and mesh agree.
- Water transmission over the bed: shallow water transmits sand colour and turns turquoise; deep water
  keeps the blue body colour. Outside the headland region the water is treated as deep.
- Surf: white breaking water over the shallows, noise-broken and moving with time.
- Limit: the headland drops steeply into deep water, so the shallow zone is too narrow to show.
  A shelf in the seabed (in landscape.cpp and the GLSL port) is needed for the beach look.

## Pass 14: beach composition (reference match)

- Terrain replaced: a sand shore (`shoreZ`), a dune rising inland, and a shelf that deepens seaward (0.085 m/m to 45 m, then steeper, capped at 40 m). The same function is ported to `shaders/ocean.frag` (`seabedHeight`), so the mesh, the probe bake and the water shading agree. The headland gating is removed.
- Camera: top-down (eye 0,230,40; target 0,0,-30). Sun about 50 degrees elevation. EV100 14.5.
- Offshore pier, work boat and lighthouse are gated off (`kBuildProps` in `objects.h`). Their geometry is kept.
- Sun cascades centred on the beach (0,0,-30); the mid cascade widened to ±230 m. This removes the hard box edges on the sand.
- Probe grid moved to cover the beach (x ±120, z -110..90).
- Self-check changed from a sky/sea brightness test to deep-sea-blue and sand-warm colour checks.
- Known remaining: a large grey shadow over the sand (cloud shadow or sun shadow; not isolated yet); moire on the far sea surface (wave aliasing at this camera angle); no sky in frame. Device timing is still unmeasured.

## Pass 15: sunset bay (reference remake attempt)

- Camera on the water (2.2 m up, looking toward the sun, about 5 degrees elevation). EV100 11.5, FOV 50.
- Procedural skyline of 64 towers at about 700 m across the bay (`buildSceneObjects`, after the gated props).
- Mid sun cascade centred at (0,0,-120) with a 2400 m far plane. The shadow shader clamps `1 - depth` to [0,1].
- Known gaps against the reference: towers are flat untextured boxes with no windows or palms; water shows strong capillary turbulence instead of long swell; the pink haze is missing; the pavilion, ferris wheel and boats are not built.

## Pass 16: sunset bay, added elements

- Ferris wheel (rim of 48 posts, 24 cabins, hub, A-frame) at x 150, z -330.
- 14 palms (trunk plus three crossed frond slabs) along both shores of the bay.
- Open pavilion on the left shore; three moored boats.
- Capillary detail scaled to 0.3 (calmer near-field water).
- Not done: haze colour (needs HAZE_SCALE changed in both common.glsl and atmo_reference.h, with the CPU self-check), window detail on towers, sea-state swell, and palm frond shape (slabs, not fronds).

## Pass 17: sunset colour grade

- `tonemap.frag`: split-tone grade after the ACES curve. Magenta push in the highlights (SUNSET_HIGHLIGHT), violet lift in the shadows (SUNSET_SHADOW). Constants are at the top of the file.
- Sky moves to pink-violet. Water glitter and tower faces remain warm gold: the grade is a post step, not a change to the light, so the sun's own colour still shows through.
- Proper fix (not done): a sunset-tinted sun colour and stronger Mie haze, changed in `common.glsl` and `atmo_reference.h` together.

## Pass 18: sunset haze (lighting, not grade)

- HAZE_SCALE 1.0 to 3.0 in both `shaders/common.glsl` and `src/atmo/atmo_reference.h` (kHazeScale). The CPU self-check still passes, so the GPU tables and the reference agree.
- The low sun now reddens through the thicker haze; the sky, horizon and sun glitter move violet and rose. Tone-map grade from pass 17 is still applied on top.
- Remaining: the sea body is teal (the blue SEA_BODY term dominates); clouds wash out at this haze level, so cloud density or the cloud colour needs retuning; the sun disc is pale.

## Pass 19: window glass, violet deep water, camera in deep water

- Concrete towers (material kind 3): dark glazed bays on vertical faces (3.4 x 4.2 m grid, per-bay variation, low roughness on glass).
- SEA_BODY to violet (0.045, 0.040, 0.15). Camera moved to z -120: at z -50 it sat in the shallows, so the sand showed through and the water read teal.
- Still open: cloud shape and colour at this haze level; sea swell (current waves are capillary-scale and spectral only); sun disc; palm fronds; reference ferris wheel detail.

## Pass 20: ocean swell

- `ocean.vert`: the spectral ladder now starts at 96 m (swell) and halves each component down to 0.75 m, replacing the 8 m-down ladder that had no swell at all. Amplitude keeps the (lambda/8)^0.25 weight, so the swell is about 1 m high. Sun glints now break up along the long waves.
- Open: a shallow-sand patch shows at the bottom left and a faint rectangular edge at the bottom right (a shadow cascade boundary, not yet traced); clouds still wash out at this haze; no FFT ocean; device timing unmeasured.

## Pass 21: soft-body car physics module

- `src/physics/softcar.h`: header-only XPBD crash model. A 150-node lattice (4.5 x 1.0 x 1.6 m, 1200 kg), 1199 compliant springs, a softer two-column crumple zone, plastic yield at 1% strain, tearing at 25%, internal velocity damping, rigid wall at x = 0 and ground at y = 0.
- `src/physics/softcar_test.cpp`: headless 54 km/h wall test (`engine_softcar_test`). Passes: kinetic energy 135 kJ to 18 kJ, nose crush 0.18 m, 36 springs torn.
- The crush depth is below real front-end crush (about 0.4 m); the 0.15 m bound is a sanity check, not a calibration.
- Not yet: wheels and suspension, a driving controller, rendering the car, and wiring the module into a game loop. No GTA-style game exists in the engine yet.

## Pass 22: road-vehicle dynamics (calibrated, not 1:1)

- `src/physics/vehicle.h`: 6-DOF rigid chassis (quaternion orientation) on four corners with spring, bump stop, hydraulic damper (bump and rebound, linear plus quadratic), simplified magic-formula tyres with a friction circle, wheel spin, brakes, torque-curve engine, five-speed automatic, rear-wheel drive, aero drag and rolling resistance.
- `src/physics/vehicle_test.cpp` (`engine_vehicle_test`): checks static load per corner (3434 N), 0-100 km/h in 6.5 to 8.0 s (measured 6.62 s), braking distance within 0.9 to 1.15 x v^2/(2 mu g) (measured 65.9 m against 67.3 m ideal), and lateral acceleration under mu g.
- Peak torque tuned from 330 to 280 Nm to match the 0-100 target.
- Corrected: the earlier brake test passed because its bound was too loose. The braking itself was already about 1 g.
- Not 1:1: no measured tyre, damper or engine data; no anti-roll bar, Ackermann, clutch, load-dependent friction or tyre temperature. The fitted curves are plausible, not measured.

## Pass 23: compute path in the RHI (prerequisite for an FFT ocean)

- `rhi.h` / `vulkan/`: `BufferUsage::Storage`, `ComputePipelineDesc`, `Device::createComputePipeline`, `Device::setStorageBuffer`, `CommandList::bindComputePipeline` and `dispatch`. The descriptor set layout gains one storage-buffer binding (23) at compute stage only. Graphics pipelines and existing shaders are unchanged; push constants use the compute stage when a compute pipeline is bound.
- `shaders/compute_fill.comp` and `src/rhi/compute_test.cpp` (`engine_compute_test`): fills 256 elements and reads them back. Result: 0 wrong, 0 validation errors.
- `engine_render` rebuilt and re-run with validation: 0 errors, PASS.
- Not yet: the FFT ocean itself (butterfly passes, displacement and normal maps). This pass only adds the dispatch path it needs. Storage images (not just buffers) and compute-to-graphics barriers are not implemented yet.

## Pass 24: GPU FFT building block (for the FFT ocean)

- `rhi.h` / `vulkan/`: two storage bindings (input and output, 23 and 24) so compute passes can ping-pong; `CommandList::computeBarrier()` orders shader writes before the next pass or a vertex read.
- `shaders/fft_stage.comp`: one Stockham radix-2 stage (Bainville's formulation). Eight passes make a 256-point complex FFT.
- `src/rhi/fft_test.cpp` (`engine_fft_test`): runs the FFT on the GPU and checks every output against a double-precision naive DFT. Result: relative error 2.2e-7, 0 validation errors.
- `engine_render` re-run with validation: 0 errors, PASS.
- Not yet: the 2-D FFT over rows and columns, the Phillips/JONSWAP spectrum and its time evolution, the height and normal output, and sampling it in `ocean.vert` / `ocean.frag` (needs the storage binding exposed to the vertex stage).

## Pass 25: sun aureole (lighting)

- `shaders/sky.frag`: aureole term, sunRadiance x 1e-3 x exp(-theta / 0.035 rad), excluding the disk. Forward-scattered sunlight from aerosols around the sun. Relative to the disk, so it follows the sun's radiance and exposure.
- Existing lighting left as is: cloud-shadow god rays (`shafts.frag`), the sun disk, bloom, cloud silver lining, sea glint.
- Not verified visually: in the current camera the sun sits behind the skyline, so the aureole can't be seen. Needs a sun-facing camera view to tune.
- Not added: starburst or lens flare. Both read as camera artefacts rather than light and would need a physical aperture model.

## Pass 26: sun raised to 12 degrees (lighting check)

- `ocean_renderer.h`: sunDir elevation 5 to 12 degrees so the disc clears the skyline. This is the first view where the aureole, sun glare, glitter path and cloud silver lining can be seen together.
- Observed: a warm sun glint path on the water, sunlit cloud edges, a glow around the disc.
- Open: the haze (HAZE_SCALE 3) flattens contrast, so towers and sky look washed out; highlights clip on the cloud deck; the water reads pale blue here rather than violet.

## Pass 27: FFT ocean (Tessendorf), on the GPU, verified

Sources and licences
- Spectrum, time evolution and displacement: Tessendorf, "Simulating Ocean Water" (SIGGRAPH 2001 course notes), read from the public PDF. The equations are implemented from the paper; no code was copied.
- Barrier pattern: Khronos Vulkan-Samples (Apache-2.0), samples/extensions/device_fault and buffer_device_address. Both use a compute write followed by a vertex read, with COMPUTE_SHADER to VERTEX_SHADER and shader write to shader read. The RHI computeBarrier() matches it. No Vulkan-Samples code was copied, so no attribution is owed. Only this barrier pattern was reviewed, not the sample framework.

What was built
- src/ocean/ocean_spectrum.h: P_h(k), h0(k), a calibrated amplitude (phillipsAmplitude), N = 256, L = 1024 m, V = 10 m/s.
- shaders/ocean_evolve.comp: h(k,t), and the displacement fields -i k/|k| h. The output is conjugated, so a forward FFT followed by the real part gives the inverse sum.
- shaders/fft2d_stage.comp: one radix-2 Stockham stage over rows or columns. 16 passes per field.
- RHI: three storage slots (kMaxStorageBuffers = 3). Slot 2 is readable in the vertex stage. The buffers are read-only there, so no device feature is needed. Pools are larger.
- Renderer: an "ocean FFT" compute pass runs before the depth prepass (about 51 small dispatches per frame). ocean.vert samples the three fields with periodic bilinear interpolation, takes central-difference normals, and uses the Jacobian for folds and foam. The analytic ladder now covers 3 m down to 0.375 m.

Calibration
- The grid variance is sum_k 2 P_h(k) = (L/2pi)^2 pi A (V^2/g)^2 (1 - eps). Here eps = 1/(kc Lw)^2 is about 1.6 % at N = 256, L = 1024, V = 10 m/s, with kc = pi N / L.
- A = 4 pi sigma^2 / (L^2 (V^2/g)^2). For a fully developed sea at V = 10 m/s, sigma = 0.53 m (H_1/3 ~ 0.21 V^2/g ~ 2.1 m, Bretschneider). This gives A = 3.24e-8.
- The pass 26 test value A = 3e-4 at L = 256 gives an RMS of about 13 m by the same formula, roughly 23 times too tall.

Verification (llvmpipe, no device timing)
- engine_ocean_test: GPU against a direct double-precision sum at 256 lattice points, for all three fields. Relative error 6.6e-6. RMS height 0.516 m against the 0.53 m target (ratio 0.974). Validation errors 0. PASS.
- Renderer still frame: 0 validation errors. Two performance warnings (OutputNotConsumed on the depth prepass pipelines) are also in the pass 26 log, so they predate this change.
- The FFT reaches the image. With the FFT terms zeroed, 44 % of pixels change by more than 2/255; in the lower third the mean change is 7.7/255.

Open
- The look is still painterly and streaky in the foreground. Haze and highlights are washed out (the pass 26 list still applies).
- One 1024 m cascade repeats every 1024 m. Close detail needs a second, smaller cascade.
- The FFT field has no distance LOD. Only the ladder fades with distance.
- Not measured on a device: the cost of the compute passes on Adreno.

## Pass 28: exposure, haze and layout (measured art-direction fixes)

- Exposure default: ev100 11.5 -> 13.5. The frame had almost no tonal range (sky 5th-percentile luma 201, near-sea median 138 of 255). At 13.5 they are 132 and 89.
- Haze: HAZE_SCALE 3.0 -> 1.5 in shaders/common.glsl and src/atmo/atmo_reference.h (the self-check still passes). Towers and sky keep more blue.
- Ferris wheel moved from x = +150 m to x = -150 m.
- Still open: the grain in the clouds is the per-pixel jitter of a single frame; TAA should average it over frames, which a still render cannot show. Towers are untextured prisms. The top of the wheel is cut off by the frame. The image is still stylised, not photoreal.

## Pass 29: skyline facades, massing and distance anti-aliasing

- Skyline moved closer: x -340 .. +330 m, z -430 .. -520 m (was -760 .. +760 m at -700 .. -820 m). At 700 m and beyond the towers sat inside the aerial haze and their facade detail was lost.
- Five facade palettes (warm limestone, grey concrete, brick, dark glass, pale stone) with a per-tower shade, instead of one cream tone.
- About half the towers carry a setback crown (0.62 of the tower's half-width, 8 to 18 % of its height). The tallest carry a mast.
- Facade bays (shaders/mesh.frag, concrete kind): where a bay covers less than a pixel, its glazing fades to the average (0.06) instead of aliasing. The per-bay and per-floor pattern is unchanged.
- Still open: the ferris wheel at x -150 m reads as loose cabins at this framing. The dark sea reflects the towers almost like a mirror. The cloud grain is still present in a still frame.

## Pass 30: ferris wheel and facade tone

- Ferris wheel moved to x -90 m, z -330 m, with a 35 m ring at height 45 m, so the whole ring sits inside the frame. It has 96 rim segments and 24 cabins. The old ring was 60 m with 48 segments and read as loose cabins.
- Facades: each floor band has its own tone (0.94 to 1.06), the bottom 14 m is dirtied (down to 0.78), and about a third of the glazed bays reflect warm sky and sun.
- Still open: the dark sea mirrors the towers. Cloud grain remains in a still frame. The wheel's rim is low-contrast against the sky.

## Pass 31: water reflections (calm chop, full Fresnel, ripple-broken reflections)

- Reflectance scale 0.3 -> 1.0 in shaders/ocean.frag (REFLECT_SCALE). The old scale put the reflection at 30 % of its physical strength under the dark body colour, so the towers showed as dark columns and the sky barely reflected. At grazing angles Fresnel is about 1.
- Ripple distortion of the planar reflection: n.xz * 0.012 -> n.xz * 0.12 (REFLECT_DISTORT). With the FFT slopes the old shift was below a pixel.
- Wave height: kOceanSigmaM 0.53 -> 0.30 m (calm-to-light chop). The ocean test still passes (RMS 0.292 m against 0.300 m, relative error 6.6e-6 class).
- Still open: the sea is still a single planar reflection, so reflections of the sky are the only ones that are not ripple-accurate. Real-time planar reflection of the ripple normals needs a higher-resolution slope buffer.

## Pass 32: four cloud layers (cumulus, stratocumulus, altostratus, cirrus)

- shaders/clouds.frag composites four layers, lowest first, in the same (radiance, transmittance) form as before, so the sky composite and god rays need no change.
  - Stratocumulus: 650 to 1000 m, flat profile, 700 m cells, coverage threshold 0.52, extinction 0.012/m.
  - Cumulus: unchanged (1200 to 1900 m, cloudDensity).
  - Altostratus: 3.8 to 4.4 km, smooth veil, threshold 0.46, extinction 0.006/m.
  - Cirrus: 8.2 to 8.7 km, streaks stretched along the wind (6000 m along x, 400 m along z), threshold 0.55, extinction 0.004/m, marched out to 120 km.
- Each layer was checked alone: the cirrus shows as fine streaks above the cumulus, and the altostratus as a soft veil.
- Cost: about 124 march steps per pixel (was 64). On llvmpipe the frame went from about 1.5 s to about 2.3 s. Device cost is not measured.
- Still open: the cumulus, stratus and altostratus cast no shadows on the sea or the buildings. Those still use the cumulus-only shadow field in common.glsl. The stratocumulus and altostratus grain is the same per-pixel jitter as before.

## Pass 33: buildings, quay, causeway and traffic

- Material kinds added: Asphalt 8, RoadPaint 9, Pavers 10, CarPaint 11, CarGlass 12, Stone 13 (src/scene/objects.h enum; shaders/mesh.frag branches 8 to 13).
- Facades (kind 3): a tower's palette colour hashes to a seed that picks its facade: punched windows (seed < 0.4), ribbon glazing (< 0.75) or a dark glass curtain wall with mullions and spandrels (the rest). Pale stone reveals surround punched and ribbon glazing. Ground floor has a dark shopfront band under a fascia.
- Facade relief: the glazing height field (glass sunk 6 cm, reveals proud 4 cm) perturbs the normal with the screen-space surface gradient (Mikkelsen 2010, perturbNormal). Its derivatives are taken at top level in main(), not inside the material branch.
- Towers: three stone pilasters on the street face, a stone cornice at the roofline, and galvanised rooftop plant on about half the towers.
- Quay: a rock seawall in two parts with a 18 m gap for the causeway, and a paved quay between z -418 and -395.
- Causeway: x 17 to 31, z -395 to -150, four 3.5 m lanes with edge, centre and dashed lines, and a sidewalk each side. 24 cars (six per lane) built from boxes by appendCar.
- Why a causeway and not an avenue along the skyline: the camera is 2.2 m high. A road running across the view at 280 m is under a pixel tall, so the avenue and its cars were hidden by the seawall. The causeway runs into the view, where it reads.
- Validation: no errors. The two OutputNotConsumed performance warnings predate this pass. llvmpipe frame about 2.5 s at 1920x1080. No device timing.
- Still open: at 430 m a 3.4 m bay is about 8 px at 1080p, so facade detail reads as texture only up close. The causeway is a dark band near the horizon: the camera is low, and the skyline shades it because the sun sits behind the towers. Cars have no lights and no detail beyond boxes. Cloud shadows are still cumulus only.

## Pass 34: capillary cascade and stratus shadows

- Capillary cascade: a second FFT patch of 96 m with N = 256 (cells 0.375 m; waves to about 0.5 m on the diagonals). It uses the same Phillips spectrum, keeping only |k| above the main grid's corner wavenumber sqrt(2) pi N / L (1.11 rad/m), so no wave is counted twice. Gain kOceanCapGain = 4 gives slope RMS about 0.02; the main grid's is about 0.05. Gain 8 (slope about 0.04) made the water body marble, so it was halved.
- Shader: ocean.frag samples the cascade height as storage slot 3 (binding 26) and perturbs the normal near the camera. Detail fades by pixel footprint: full below 0.06 m per pixel, none above 0.18 m per pixel (about 100 to 200 m at 1080p).
- RHI: storage slots are now visible to the fragment stage (vk_device.cpp stage flags; kMaxStorageBuffers 3 -> 4). The buffers stay read-only, so no device feature is needed.
- Cloud shadows: cloudShadowTransmittance (common.glsl) adds the stratocumulus (650 to 1000 m) and altostratus (3.8 to 4.4 km) decks. Each gets one sample at mid-layer, with the same noise and profile as clouds.frag's layerDensity, so the shadows fall where the decks are.
- Measured: the sea is 0.6 % darker on average when the layer shadows are on. The changes show on the left-hand towers and the sea there (amplified difference). The sun is 12 degrees up behind the skyline, so the shadows land far out to sea and are sparse in this framing.
- Toggles: ENGINE_SEA_CAP=0 and ENGINE_CLOUD_LAYERS=0 reproduce the Pass 33 sea and shadows exactly. Both are on by default.
- Cost: one more evolve and 16 FFT passes per frame on 256 x 256, plus two extra layer density evaluations per shadow lookup. On llvmpipe the frame was 2.6 s (2.55 s with both off). No device timing.
- Validation: no errors. The two OutputNotConsumed performance warnings predate this pass.
- Still open: each layer's shadow is a single sample, so there is no penumbra. The layers do not yet cast shadows on the sky-facing sides of the towers.

## Pass 35: GPU timing per render pass

- RHI: CommandList::resetTimestamps and writeTimestamp, and Device::readTimestamps, timestampPeriodNs and timestampsSupported. Backed by a Vulkan timestamp query pool of kMaxTimestamps (256) queries, created only when the device reports timer support for graphics and compute.
- Render graph: with profiling on, Graph::execute brackets each pass with two timestamps. Skipped runOnce passes record zero time. Graph::passNames gives the names in order.
- Renderer: ENGINE_PROFILE=1 reads the ticks after the fence and keeps per-pass milliseconds (lastPassTimesMs). ENGINE_PROFILE unset leaves the frame path unchanged.
- Report: main_render prints each pass's median GPU time over the timed frames, slowest first, and the sum.
- Check on llvmpipe, one still frame: the per-pass sum is 2592 ms against a 2596 ms frame time. Validation: no errors.
- llvmpipe's ratios are not the phone's. Its clouds and ocean surface dominate; the Adreno 830 needs its own run before any budget is set.
