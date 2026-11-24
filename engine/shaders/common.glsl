// Shared by every engine shader (prepended by the build; no #version here).
// Units: metres and seconds. Radiance is pre-exposed: a value stored in an HDR target is
// cd/m^2 multiplied by uExposure.x, so the largest sun radiance stays inside RGBA16F range.

layout(std140, set = 0, binding = 0) uniform FrameData {
    mat4 uViewProj;        // clip from world
    mat4 uInvViewProj;     // world from clip
    vec4 uCamPos;          // xyz camera position (m)
    vec4 uSunDir;          // xyz unit vector toward the sun; w = cos(sun angular radius)
    vec4 uSunIlluminance;  // xyz sun illuminance on a sun-facing surface (lux), pre-exposed
    vec4 uSunRadiance;     // xyz sun disk radiance (cd/m^2), pre-exposed
    vec4 uExposure;        // x = pre-exposure factor 1 / (1.2 * 2^EV100)
    vec4 uTime;            // x = time (s)
    mat4 uSunViewProj;     // clip from world for the sun shadow map
    mat4 uReflViewProj;    // clip from world for the planar reflection (camera mirrored in the sea)
    vec4 uProbeOrigin;     // xyz world position of probe (0,0,0)
    vec4 uProbeStep;       // xyz probe spacing (m)
    vec4 uProbeDims;       // xyz probe counts; w = 1 in the reflection pass (clip below the sea)
    vec4 uProbes[384];     // baked ambient cubes: 6 faces per probe, rgb average radiance
    mat4 uMainViewProj;    // the main camera's clip matrix (the froxels are keyed to it, also in the mirrored pass)
    mat4 uSunNearViewProj; // clip from world for the near sun cascade (+-40 m)
    mat4 uSunFarViewProj;  // clip from world for the far sun cascade (+-500 m)
    mat4 uPrevViewProj;    // last frame's unjittered clip from world (TAA reprojection)
    vec4 uTaa;             // z = 1 when the TAA history holds a previous frame
    vec4 uDof;             // adaptive depth-of-field: x = focal length (mm), y = f-number,
                           // z = focus distance (m), w = sensor height (mm)
    vec4 uWeather;         // weather/wetness: x = rain intensity (0..1), y = wetness factor (0..1),
                           // z = flood level (metres above shore), w = foam intensity (0..1)
} uF;

// Textures in the fixed descriptor layout (see rhi.h). Each pass reads only what it declares.
layout(set = 0, binding = 1) uniform sampler2D uTransmittanceLut;  // T(height, cos zenith)
layout(set = 0, binding = 2) uniform sampler2D uSkyViewLut;        // sky radiance by direction
layout(set = 0, binding = 3) uniform sampler2D uCloudTex;          // cloud radiance, transmittance
layout(set = 0, binding = 4) uniform sampler2D uHdrTex;            // HDR input to tone mapping / blur
layout(set = 0, binding = 5) uniform sampler2D uMultiScatterLut;   // mean multiple-scattering radiance
layout(set = 0, binding = 6) uniform sampler2D uBloomTex;          // blurred bright areas
layout(set = 0, binding = 7) uniform sampler2D uShadowMap;         // sun shadow map: 1 - light-clip depth
layout(set = 0, binding = 9) uniform sampler2D uCloudShadowTex;    // screen-projected cloud transmittance
layout(set = 0, binding = 8) uniform sampler2D uWoodAlbedo;        // scanned wood diffuse (sRGB, mipmapped)
layout(set = 0, binding = 10) uniform sampler2D uWoodNormal;       // tangent-space normal derived from it
layout(set = 0, binding = 11) uniform sampler2D uRockAlbedo;       // scanned rock diffuse (sRGB, mipmapped)
layout(set = 0, binding = 12) uniform sampler2D uRockNormal;       // tangent-space normal derived from it
layout(set = 0, binding = 13) uniform sampler2D uShaftTex;         // god rays: negative in-scatter removed by clouds
layout(set = 0, binding = 14) uniform sampler2D uReflTex;          // planar reflection of the objects (rgb, coverage a)
layout(set = 0, binding = 15) uniform sampler2D uSkyPrefilter;      // prefiltered sky rows (see sky_prefilter.frag)
layout(set = 0, binding = 16) uniform sampler2D uAerialL;          // aerial in-scatter froxels (see aerial.frag)
layout(set = 0, binding = 17) uniform sampler2D uAerialT;          // aerial transmittance froxels
layout(set = 0, binding = 18) uniform sampler2D uShadowNear;      // near sun cascade (see sunVisibility)
layout(set = 0, binding = 19) uniform sampler2D uShadowFar;       // far sun cascade
layout(set = 0, binding = 20) uniform sampler2D uLinDepth;        // linear depth from the prepass (metres)
layout(set = 0, binding = 21) uniform sampler2D uAO;              // ambient occlusion (see gtao.frag, ao_blur.frag)
layout(set = 0, binding = 22) uniform sampler2D uHistory;        // TAA history: last frame's resolved HDR

// Per-pass constant: the size (w, h) of the target being rendered, in pixels.
layout(push_constant) uniform PassPush {
    vec4 uTargetSize;
} uPush;

const float PI = 3.14159265358979;

// ---------- Atmosphere (single scattering, Preetham turbidity 3) ----------
const float RG = 6360000.0;                            // planet radius (m)
const float RT = 6460000.0;                            // top of atmosphere (m)
const vec3  BETA_R = vec3(5.8e-6, 13.5e-6, 33.1e-6);   // Rayleigh scattering, sea level (1/m)
const float H_R = 8000.0;                              // Rayleigh scale height (m)
const float H_M = 1200.0;                              // Mie scale height (m)
const float MIE_G = 0.76;                              // Henyey-Greenstein asymmetry
const float HAZE_SCALE = 1.5;                         // haze knob (1 = Preetham turbidity 3), shared with atmo_reference.h
const vec3  BETA_M = HAZE_SCALE * (0.04608 * 3.0 - 0.04586) * vec3(pow(0.68, -1.3), pow(0.55, -1.3), pow(0.44, -1.3)) / H_M;
const vec3  SOLAR_LUX = vec3(128000.0);                // top-of-atmosphere sun illuminance (lux)

// LUT sizes. The sky-view table has one duplicate azimuth column so that the wrap at 2*pi
// needs no special case.
const float TRANS_W = 256.0;
const float TRANS_H = 64.0;
const float SKY_W = 193.0;
const float SKY_H = 108.0;
const int   TRANS_STEPS = 40;   // transmittance integration (table generation only)
const int   VIEW_STEPS = 16;    // sky-view integration per texel

// Maps a normalised parameter x01 in [0,1] (texel i holds x01 = i / (n - 1)) to the
// sampling coordinate of an n-texel texture with clamp addressing.
float lutCoord(float x01, float n) { return x01 * (n - 1.0) / n + 0.5 / n; }

// Distance along a ray from radius r0 (cos zenith mu) to the sphere of radius R; -1 if none.
float sphereRoot(float r0, float mu, float R, bool farRoot) {
    float b = r0 * mu;
    float disc = b * b - (r0 * r0 - R * R);
    if (disc < 0.0) return -1.0;
    float s = sqrt(disc);
    return farRoot ? (-b + s) : (-b - s);
}

// True if a ray from radius r0 with cos zenith mu enters the planet.
bool hitsGround(float r0, float mu) {
    if (mu >= 0.0) return false;
    float b = r0 * mu;
    return b * b - (r0 * r0 - RG * RG) >= 0.0;
}

// Transmittance table parametrisation. Both axes are non-linear so that texels cluster
// where transmittance changes fastest: mu = t|t| (dense near the horizon, where the air
// path is long) and h = H * v^2 (dense near the ground). t and v are the texel axes in [-1, 1]
// and [0, 1].
float muToU(float mu) { return 0.5 + 0.5 * sign(mu) * sqrt(abs(mu)); }
float rToV(float r) { return sqrt(max((r - RG) / (RT - RG), 0.0)); }

// Sun transmittance (RGB) from radius r with local cos zenith mu, from the LUT. Zero if
// the planet blocks the sun.
vec3 sunTransmittance(float r, float mu) {
    if (hitsGround(r, mu)) return vec3(0.0);
    vec2 uv = vec2(lutCoord(muToU(mu), TRANS_W), lutCoord(rToV(r), TRANS_H));
    return texture(uTransmittanceLut, uv).rgb;
}

// Multiple-scattering LUT: mean over all directions of the once-scattered sun radiance at a
// point, with the higher orders summed. Linear in mu, sampled by height as for transmittance.
const float MS_W = 32.0;
const float MS_H = 32.0;
vec3 msLookup(float r, float mu) {
    return texture(uMultiScatterLut, vec2(lutCoord(mu * 0.5 + 0.5, MS_W), lutCoord(rToV(r), MS_H))).rgb;
}

// Sky-view table coordinates for a unit direction. Azimuth is linear; elevation uses
// e = s|s| * pi/2 with s in [-1, 1], which packs texels near the horizon.
vec2 skyViewUV(vec3 d) {
    float phi = atan(d.z, d.x);
    float x01 = fract(phi / (2.0 * PI));
    float e = asin(clamp(d.y, -1.0, 1.0));
    float s = sign(e) * sqrt(abs(e) / (0.5 * PI));
    float y01 = 0.5 * s + 0.5;
    return vec2(lutCoord(x01, SKY_W), lutCoord(y01, SKY_H));
}

// Sky radiance (pre-exposed) for a unit direction seen from the camera.
vec3 skyViewLookup(vec3 d) { return texture(uSkyViewLut, skyViewUV(d)).rgb; }

// Sky radiance including the existing volumetric cloud slab. Projecting a reflected world-space
// direction through the camera matrix addresses the same directional cloud field used by the sky
// pass. Directions outside the camera frustum fall back to the cloud-free atmosphere LUT.
vec3 skyWithClouds(vec3 d) {
    d = normalize(d);
    vec3 L = skyViewLookup(d);
    if (d.y <= 0.0) return L;
    vec4 clipPos = uF.uViewProj * vec4(uF.uCamPos.xyz + d * 1000.0, 1.0);
    if (clipPos.w <= 1e-4) return L;
    vec2 uv = clipPos.xy / clipPos.w * 0.5 + 0.5;
    if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0)))) return L;
    vec4 cloud = texture(uCloudTex, uv);
    return L * cloud.a + cloud.rgb;
}

// Prefiltered sky (sky_prefilter.frag): three rows of the sky-view LUT blurred over cones of
// radius 0.03, 0.19 and 0.35 rad. A lookup blends the two rows nearest to the requested radius.
const float PRE_ROUGH0 = 0.03;
const float PRE_ROUGH_STEP = 0.16;
vec3 skyPrefiltered(vec3 d, float rough) {
    vec2 uv = skyViewUV(d);
    float lf = clamp((rough - PRE_ROUGH0) / PRE_ROUGH_STEP, 0.0, 2.0);
    float lo = floor(min(lf, 1.999));
    float t = lf - lo;
    vec3 a = texture(uSkyPrefilter, vec2(uv.x, (uv.y + lo) / 3.0)).rgb;
    vec3 b = texture(uSkyPrefilter, vec2(uv.x, (uv.y + lo + 1.0) / 3.0)).rgb;
    return mix(a, b, t);
}

// Glossy sky reflection: the prefiltered atmosphere, plus the cloud layer averaged over a small cone
// around d. One cloud sample per pixel aliases as the reflected direction changes across waves, so
// the cloud is averaged over the same cone radius as the atmosphere (four taps).
vec3 skyGlossy(vec3 d, float rough) {
    d = normalize(d);
    vec3 L = skyPrefiltered(vec3(d.x, max(d.y, 0.0005), d.z), rough);
    if (d.y <= 0.0) return L;
    vec3 helper = abs(d.y) < 0.92 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 t = normalize(cross(helper, d));
    vec3 b = cross(d, t);
    float cone = max(rough, 0.02);
    vec4 cloudSum = vec4(0.0);
    int taps = 0;
    for (int i = 0; i < 4; i++) {
        float r = cone * sqrt((float(i) + 0.5) / 4.0);
        float a = float(i) * 2.39996323;
        vec3 sd = normalize(d + r * (cos(a) * t + sin(a) * b));
        vec4 clipPos = uF.uViewProj * vec4(uF.uCamPos.xyz + sd * 1000.0, 1.0);
        if (clipPos.w <= 1e-4) continue;
        vec2 cuv = clipPos.xy / clipPos.w * 0.5 + 0.5;
        if (any(lessThan(cuv, vec2(0.0))) || any(greaterThan(cuv, vec2(1.0)))) continue;
        cloudSum += texture(uCloudTex, cuv);
        taps++;
    }
    if (taps == 0) return L;
    vec4 cloud = cloudSum / float(taps);
    return L * cloud.a + cloud.rgb;
}

// Simple upward-only sky average retained for cloud scattering and open-water ambient terms.
vec3 skyAmbient() {
    return 0.25 * (skyViewLookup(vec3(0.0, 1.0, 0.0)) + skyViewLookup(normalize(vec3(0.7, 1.0, 0.0)))
                 + skyViewLookup(normalize(vec3(-0.35, 1.0, 0.6))) + skyViewLookup(normalize(vec3(-0.35, 1.0, -0.6))));
}

// Cosine-hemisphere approximation oriented to the actual surface normal. This avoids
// lighting upward-facing decks, vertical masonry and downward-facing undersides identically.
vec3 skyDiffuseIrradiance(vec3 normal) {
    vec3 n = normalize(normal);
    vec3 helper = abs(n.y) < 0.92 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 t = normalize(cross(helper, n));
    vec3 b = normalize(cross(n, t));
    vec3 d0 = normalize(n + 0.62 * t + 0.18 * b);
    vec3 d1 = normalize(n - 0.62 * t + 0.18 * b);
    vec3 d2 = normalize(n + 0.62 * b - 0.18 * t);
    vec3 d3 = normalize(n - 0.62 * b - 0.18 * t);
    vec3 d4 = normalize(n + 0.25 * t + 0.55 * b);
    vec3 d5 = normalize(n - 0.25 * t - 0.55 * b);
    return (skyViewLookup(d0) + skyViewLookup(d1) + skyViewLookup(d2) +
            skyViewLookup(d3) + skyViewLookup(d4) + skyViewLookup(d5)) / 6.0;
}

// ---------- Cloud self-shadowing and cast shadows ----------
// The view-ray cloud pass and direct-lighting passes share the same world-space coverage
// convention. This compact density evaluation samples the cloud layer along the sun ray so
// clouds attenuate sunlight on the sea and on scene geometry, not only in the camera background.
float cloudShadowHash3(vec3 p) {
    p = fract(p * 0.1031);
    p += dot(p, p.zyx + 31.32);
    return fract((p.x + p.y) * p.z);
}
float cloudShadowNoise3(vec3 x) {
    vec3 i = floor(x), f = fract(x);
    vec3 u = f * f * (3.0 - 2.0 * f);
    float c000 = cloudShadowHash3(i);
    float c100 = cloudShadowHash3(i + vec3(1.0, 0.0, 0.0));
    float c010 = cloudShadowHash3(i + vec3(0.0, 1.0, 0.0));
    float c110 = cloudShadowHash3(i + vec3(1.0, 1.0, 0.0));
    float c001 = cloudShadowHash3(i + vec3(0.0, 0.0, 1.0));
    float c101 = cloudShadowHash3(i + vec3(1.0, 0.0, 1.0));
    float c011 = cloudShadowHash3(i + vec3(0.0, 1.0, 1.0));
    float c111 = cloudShadowHash3(i + vec3(1.0, 1.0, 1.0));
    float x00 = mix(c000, c100, u.x), x10 = mix(c010, c110, u.x);
    float x01 = mix(c001, c101, u.x), x11 = mix(c011, c111, u.x);
    return mix(mix(x00, x10, u.y), mix(x01, x11, u.y), u.z);
}
float cloudShadowDensity(vec3 p) {
    p -= vec3(uF.uTime.x * 6.0, 0.0, uF.uTime.x * 2.1);
    float h = (p.y - 1200.0) / 700.0;
    if (h <= 0.0 || h >= 1.0) return 0.0;
    float profile = smoothstep(0.0, 0.25, h) * (1.0 - smoothstep(0.6, 1.0, h));
    vec3 q = p / vec3(450.0, 300.0, 450.0);
    float wx = cloudShadowNoise3(q * 0.62 + vec3(2.1, 4.7, 1.3)) - 0.5;
    float wz = cloudShadowNoise3(q * 0.62 + vec3(7.4, 1.8, 5.2)) - 0.5;
    q.xz += vec2(wx, wz) * 0.34;
    float n = 0.0, a = 0.5;
    for (int i = 0; i < 4; ++i) {
        n += a * cloudShadowNoise3(q);
        q = q * 2.03 + vec3(17.1, 9.3, 5.7);
        a *= 0.5;
    }
    float detail = cloudShadowNoise3(p / vec3(54.0, 42.0, 54.0));
    float coverPatch = cloudShadowNoise3(p / vec3(2600.0, 1.0, 2600.0));
    float coverage = smoothstep(0.49, 0.69, n - (0.50 - detail) * 0.075 - (coverPatch - 0.5) * 0.24);
    return profile * coverage;
}
// Stratocumulus and altostratus density for the sun shadows: the same fbm and profile as clouds.frag's layerDensity
// (the noise and hash match cloudShadowNoise3), so a shadow falls where the visible deck is.
float shadowLayerDensity(vec3 p, vec2 wind, float h0, float h1, vec3 scale, float threshold, int octaves) {
    p.xz -= uF.uTime.x * wind;
    float h = (p.y - h0) / (h1 - h0);
    if (h <= 0.0 || h >= 1.0) return 0.0;
    float prof = smoothstep(0.0, 0.2, h) * (1.0 - smoothstep(0.7, 1.0, h));
    vec3 q = p / scale;
    float n = 0.0, a = 0.5;
    for (int i = 0; i < octaves; i++) {
        n += a * cloudShadowNoise3(q);
        q = q * 2.03 + vec3(17.1, 9.3, 5.7);
        a *= 0.5;
    }
    n /= 1.0 - pow(0.5, float(octaves));
    return prof * smoothstep(threshold, threshold + 0.18, n);
}

float cloudShadowTransmittance(vec3 worldPos) {
    float sunY = uF.uSunDir.y;
    if (sunY <= 0.015) return 1.0;
    // Evaluate the centre of the cumulus slab where the sun ray from this surface passes through.
    float t = max((1550.0 - worldPos.y) / sunY, 0.0);
    vec3 cloudPoint = worldPos + uF.uSunDir.xyz * t;
    float density = cloudShadowDensity(cloudPoint);
    float opticalDepth = 0.03 * density * (700.0 / max(sunY, 0.04));
    // Stratocumulus (650 to 1000 m) and altostratus (3.8 to 4.4 km), one sample each at mid-layer. Optical depth is
    // extinction times the slant path through the layer. Off when uExposure.z is 0 (ENGINE_CLOUD_LAYERS=0).
    if (uF.uExposure.z > 0.5) {
        float tS = max((825.0 - worldPos.y) / sunY, 0.0);
        float dS = shadowLayerDensity(worldPos + uF.uSunDir.xyz * tS, vec2(3.0, 1.0), 650.0, 1000.0,
                                      vec3(700.0, 90.0, 700.0), 0.52, 5);
        opticalDepth += 0.012 * dS * (350.0 / max(sunY, 0.04));
        float tA = max((4100.0 - worldPos.y) / sunY, 0.0);
        float dA = shadowLayerDensity(worldPos + uF.uSunDir.xyz * tA, vec2(5.0, 2.0), 3800.0, 4400.0,
                                      vec3(2500.0, 120.0, 2500.0), 0.46, 4);
        opticalDepth += 0.006 * dA * (600.0 / max(sunY, 0.04));
    }
    // Preserve a little skylight under thin cloud and soften the single-sample footprint.
    return mix(1.0, exp(-opticalDepth), 0.88);
}

float cloudShadowAt(vec3 worldPos) {
    vec4 clip = uF.uViewProj * vec4(worldPos, 1.0);
    if (clip.w <= 1e-4) return 1.0;
    vec2 uv = clip.xy / clip.w * 0.5 + 0.5;
    if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0)))) return 1.0;
    return texture(uCloudShadowTex, uv).r;
}

// ---------- Sun shadows ----------
// The shadow pass stores 1 - depth, so the cleared background (0) never casts a shadow.
const float SHADOW_NORMAL_BIAS = 0.12;    // m, offset along the surface normal against acne
const float SHADOW_DEPTH_BIAS = 0.00012; // light-clip depth units

// Stable, sine-free per-point rotation for the soft shadow kernel.
float shadowHash(vec2 p) {
    vec3 h = fract(vec3(p.xyx) * 0.1031);
    h += dot(h, h.yzx + 33.33);
    return fract((h.x + h.y) * h.z);
}

// Cascaded sun shadows: a near box (+-40 m), the mid box (+-160 m, binding 7) and a far box
// (+-500 m). Every cascade has the same depth range, so the penumbra and bias use one depth scale.
const float CASCADE_HALF_NEAR = 40.0;
const float CASCADE_HALF_MID = 160.0;
const float CASCADE_HALF_FAR = 500.0;
const float CASCADE_DEPTH = 1200.0;

// PCF visibility from one cascade. occ is the receiver's light-clip depth inverted (1 - ndc.z), so
// a stored occluder is occ <= stored depth. halfWorld is the cascade's half width in metres.
float cascadeVisibility(sampler2D sm, vec2 uv, float ndcZ, vec3 p, float halfWorld) {
    float occ = 1.0 - ndcZ;
    vec2 mapSize = vec2(textureSize(sm, 0));
    vec2 texel = 1.0 / mapSize;
    // Penumbra grows with blocker/receiver separation (the solar disc's angular extent,
    // 0.2665 degrees), in texels of this cascade.
    float blockerDepth = 1.0 - texture(sm, uv).r;
    float separationWorld = max(ndcZ - blockerDepth, 0.0) * CASCADE_DEPTH;
    float texelsPerMetre = mapSize.x / (2.0 * halfWorld);
    float sunRadiusTexels = clamp(separationWorld * 0.00465 * texelsPerMetre, 0.65, 5.0);
    const vec2 poisson[12] = vec2[12](
        vec2(-0.326, -0.406), vec2(-0.840, -0.074), vec2(-0.696,  0.457),
        vec2(-0.203,  0.621), vec2( 0.962, -0.195), vec2( 0.473, -0.480),
        vec2( 0.519,  0.767), vec2( 0.185, -0.893), vec2( 0.507,  0.064),
        vec2( 0.896,  0.412), vec2(-0.321,  0.914), vec2(-0.982,  0.112));
    float angle = 6.2831853 * shadowHash(p.xz * 0.37);
    mat2 rot = mat2(cos(angle), -sin(angle), sin(angle), cos(angle));
    float lit = 0.0;
    for (int i = 0; i < 12; i++) {
        float s = texture(sm, uv + rot * poisson[i] * texel * sunRadiusTexels).r;
        // Contact hardening: soften the shadow boundary by biasing the occlusion test
        // toward the surface normal. A surface facing the sun receives less shadow.
        float nDotL = max(dot(normalize(p.xyz), normalize(uF.uSunDir.xyz)), 0.0);
        float contact = smoothstep(0.0, 0.5, 1.0 - nDotL);
        lit += (s <= occ + SHADOW_DEPTH_BIAS * contact) ? 1.0 : 0.0;
    }
    return lit / 12.0;
}

bool inCascade(vec3 ndc) {
    vec2 uv = ndc.xy * 0.5 + 0.5;
    return all(greaterThanEqual(uv, vec2(0.0))) && all(lessThanEqual(uv, vec2(1.0)))
        && ndc.z >= 0.0 && ndc.z <= 1.0;
}

// Sun visibility (0 in shadow, 1 lit) at world position p: the nearest cascade that contains p.
// Points outside all cascades are lit.
float sunVisibility(vec3 p, vec3 n) {
    float NoL = max(dot(normalize(n), normalize(uF.uSunDir.xyz)), 0.0);
    float normalBias = 0.015 + (1.0 - NoL) * 0.085;
    vec3 pb = p + normalize(n) * normalBias;

    vec3 ndc = (uF.uSunNearViewProj * vec4(pb, 1.0)).xyz;
    if (inCascade(ndc)) return cascadeVisibility(uShadowNear, ndc.xy * 0.5 + 0.5, ndc.z, p, CASCADE_HALF_NEAR);
    ndc = (uF.uSunViewProj * vec4(pb, 1.0)).xyz;
    if (inCascade(ndc)) return cascadeVisibility(uShadowMap, ndc.xy * 0.5 + 0.5, ndc.z, p, CASCADE_HALF_MID);
    ndc = (uF.uSunFarViewProj * vec4(pb, 1.0)).xyz;
    if (inCascade(ndc)) return cascadeVisibility(uShadowFar, ndc.xy * 0.5 + 0.5, ndc.z, p, CASCADE_HALF_FAR);
    return 1.0;
}

// Henyey-Greenstein phase function. cosT = dot(view direction, direction toward the light).
float hgPhase(float cosT, float g) {
    float g2 = g * g;
    return (1.0 - g2) / (4.0 * PI * pow(1.0 + g2 - 2.0 * g * cosT, 1.5));
}

// ---------- Sea ----------
const float SEA_IOR = 1.33;
// Pure-water absorption (1/m) at 650, 550, 440 nm (Pope and Fry 1997, NASA erratum
// Table 1.1). The red channel uses the 650 nm value.
const vec3  SEA_SIGMA_A = vec3(0.340, 0.0565, 0.00635);
// Scattering (1/m): assumed clear-sea values, not measured.
const vec3  SEA_SIGMA_S = vec3(0.0072, 0.0120, 0.0293);
const vec3  SEA_SIGMA_T = SEA_SIGMA_A + SEA_SIGMA_S;
const float SEA_G = 0.8;             // particle phase asymmetry (assumed)
// Wave LOD footprint (rad) = uF.uExposure.y, set by the host to max(pixel angle, mesh cell angle).
const float SEA_CAP = 0.0032;         // capillary slope scale

// Value noise in [0, 1] (bilinear over hashed lattice values with a smooth fade).
// Sine-free hash: sin()-based hashes lose precision on mobile GPUs and show as blocks.
float hash21(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}
float valueNoise(vec2 x) {
    vec2 i = floor(x), f = fract(x);
    vec2 u = f * f * (3.0 - 2.0 * f);
    float a = hash21(i), b = hash21(i + vec2(1.0, 0.0));
    float c = hash21(i + vec2(0.0, 1.0)), d = hash21(i + vec2(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// Camera-space view direction (unit, world space) through the pixel at uv (0..1, top-left
// origin), using the same projection as the rasteriser.
vec3 viewDirAt(vec2 uv) {
    vec2 ndc = uv * 2.0 - 1.0;
    vec4 a = uF.uInvViewProj * vec4(ndc, 0.0, 1.0);
    vec4 b = uF.uInvViewProj * vec4(ndc, 1.0, 1.0);
    return normalize(b.xyz / b.w - a.xyz / a.w);
}

// Aerial-perspective froxel grid: AER_NZ slices, log-spaced from AER_Z0 to AER_Z1 metres.
const int AER_NZ = 16;
const float AER_TW = 40.0;
const float AER_TH = 24.0;
const float AER_W = 640.0;  // AER_NZ * AER_TW
const float AER_Z0 = 1.0;
const float AER_Z1 = 60000.0;
float aerialSliceDist(float k) { return AER_Z0 * pow(AER_Z1 / AER_Z0, k / float(AER_NZ - 1)); }

// Aerial in-scatter (returned) and transmittance (Tv) from the camera to world point p at distance
// dist. Two slices are blended in depth; the screen position comes from the main camera.
vec3 aerialLookup(vec3 p, float dist, out vec3 Tv) {
    vec4 c = uF.uMainViewProj * vec4(p, 1.0);
    vec2 uv = clamp(c.xy / c.w * 0.5 + 0.5, vec2(0.0), vec2(1.0));
    vec2 tex = vec2((uv.x * (AER_TW - 1.0) + 0.5) / AER_TW, (uv.y * (AER_TH - 1.0) + 0.5) / AER_TH);
    float zf = clamp(log(max(dist, AER_Z0) / AER_Z0) / log(AER_Z1 / AER_Z0) * float(AER_NZ - 1),
                     0.0, float(AER_NZ - 1) - 1e-3);
    float k0 = floor(zf);
    float k1 = min(k0 + 1.0, float(AER_NZ - 1));
    float kf = zf - k0;
    vec3 L0 = texture(uAerialL, vec2((k0 + tex.x) / float(AER_NZ), tex.y)).rgb;
    vec3 L1 = texture(uAerialL, vec2((k1 + tex.x) / float(AER_NZ), tex.y)).rgb;
    vec3 T0 = texture(uAerialT, vec2((k0 + tex.x) / float(AER_NZ), tex.y)).rgb;
    vec3 T1 = texture(uAerialT, vec2((k1 + tex.x) / float(AER_NZ), tex.y)).rgb;
    Tv = mix(T0, T1, kf);
    return mix(L0, L1, kf);
}

// Screen UV (0..1) of a world point, for sampling screen-space results such as the god rays.
vec2 screenUVOf(vec3 p) {
    vec4 c = uF.uViewProj * vec4(p, 1.0);
    return c.xy / c.w * 0.5 + 0.5;
}

// ---------- Single-scattering integrator (shared by sky-view and sea aerial perspective) ----------
vec3 scatterAlong(float r0, vec3 dir, float tMax, out vec3 Tv) {
    float cosT = dot(dir, uF.uSunDir.xyz);
    float phR = 3.0 / (16.0 * PI) * (1.0 + cosT * cosT);
    float phM = hgPhase(cosT, MIE_G);
    vec3 L = vec3(0.0);
    Tv = vec3(1.0);
    float tPrev = 0.0;
    for (int i = 1; i <= VIEW_STEPS; i++) {
        float f = float(i) / float(VIEW_STEPS);
        float t1 = tMax * f * f;
        float ds = t1 - tPrev;
        float tm = 0.5 * (t1 + tPrev);
        float r = sqrt(max(r0 * r0 + 2.0 * r0 * dir.y * tm + tm * tm, 0.0));
        float h = r - RG;
        float dR = exp(-h / H_R);
        float dM = exp(-h / H_M);
        float muSun = (r0 * uF.uSunDir.y + tm * cosT) / r;
        vec3 Ts = sunTransmittance(r, muSun);
        vec3 sigE = BETA_R * dR + BETA_M * dM;
        vec3 sigS = BETA_R * dR * phR + BETA_M * dM * phM;
        // Single scattering (phase-weighted) plus isotropic multiple scattering.
        vec3 sigIso = BETA_R * dR + BETA_M * dM;
        vec3 src = sigS * SOLAR_LUX * Ts + sigIso * msLookup(r, muSun);
        vec3 stepT = exp(-sigE * ds);
        L += Tv * src * (vec3(1.0) - stepT) / sigE;
        Tv *= stepT;
        tPrev = t1;
    }
    return L;
}

// Sky radiance along d from radius r0. Rays into the ground see a Lambertian ground
// (albedo 0.2) lit by the sun, with no shadows and no sky light.
vec3 skyRadiance(float r0, vec3 d) {
    vec3 Tv;
    if (hitsGround(r0, d.y)) {
        float tG = sphereRoot(r0, d.y, RG, false);
        vec3 L = scatterAlong(r0, d, tG, Tv);
        vec3 Tsun = sunTransmittance(r0, uF.uSunDir.y);
        vec3 groundRad = (0.2 / PI) * SOLAR_LUX * Tsun * max(uF.uSunDir.y, 0.0);
        return L + Tv * groundRad;
    }
    float tTop = sphereRoot(r0, d.y, RT, true);
    return scatterAlong(r0, d, tTop, Tv);
}

// weather uniform added
