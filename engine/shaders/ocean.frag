// Open-sea shading. Reflection of the sky (Fresnel), transmitted light from the water
// column (closed-form single scattering, not a volume path trace), GGX sun glint, Jacobian
// foam, and aerial perspective between the camera and the surface.
layout(location = 0) in vec3 vWorld;
layout(location = 1) in vec3 vNormal;
layout(location = 2) in float vJacDet;
layout(location = 0) out vec4 outHdr;

const vec3 FOAM_ALB = vec3(0.75, 0.77, 0.78);
const float GLINT_ROUGHNESS = 0.18;  // GGX alpha of the wave facets
// Reflectance scale: 1 is pure Fresnel, which is what calm water does at grazing angles (the sky and the
// skyline show clearly, as on a photographed canal). The earlier 0.3 hid the reflections under the body colour.
const float REFLECT_SCALE = 1.0;
const float REFLECT_DISTORT = 0.12;  // how far the wave slope shifts the planar image, in uv units per unit slope
const vec3 SEA_BODY = vec3(0.045, 0.040, 0.15);  // violet body colour under the sunset sky  // blue body colour of the water column (linear), lit by sun and sky

float fresnelSchlick(float cosT) {
    return 0.02 + 0.98 * pow(1.0 - clamp(cosT, 0.0, 1.0), 5.0);
}

// Unit direction toward the sun inside the water (Snell's law from air, flat surface).
vec3 sunDirInWater(vec3 S) {
    float ct = sqrt(max(1.0 - (1.0 - S.y * S.y) / (SEA_IOR * SEA_IOR), 1e-4));
    float st = sqrt(max(1.0 - ct * ct, 0.0));
    vec2 hz = normalize(S.xz);
    return vec3(hz.x * st, ct, hz.y * st);
}

// Radiance (in water) along the downward direction Wd, from single scattering of the sun
// in a deep water column. Per channel: sigma_s * p * E / (sigma_t * (1 + cos_v / cos_s)),
// the integral over depth with both the view and sun paths attenuated.
vec3 deepWaterRadiance(vec3 Wd, vec3 sunW, vec3 Eperp) {
    float cosV = max(-Wd.y, 1e-3);
    float cosS = max(sunW.y, 1e-3);
    float p = hgPhase(dot(Wd, sunW), SEA_G);
    return Eperp * SEA_SIGMA_S * p / (SEA_SIGMA_T * (1.0 + cosV / cosS));
}

// Capillary cascade: a 96 m FFT patch of waves from 0.75 m up to the main grid's band (see ocean_spectrum.h). Its
// height is a periodic bilinear sample, as in ocean.vert, and its slopes are central differences of that height.
layout(std430, set = 0, binding = 28) readonly buffer OceanCapillary { vec2 oceanCap[]; };
const int CAP_N = 256;
const float CAP_L = 96.0;  // patch period (m); equals kOceanCapL

float capHeight(vec2 p) {
    vec2 g = p * (float(CAP_N) / CAP_L);
    vec2 fl = floor(g);
    vec2 f = g - fl;
    ivec2 i0 = ivec2(fl) & ivec2(CAP_N - 1);
    ivec2 i1 = (i0 + ivec2(1)) & ivec2(CAP_N - 1);
    float a = oceanCap[i0.y * CAP_N + i0.x].x;
    float b = oceanCap[i0.y * CAP_N + i1.x].x;
    float c = oceanCap[i1.y * CAP_N + i0.x].x;
    float d = oceanCap[i1.y * CAP_N + i1.x].x;
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

// Normal from the capillary slopes. Off when uTaa.w is 0 (ENGINE_SEA_CAP=0). Faded out where a pixel is wider than
// the short waves, so distant water shows no shimmer.
vec3 capNormal(vec3 n, vec3 p, float dist) {
    if (uF.uTaa.w < 0.5) return n;
    float footprint = dist * uF.uExposure.y;  // metres per pixel
    float fade = 1.0 - smoothstep(0.06, 0.18, footprint);
    if (fade <= 0.0) return n;
    const float e = 0.5 * CAP_L / float(CAP_N);  // half a cell
    float sx = (capHeight(p.xz + vec2(e, 0.0)) - capHeight(p.xz - vec2(e, 0.0))) / (2.0 * e);
    float sz = (capHeight(p.xz + vec2(0.0, e)) - capHeight(p.xz - vec2(0.0, e))) / (2.0 * e);
    n.xz -= fade * vec2(sx, sz);
    return normalize(n);
}

// Capillary slope near the camera: small-scale noise the mesh cannot resolve.
vec3 capillaryNormal(vec3 n, vec3 p, float dist) {
    // Two band-limited, wind-advected normal detail layers approximate unresolved
    // capillary waves. Their footprint fade prevents distant shimmer and moire.
    float footprint = dist * uF.uExposure.y;
    float fade = 0.3 * (1.0 - smoothstep(0.035, 0.16, footprint));  // calm sea: weaker capillary detail
    if (fade <= 0.0) return n;

    vec2 wind = vec2(0.73, 0.68);
    float t = uF.uTime.x;
    vec2 q0 = p.xz * 5.5 + wind * (t * 0.22);
    vec2 q1 = p.xz * 13.0 - wind.yx * (t * 0.37);
    const float e = 0.012;
    vec2 g = vec2(0.0);
    for (int layer = 0; layer < 2; ++layer) {
        vec2 q = (layer == 0) ? q0 : q1;
        float amp = (layer == 0) ? 1.0 : 0.38;
        float dx = valueNoise(q + vec2(e, 0.0)) - valueNoise(q - vec2(e, 0.0));
        float dz = valueNoise(q + vec2(0.0, e)) - valueNoise(q - vec2(0.0, e));
        g += amp * vec2(dx, dz) / (2.0 * e);
    }
    n.xz -= fade * SEA_CAP * 0.55 * g;
    return normalize(n);
}

// Seabed height (m, sea level at 0). A port of landHeight in src/scene/landscape.cpp: the shore runs
// along shoreZ, land rises inland, the shelf shelves seaward. Analytic (no noise), so the shading and
// the mesh agree exactly.
const vec3 SAND_ALB = vec3(0.66, 0.57, 0.43);

float smooth01b(float t) { t = clamp(t, 0.0, 1.0); return t * t * (3.0 - 2.0 * t); }
float shoreZ(float x) { return -34.0 + 12.0 * sin(0.011 * x + 0.4) + 5.0 * sin(0.029 * x - 1.1); }
float seabedHeight(vec2 xz) {
    float d = xz.y - shoreZ(xz.x);  // > 0 on land, < 0 at sea
    if (d >= 0.0) {
        float s = smooth01b(d / 28.0);
        float w = smooth01b(d / 12.0);
        return 3.5 * s + w * (0.5 * sin(0.09 * xz.x + 0.04 * xz.y) + 0.4 * sin(0.053 * xz.y - 0.031 * xz.x + 1.7));
    }
    float u = -d;
    float depth = 0.085 * min(u, 45.0) + 0.3 * max(u - 45.0, 0.0);
    depth = min(depth, 40.0);
    depth += 0.35 * sin(0.21 * xz.x + 0.37 * xz.y) * smooth01b(u / 10.0);
    return -max(depth, 0.0);
}

void main() {
    vec3 p = vWorld;
    vec3 cam = uF.uCamPos.xyz;
    vec3 V = normalize(cam - p);
    float dist = distance(p, cam);
    vec3 n = capillaryNormal(normalize(vNormal), p, dist);
    n = capNormal(n, p, dist);
    vec3 S = uF.uSunDir.xyz;
    vec3 E = uF.uSunIlluminance.xyz;  // sun-facing illuminance, pre-exposed
    E *= sunVisibility(p, vec3(0.0, 1.0, 0.0));
    E *= cloudShadowAt(p);  // precomputed screen-space cloud transmittance
    float NoV = max(dot(n, V), 1e-4);

    // Reflected light (no sun disk: the glint term covers it): the sky blurred over the wave
    // slopes, and the planar render of the objects wherever one covers the reflected ray.
    float F = fresnelSchlick(NoV);
    vec3 R = reflect(-V, n);
    vec3 Lsky = skyGlossy(vec3(R.x, max(R.y, 0.0005), R.z), 0.12);
    vec4 rc = uF.uReflViewProj * vec4(p, 1.0);
    vec2 ruv = rc.xy / rc.w * 0.5 + 0.5 + n.xz * REFLECT_DISTORT;  // the slope breaks the reflected image up with the ripples
    vec4 planar = texture(uReflTex, ruv);
    vec3 Lrefl = mix(Lsky, planar.rgb, planar.a);

    // Transmitted light: refracted view ray into the water, radiance factor 1/eta^2 on exit
    // (radiance divided by n^2 is conserved across the surface).
    vec3 Wd = refract(-V, n, 1.0 / SEA_IOR);
    vec3 Ltrans = vec3(0.0);
    if (Wd.y < 0.0) {
        vec3 sunW = sunDirInWater(S);
        float cosS = sunW.y;
        float Fsun = fresnelSchlick(S.y);
        vec3 Eperp = E * (1.0 - Fsun) * (S.y / cosS);   // sun-beam irradiance in the water
        Ltrans = (1.0 - F) * deepWaterRadiance(Wd, sunW, Eperp) / (SEA_IOR * SEA_IOR);
    }

    // GGX glint from the sun, evaluated at the sun direction.
    vec3 glint = vec3(0.0);
    float NoL = dot(n, S);
    if (NoL > 0.0) {
        vec3 H = normalize(V + S);
        float NoH = max(dot(n, H), 0.0);
        float VoH = max(dot(V, H), 1e-4);
        float a2 = GLINT_ROUGHNESS * GLINT_ROUGHNESS;
        float dd = NoH * NoH * (a2 - 1.0) + 1.0;
        float D = a2 / (PI * dd * dd);
        float Vis = 0.5 / (NoL * sqrt(NoV * NoV * (1.0 - a2) + a2) + NoV * sqrt(NoL * NoL * (1.0 - a2) + a2));
        glint = D * Vis * fresnelSchlick(VoH) * NoL * E;
    }

    // Sunlight through thin wave crests: looking toward the sun, a crest whose far face turns away
    // from the sun glows turquoise. The effect is strongest on the crest tops and fades in the troughs.
    float crest = smoothstep(0.02, 0.20, vWorld.y);
    float through = pow(clamp(dot(-V, S), 0.0, 1.0), 4.0) * clamp(0.25 - dot(n, S), 0.0, 1.0) * crest;
    vec3 crestGlow = vec3(0.05, 0.40, 0.36) * E * through * (1.0 - F) / PI * 0.35;

    // Body colour: light scattered back from the water column. Water absorbs red first, so this is the
    // saturated blue of open sea, not a reflection of the sky.
    // Water column over the seabed. Red light is absorbed within metres, so shallow water
    // transmits the sand's green and blue: the turquoise of a beach.
    float depth = max(-seabedHeight(p.xz), 0.0);
    vec3 transmit = exp(-SEA_SIGMA_A * depth * 1.25);
    vec3 body = SEA_BODY * (E * max(NoL, 0.0) / PI + skyAmbient()) * (1.0 - F);
    vec3 bottom = SAND_ALB * (E * max(S.y, 0.0) / PI + 0.5 * skyAmbient()) * (1.0 - F);
    vec3 waterBody = mix(body, bottom, transmit);
    vec3 base = REFLECT_SCALE * F * Lrefl + Ltrans + waterBody + glint + crestGlow;

    // Foam where the surface compresses (Jacobian determinant falls below ~0.5), broken up by noise.
    float foam = 0.48 * (1.0 - smoothstep(0.16, 0.42, vJacDet)) * (0.6 + 0.4 * valueNoise(p.xz * 0.6));
    vec3 foamRad = FOAM_ALB * (E * max(NoL, 0.0) / PI + skyAmbient());
    // Surf: white breaking water over the shallows, broken up and moving with time.
    float shore = 1.0 - smoothstep(0.25, 1.4, depth);
    float surfNoise = valueNoise(p.xz * 1.3 + vec2(uF.uTime.x * 0.35, -uF.uTime.x * 0.2));
    float surf = shore * smoothstep(0.35, 0.8, surfNoise);
    foam = max(foam, 0.85 * surf);
    vec3 col = mix(base, foamRad, foam);

    // Aerial perspective from the camera to the surface.
    vec3 Tv;
    vec3 Lin = aerialLookup(p, dist, Tv);
    // God rays: the in-scatter that clouds shade, removed along the view ray to the sea surface.
    Lin = max(Lin + texture(uShaftTex, screenUVOf(p)).rgb, vec3(0.0));
    outHdr = vec4(col * Tv + Lin, 1.0);
}
