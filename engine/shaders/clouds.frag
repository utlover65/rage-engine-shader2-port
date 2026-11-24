// Cloud slab at half resolution: in-scattered radiance (rgb, pre-exposed) and transmittance
// (a) along each view ray. The sky composite multiplies the sky by the transmittance and
// adds the radiance. Rays that do not reach the layer write (0, 0, 0, 1).
layout(location = 0) out vec4 outC;

const float CLOUD_H0 = 1200.0;   // base (m)
const float CLOUD_H1 = 1900.0;   // top (m)
const float CLOUD_SIGMA = 0.03;  // extinction per unit density (1/m)
const float CLOUD_G = 0.6;
const float CLOUD_TMAX = 30000.0;  // beyond this the slab is at the horizon haze and is not drawn
const vec3 CLOUD_SCALE = vec3(450.0, 300.0, 450.0);  // feature size (m): width, height, depth

// Hash of a lattice point, in [0, 1). Sine-free, as in hash21.
float hash31(vec3 p) {
    p = fract(p * 0.1031);
    p += dot(p, p.zyx + 31.32);
    return fract((p.x + p.y) * p.z);
}

// Trilinear value noise in [0, 1] with a smooth fade. Three-dimensional on purpose: a grazing
// view ray that crosses a cloud sees the blob's shape, not a 2-D pattern stretched along the ray.
float valueNoise3(vec3 x) {
    vec3 i = floor(x), f = fract(x);
    vec3 u = f * f * (3.0 - 2.0 * f);
    float c000 = hash31(i);
    float c100 = hash31(i + vec3(1.0, 0.0, 0.0));
    float c010 = hash31(i + vec3(0.0, 1.0, 0.0));
    float c110 = hash31(i + vec3(1.0, 1.0, 0.0));
    float c001 = hash31(i + vec3(0.0, 0.0, 1.0));
    float c101 = hash31(i + vec3(1.0, 0.0, 1.0));
    float c011 = hash31(i + vec3(0.0, 1.0, 1.0));
    float c111 = hash31(i + vec3(1.0, 1.0, 1.0));
    float x00 = mix(c000, c100, u.x), x10 = mix(c010, c110, u.x);
    float x01 = mix(c001, c101, u.x), x11 = mix(c011, c111, u.x);
    return mix(mix(x00, x10, u.y), mix(x01, x11, u.y), u.z);
}

// Cloud density per unit extinction: 3-D fbm coverage with a vertical profile (flat base,
// rounded top, as in cumulus).
float cloudDensity(vec3 p) {
    // Coherent low-altitude wind drift; density remains a world-space 3-D field.
    p -= vec3(uF.uTime.x * 6.0, 0.0, uF.uTime.x * 2.1);
    float h = (p.y - CLOUD_H0) / (CLOUD_H1 - CLOUD_H0);
    if (h <= 0.0 || h >= 1.0) return 0.0;
    float prof = smoothstep(0.0, 0.25, h) * (1.0 - smoothstep(0.6, 1.0, h));
    vec3 q = p / CLOUD_SCALE;
    // Low-frequency domain warp breaks up the regularity of the base blobs.
    float warpX = valueNoise3(q * 0.62 + vec3(2.1, 4.7, 1.3)) - 0.5;
    float warpZ = valueNoise3(q * 0.62 + vec3(7.4, 1.8, 5.2)) - 0.5;
    q.xz += vec2(warpX, warpZ) * 0.34;
    float n = 0.0, a = 0.5;
    for (int i = 0; i < 5; i++) {
        n += a * valueNoise3(q);
        q = q * 2.03 + vec3(17.1, 9.3, 5.7);
        a *= 0.5;
    }
    // Erode the cloud edges with a smaller 3-D detail field instead of a soft blob contour.
    float detail = valueNoise3(p / vec3(54.0, 42.0, 54.0));
    // Large-scale patch field: clusters of cumulus with clear gaps, not an even sheet.
    float coverPatch = valueNoise3(p / vec3(2600.0, 1.0, 2600.0));
    float cov = smoothstep(0.49, 0.69, n - (0.50 - detail) * 0.075 - (coverPatch - 0.5) * 0.24);
    return prof * cov;
}

// Generic cloud layer: fbm coverage with a vertical profile, used by the stratocumulus, altostratus and cirrus
// layers. The cumulus layer keeps its own field (cloudDensity). wind is the layer's drift (m/s, world x and z).
float layerDensity(vec3 p, vec2 wind, float h0, float h1, vec3 scale, float threshold, int octaves) {
    p.xz -= uF.uTime.x * wind;
    float h = (p.y - h0) / (h1 - h0);
    if (h <= 0.0 || h >= 1.0) return 0.0;
    float prof = smoothstep(0.0, 0.2, h) * (1.0 - smoothstep(0.7, 1.0, h));
    vec3 q = p / scale;
    float n = 0.0, a = 0.5;
    for (int i = 0; i < octaves; i++) {
        n += a * valueNoise3(q);
        q = q * 2.03 + vec3(17.1, 9.3, 5.7);
        a *= 0.5;
    }
    n /= 1.0 - pow(0.5, float(octaves));  // back to [0, 1]
    return prof * smoothstep(threshold, threshold + 0.18, n);
}

// Layer order: 0 cumulus, 1 stratocumulus, 2 altostratus, 3 cirrus.
float layerH0(int kind) { return kind == 0 ? CLOUD_H0 : (kind == 1 ? 650.0 : (kind == 2 ? 3800.0 : 8200.0)); }
float layerH1(int kind) { return kind == 0 ? CLOUD_H1 : (kind == 1 ? 1000.0 : (kind == 2 ? 4400.0 : 8700.0)); }
float layerSigma(int kind) { return kind == 0 ? CLOUD_SIGMA : (kind == 1 ? 0.012 : (kind == 2 ? 0.006 : 0.004)); }
int layerSteps(int kind) { return kind == 0 ? 64 : (kind == 1 ? 24 : (kind == 2 ? 20 : 16)); }

float densityOf(vec3 p, int kind) {
    if (kind == 0) return cloudDensity(p);
    if (kind == 1) return layerDensity(p, vec2(3.0, 1.0), 650.0, 1000.0, vec3(700.0, 90.0, 700.0), 0.52, 5);
    if (kind == 2) return layerDensity(p, vec2(5.0, 2.0), 3800.0, 4400.0, vec3(2500.0, 120.0, 2500.0), 0.46, 4);
    // Cirrus: streaks stretched along the wind (x), thin, and only where the high-altitude fbm is strong.
    return layerDensity(p, vec2(14.0, 3.0), 8200.0, 8700.0, vec3(6000.0, 80.0, 400.0), 0.55, 3);
}

// Ray march through one layer: in-scatter (rgb) and transmittance (a) for that layer alone.
vec4 marchLayer(int kind, vec3 cam, vec3 rd, vec3 sunDir, vec3 lamb, float cosS, float jitter) {
    float t0 = (layerH0(kind) - cam.y) / rd.y;
    float tMax = kind == 3 ? 120000.0 : CLOUD_TMAX;  // cirrus is seen far beyond the cumulus range
    float t1 = min((layerH1(kind) - cam.y) / rd.y, tMax);
    if (t0 > tMax || t1 <= t0) return vec4(0.0, 0.0, 0.0, 1.0);
    float sigma = layerSigma(kind);
    int N = layerSteps(kind);
    float tr = 1.0;
    vec3 Lc = vec3(0.0);
    for (int i = 0; i < N; i++) {
        float fa = float(i) / float(N), fb = float(i + 1) / float(N);
        float ta = t0 + (t1 - t0) * fa * fa;
        float tb = t0 + (t1 - t0) * fb * fb;
        float dt = tb - ta;
        vec3 p = cam + rd * (ta + jitter * dt);
        float rho = densityOf(p, kind);
        if (rho <= 0.0) continue;
        float tau1 = 0.0;
        for (int k = 1; k <= 3; k++) tau1 += densityOf(p + sunDir * (120.0 * float(k)), kind) * 120.0;
        // Multiple-scattering octaves: attenuation and phase anisotropy decay per octave.
        float sunTerm = 0.0, aN = 1.0, cN = 1.0, gN = CLOUD_G;
        for (int n = 0; n < 3; n++) {
            sunTerm += aN * hgPhase(cosS, gN) * exp(-sigma * cN * tau1);
            aN *= 0.5; cN *= 0.5; gN *= 0.5;
        }
        float powder = 1.0 - exp(-2.0 * sigma * rho * 80.0);
        vec3 S = uF.uSunIlluminance.xyz * sunTerm * mix(1.0, powder, 0.6) + lamb;
        float att = exp(-sigma * rho * dt);
        Lc += tr * (1.0 - att) * S;
        tr *= att;
        if (tr < 0.02) break;
    }
    return vec4(Lc, tr);
}

void main() {
    vec2 uv = gl_FragCoord.xy / uPush.uTargetSize.xy;
    vec3 rd = viewDirAt(uv);
    vec3 cam = uF.uCamPos.xyz;
    if (rd.y < 0.01) { outC = vec4(0.0, 0.0, 0.0, 1.0); return; }

    // Sky ambient: average of the sky over four upward directions (about 45 degrees).
    vec3 lamb = 0.25 * (skyViewLookup(vec3(0.0, 1.0, 0.0)) + skyViewLookup(normalize(vec3(0.7, 1.0, 0.0)))
                      + skyViewLookup(normalize(vec3(-0.35, 1.0, 0.6))) + skyViewLookup(normalize(vec3(-0.35, 1.0, -0.6))));

    float jitter = hash21(gl_FragCoord.xy);  // fixed per pixel: banding becomes fine noise
    float cosS = dot(uF.uSunDir.xyz, rd);
    vec3 sunDir = uF.uSunDir.xyz;

    // Layers, lowest first: a ray that looks up meets them in altitude order, so each layer attenuates the
    // layers above it. The composite is kept in the same (radiance, transmittance) form as before.
    const int order[4] = int[4](1, 0, 2, 3);
    vec3 Lc = vec3(0.0);
    float tr = 1.0;
    for (int l = 0; l < 4; l++) {
        vec4 L = marchLayer(order[l], cam, rd, sunDir, lamb, cosS, jitter);
        Lc += tr * L.rgb;
        tr *= L.a;
    }
    outC = vec4(Lc, tr);
}
