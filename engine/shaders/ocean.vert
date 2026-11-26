// Ocean surface. Each mesh vertex is a rest-plane point. Its height and horizontal displacement come from
// Tessendorf's FFT ocean: ocean_evolve.comp and fft2d_stage.comp fill the three fields (the spectrum is in
// src/ocean/ocean_spectrum.h). A short ladder of analytic components (3 m down to 0.4 m) adds the wavelets the
// grid cannot hold. Outputs world position, the normal from the surface tangents and the Jacobian determinant of
// the displacement (folds, used for foam).
layout(location = 0) in vec2 aRest;   // rest-plane offset from the camera (m)

layout(location = 0) out vec3 vWorld;
layout(location = 1) out vec3 vNormal;
layout(location = 2) out float vJacDet;

// FFT fields (real parts): height and the horizontal choppy displacement along x and z, in metres.
// Storage slots 0..2 of the descriptor set (kMaxStorageBuffers), bindings 25..27.
layout(std430, set = 0, binding = 25) readonly buffer OceanHeight { vec2 oceanH[]; };
layout(std430, set = 0, binding = 26) readonly buffer OceanDispX { vec2 oceanDx[]; };
layout(std430, set = 0, binding = 27) readonly buffer OceanDispZ { vec2 oceanDz[]; };
const int OCEAN_N = 256;       // grid points per side; equals kOceanN
const float OCEAN_L = 1024.0;  // patch period (m); equals kOceanL

// Periodic bilinear sample of the three FFT fields at rest point p: (height, Dx, Dz).
vec3 oceanFft(vec2 p) {
    vec2 g = p * (float(OCEAN_N) / OCEAN_L);
    vec2 fl = floor(g);
    vec2 f = g - fl;
    ivec2 i0 = ivec2(fl) & ivec2(OCEAN_N - 1);
    ivec2 i1 = (i0 + ivec2(1)) & ivec2(OCEAN_N - 1);
    int a = i0.y * OCEAN_N + i0.x;
    int b = i0.y * OCEAN_N + i1.x;
    int c = i1.y * OCEAN_N + i0.x;
    int d = i1.y * OCEAN_N + i1.x;
    return vec3(mix(mix(oceanH[a].x, oceanH[b].x, f.x), mix(oceanH[c].x, oceanH[d].x, f.x), f.y),
                mix(mix(oceanDx[a].x, oceanDx[b].x, f.x), mix(oceanDx[c].x, oceanDx[d].x, f.x), f.y),
                mix(mix(oceanDz[a].x, oceanDz[b].x, f.x), mix(oceanDz[c].x, oceanDz[d].x, f.x), f.y));
}

// Analytic detail ladder, above the FFT's band (the grid holds waves of 8 m and longer).
const int SEA_OCT = 4;
const float SEA_S0 = 0.04;     // slope per component at the reference wavelength
const float SEA_Q_HI = 0.2;    // steepness, first two components
const float SEA_Q_LO = 0.05;   // steepness, the rest
const float SEA_LAMBDA0 = 8.0; // reference wavelength (m)

// Deterministic hash in [0, 1) for component seeds (sine-free).
float seaHash(float n) {
    float h = fract(n * 0.1031);
    h *= h + 33.33;
    return fract(h * (h + n));
}

struct SeaSurf {
    float y;              // height
    vec2  gy;             // gradient of height over the rest point
    vec2  D;              // horizontal displacement, rest point to world point
    float Jxx, Jxz, Jzz;  // Jacobian of rest point -> world xz (symmetric)
};

// Detail ladder at rest point x0. Components shorter than the pixel footprint at this distance fade out.
SeaSurf seaWave(vec2 x0, float dist) {
    SeaSurf s;
    s.y = 0.0; s.D = vec2(0.0); s.gy = vec2(0.0);
    s.Jxx = 1.0; s.Jxz = 0.0; s.Jzz = 1.0;
    float f = dist * uF.uExposure.y;
    float lambda = 3.0;  // the ladder runs 3 m down to 0.375 m
    for (int j = 0; j < SEA_OCT; j++) {
        float fj = float(j);
        float k = 6.2831853 / lambda;
        float w = smoothstep(2.0 * f, 6.0 * f, lambda);
        // Directional spread: about +-1.1 rad around the wind, varying per component.
        float th = 1.1 * (2.0 * seaHash(fj + 0.5) - 1.0);
        vec2 d = vec2(sin(th), cos(th));
        float q = (j < 2) ? SEA_Q_HI : SEA_Q_LO;
        float amp = SEA_S0 * pow(lambda / SEA_LAMBDA0, 0.25);
        float ph = k * dot(d, x0) + 6.2831853 * seaHash(fj + 7.3);
        float sn = sin(ph), cs = cos(ph);
        s.y += w * (amp / k) * sn;
        s.gy += w * amp * d * cs;
        s.D += w * (q / k) * d * cs;
        float qs = w * q * sn;
        s.Jxx -= qs * d.x * d.x;
        s.Jxz -= qs * d.x * d.y;
        s.Jzz -= qs * d.y * d.y;
        lambda *= 0.5;
    }
    return s;
}

void main() {
    vec2 x0 = aRest + uF.uCamPos.xz;
    float dist = distance(vec3(x0.x, 0.0, x0.y), uF.uCamPos.xyz);
    // FFT field and its derivatives, by central differences over one grid cell.
    float e = 0.5 * OCEAN_L / float(OCEAN_N);
    vec3 s  = oceanFft(x0);
    vec3 sx = (oceanFft(x0 + vec2(e, 0.0)) - oceanFft(x0 - vec2(e, 0.0))) / (2.0 * e);
    vec3 sz = (oceanFft(x0 + vec2(0.0, e)) - oceanFft(x0 - vec2(0.0, e))) / (2.0 * e);
    // sx = (dh/dx, dDx/dx, dDz/dx); sz = (dh/dz, dDx/dz, dDz/dz).
    SeaSurf w = seaWave(x0, dist);
    float y = s.x + w.y;
    vec2 D = s.yz + w.D;
    vec2 gy = vec2(sx.x, sz.x) + w.gy;
    float Jxx = (1.0 + sx.y) + (w.Jxx - 1.0);
    float Jzz = (1.0 + sz.z) + (w.Jzz - 1.0);
    float Jxz = 0.5 * (sz.y + sx.z) + w.Jxz;
    vec3 world = vec3(x0.x + D.x, y, x0.y + D.y);
    vec3 tx = vec3(Jxx, gy.x, Jxz);
    vec3 tz = vec3(Jxz, gy.y, Jzz);
    vec3 n = normalize(cross(tz, tx));
    if (n.y < 0.0) n = -n;
    vWorld = world;
    vNormal = n;
    vJacDet = Jxx * Jzz - Jxz * Jxz;
    gl_Position = uF.uViewProj * vec4(world, 1.0);
}
