// Multiple-scattering LUT. Texel (x, y): x maps to the cosine of the zenith angle of the sun
// mu = 2x - 1 at the point; y maps to the radius through rToV, the same as the transmittance
// table. For each point, 64 near-uniform directions are marched in 20 linear steps. L1 is the
// mean over directions of the once-scattered sun light (per 4*pi, isotropic phase); fsum is the
// mean fraction of light scattered per unit path, which feeds the higher orders. The infinite
// multiple-scattering sum is then L1 / (1 - fms). Computed once.
layout(location = 0) out vec4 outM;

const int MS_DIRS = 64;
const int MS_STEPS = 20;

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    float mu = 2.0 * float(p.x) / (MS_W - 1.0) - 1.0;
    float v = float(p.y) / (MS_H - 1.0);
    float r0 = RG + (RT - RG) * v * v;

    // Sun direction in the local frame of the point (y is the local zenith).
    vec3 sunLocal = vec3(sqrt(max(1.0 - mu * mu, 0.0)), mu, 0.0);

    vec3 L1 = vec3(0.0);
    vec3 fsum = vec3(0.0);
    for (int d = 0; d < MS_DIRS; d++) {
        // Fibonacci sphere: near-uniform directions. w.y is the cosine to the local zenith.
        float z = 1.0 - 2.0 * (float(d) + 0.5) / float(MS_DIRS);
        float phi = float(d) * 2.39996323;
        float rxy = sqrt(max(1.0 - z * z, 0.0));
        vec3 w = vec3(rxy * cos(phi), z, rxy * sin(phi));

        float tMax = hitsGround(r0, w.y) ? sphereRoot(r0, w.y, RG, false)
                                         : sphereRoot(r0, w.y, RT, true);
        if (tMax <= 0.0) continue;

        vec3 Tv = vec3(1.0);
        float tPrev = 0.0;
        for (int i = 1; i <= MS_STEPS; i++) {
            float t1 = tMax * float(i) / float(MS_STEPS);
            float ds = t1 - tPrev;
            float tm = 0.5 * (t1 + tPrev);
            float r = sqrt(max(r0 * r0 + 2.0 * r0 * w.y * tm + tm * tm, 0.0));
            float h = r - RG;
            // Extinction equals scattering (no absorption in the air model), so the step's
            // scattered fraction is 1 - exp(-sigma ds) per channel.
            vec3 sigE = BETA_R * exp(-h / H_R) + BETA_M * exp(-h / H_M);
            vec3 stepT = exp(-sigE * ds);
            // Cosine of the sun zenith at the sample: (position . sun) / r.
            float muSun = (r0 * mu + tm * dot(w, sunLocal)) / r;
            vec3 Ts = sunTransmittance(r, muSun);
            vec3 scattered = vec3(1.0) - stepT;
            L1 += Tv * scattered * SOLAR_LUX * Ts;
            fsum += Tv * scattered;
            Tv *= stepT;
            tPrev = t1;
        }
    }

    // Mean over the sphere: divide the direction sum by the count and by 4*pi (isotropic phase).
    vec3 Lmean = L1 / (4.0 * PI * float(MS_DIRS));
    vec3 fms = clamp(fsum / float(MS_DIRS), vec3(0.0), vec3(0.95));
    outM = vec4(Lmean / (vec3(1.0) - fms), 1.0);
}
