// Transmittance LUT: T(r, mu) from the observer radius r to the top of the atmosphere along
// a ray with cos zenith mu. Zero where the planet blocks the ray. Computed once.
layout(location = 0) out vec4 outT;

vec3 computeTransmittance(float r, float mu) {
    if (hitsGround(r, mu)) return vec3(0.0);
    float tt = sphereRoot(r, mu, RT, true);
    vec3 tau = vec3(0.0);
    float tPrev = 0.0;
    for (int i = 1; i <= TRANS_STEPS; i++) {
        float f = float(i) / float(TRANS_STEPS);
        float t1 = tt * f * f;
        float tm = 0.5 * (tPrev + t1);
        float rm = sqrt(max(r * r + 2.0 * r * mu * tm + tm * tm, 0.0));
        float h = rm - RG;
        tau += (BETA_R * exp(-h / H_R) + BETA_M * exp(-h / H_M)) * (t1 - tPrev);
        tPrev = t1;
    }
    return exp(-tau);
}

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    float t = 2.0 * float(p.x) / (TRANS_W - 1.0) - 1.0;
    float mu = t * abs(t);
    float v = float(p.y) / (TRANS_H - 1.0);
    float r = RG + (RT - RG) * v * v;
    outT = vec4(computeTransmittance(r, mu), 1.0);
}
