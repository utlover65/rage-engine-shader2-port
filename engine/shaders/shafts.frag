// God rays. Sunlight that a scattering medium would send toward the camera is missing wherever
// a cloud shades that air. This pass integrates the single-scattered in-scatter that the clouds
// remove, along each view ray, up to the sea surface (below the horizon) or the top of the cloud
// layer (above it). The result is a negative radiance, pre-exposed, which the sky composite and
// the sea's aerial perspective add to the unshaded sky. Quarter resolution.
layout(location = 0) out vec4 outShaft;

const int SHAFT_STEPS = 20;
const float SHAFT_TOP = 2000.0;   // above the cloud top (1900 m) the sun ray never meets a cloud
const float SHAFT_TMAX = 60000.0;

// Cloud shade for the shafts: the same slab, drift and profile as the cloud shadows on the sea,
// but only the large-scale noise (no domain warp or detail erosion). Beams are soft, and the
// smoother field keeps the upsampled quarter-resolution result free of pixel grain.
float shaftShade(vec3 p) {
    float sunY = uF.uSunDir.y;
    if (sunY <= 0.015) return 0.0;
    float t = max((1550.0 - p.y) / sunY, 0.0);
    vec3 c = p + uF.uSunDir.xyz * t - vec3(uF.uTime.x * 6.0, 0.0, uF.uTime.x * 2.1);
    float h = (c.y - 1200.0) / 700.0;
    if (h <= 0.0 || h >= 1.0) return 0.0;
    float profile = smoothstep(0.0, 0.25, h) * (1.0 - smoothstep(0.6, 1.0, h));
    vec3 q = c / vec3(450.0, 300.0, 450.0);
    float n = 0.5 * cloudShadowNoise3(q)
            + 0.25 * cloudShadowNoise3(q * 2.03 + vec3(17.1, 9.3, 5.7))
            + 0.125 * cloudShadowNoise3(q * 4.06 + vec3(3.1, 7.7, 1.9));
    float density = profile * smoothstep(0.30, 0.46, n);
    float opticalDepth = 0.03 * density * (700.0 / max(sunY, 0.04));
    return 1.0 - exp(-opticalDepth);
}

void main() {
    vec2 uv = gl_FragCoord.xy / uPush.uTargetSize.xy;
    vec3 rd = viewDirAt(uv);
    vec3 cam = uF.uCamPos.xyz;

    float tMax;
    if (rd.y < -1e-3) {
        tMax = min(-cam.y / rd.y, SHAFT_TMAX);
    } else {
        tMax = min(max(SHAFT_TOP - cam.y, 1.0) / max(rd.y, 1e-3), SHAFT_TMAX);
    }

    const float r0 = RG + cam.y;
    const float cosT = dot(rd, uF.uSunDir.xyz);
    const float phR = 3.0 / (16.0 * PI) * (1.0 + cosT * cosT);
    const float phM = hgPhase(cosT, MIE_G);

    vec3 removed = vec3(0.0);
    vec3 Tv = vec3(1.0);
    float tPrev = 0.0;
    for (int i = 1; i <= SHAFT_STEPS; i++) {
        float f = float(i) / float(SHAFT_STEPS);
        float t1 = tMax * f * f;
        float ds = t1 - tPrev;
        float tm = tPrev + 0.5 * ds;
        vec3 p = cam + rd * tm;

        float rr = sqrt(max(r0 * r0 + 2.0 * r0 * rd.y * tm + tm * tm, 0.0));
        float h = rr - RG;
        float dR = exp(-h / H_R);
        float dM = exp(-h / H_M);
        float muSun = (r0 * uF.uSunDir.y + tm * cosT) / rr;
        vec3 Ts = sunTransmittance(rr, muSun);
        // The share of direct sunlight at this point that clouds block.
        float shade = shaftShade(p);

        vec3 sigE = BETA_R * dR + BETA_M * dM;
        vec3 sigS = BETA_R * dR * phR + BETA_M * dM * phM;
        vec3 src = sigS * SOLAR_LUX * Ts * shade;
        vec3 stepT = exp(-sigE * ds);
        removed += Tv * src * (vec3(1.0) - stepT) / sigE;
        Tv *= stepT;
        tPrev = t1;
    }
    outShaft = vec4(-removed * uF.uExposure.x, 1.0);
}
