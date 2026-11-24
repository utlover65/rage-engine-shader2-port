// Separable bokeh blur for the adaptive DOF. Reads the prepared image (colour + CoC in alpha)
// at binding 4 (uHdrTex) and writes the horizontally (uPush.uTargetSize.z = 0) or vertically
// (= 1) blurred result. The blur radius scales with the pixel's circle of confusion; taps are
// weighted by a Gaussian of their offset. The original CoC is carried in alpha so the second
// separable pass and the composite keep the same radius.
layout(location = 0) out vec4 outC;

const int TAPS = 4;  // kernel radius in taps (kernel is 2 * TAPS + 1)

void main() {
    vec2 uv = gl_FragCoord.xy / uPush.uTargetSize.xy;
    vec2 px = 1.0 / uPush.uTargetSize.xy;
    vec4 c = texture(uHdrTex, uv);
    float coc = c.a;
    float radius = abs(coc);
    vec2 dir = uPush.uTargetSize.z > 0.5 ? vec2(0.0, 1.0) : vec2(1.0, 0.0);
    vec3 acc = vec3(0.0);
    float wsum = 0.0;
    for (int i = -TAPS; i <= TAPS; i++) {
        float t = float(i) / float(TAPS);
        vec2 off = dir * (t * radius) * px;
        vec4 s = texture(uHdrTex, uv + off);
        float w = exp(-t * t * 3.0);
        acc += s.rgb * w;
        wsum += w;
    }
    vec3 blurred = acc / max(wsum, 1e-4);
    outC = vec4(blurred, coc);
}
