// DOF composite: blend the sharp image (binding 4, uHdrTex) and the blurred image (binding 6,
// uBloomTex) by the circle of confusion. Pixels on the focus plane stay sharp; off-plane pixels
// fall back to the bokeh blur.
layout(location = 0) out vec4 outC;

void main() {
    vec2 uv = gl_FragCoord.xy / uPush.uTargetSize.xy;
    vec4 sharp = texture(uHdrTex, uv);
    vec3 blurred = texture(uBloomTex, uv).rgb;
    float coc = sharp.a;
    float strength = smoothstep(0.5, 24.0, abs(coc));
    outC = vec4(mix(sharp.rgb, blurred, strength), 1.0);
}
