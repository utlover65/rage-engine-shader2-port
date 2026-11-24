// Temporal anti-aliasing resolve. The current frame is jittered by a sub-pixel offset (see
// renderFrame). Each pixel is reprojected into the previous frame's resolved colour using its world
// position from this frame's linear depth (the sky reprojects by camera rotation), the history is
// clamped to the 3 x 3 neighbourhood of the current frame, and the two are blended.
layout(location = 0) out vec4 outResolved;

const float TAA_BLEND = 0.1;   // weight of the current frame; the rest comes from history

void main() {
    vec2 uv = gl_FragCoord.xy / uPush.uTargetSize.xy;
    vec3 cur = texture(uHdrTex, uv).rgb;
    if (uF.uTaa.z < 0.5) { outResolved = vec4(cur, 1.0); return; }

    float d = texture(uLinDepth, uv).r;
    vec3 rd = viewDirAt(uv);
    vec3 world = uF.uCamPos.xyz + rd * (d > 0.0 ? d : 1000.0);
    vec4 c = uF.uPrevViewProj * vec4(world, 1.0);
    vec2 huv = c.xy / c.w * 0.5 + 0.5;
    if (any(lessThan(huv, vec2(0.0))) || any(greaterThan(huv, vec2(1.0)))) { outResolved = vec4(cur, 1.0); return; }

    vec2 texel = 1.0 / uPush.uTargetSize.xy;
    vec3 lo = cur, hi = cur;
    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            vec3 s = texture(uHdrTex, uv + vec2(float(x), float(y)) * texel).rgb;
            lo = min(lo, s);
            hi = max(hi, s);
        }
    }
    vec3 hist = clamp(texture(uHistory, huv).rgb, lo, hi);
    outResolved = vec4(mix(hist, cur, TAA_BLEND), 1.0);
}
