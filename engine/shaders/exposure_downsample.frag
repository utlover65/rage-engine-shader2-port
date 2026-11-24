// Auto-exposure: first reduction. Downsample the resolved HDR (binding 4, uHdrTex) to a small
// luminance grid; each output texel holds the log-average luminance of its source block (the
// geometric mean, computed in log space to keep highlights from dominating).
layout(location = 0) out float outLum;

void main() {
    vec2 outUV = gl_FragCoord.xy / uPush.uTargetSize.xy;  // 0..1 over the small target
    ivec2 srcSize = textureSize(uHdrTex, 0);
    vec2 block = vec2(srcSize) / uPush.uTargetSize.xy;    // source pixels covered by one texel
    vec2 base = outUV * vec2(srcSize);                    // source coordinate of the block origin
    float sum = 0.0;
    int n = 0;
    for (int y = 0; y < 4; y++)
        for (int x = 0; x < 4; x++) {
            vec2 p = (base + (vec2(float(x), float(y)) + 0.5) * block / 4.0) / vec2(srcSize);
            vec3 c = max(texture(uHdrTex, p).rgb, vec3(0.0));
            float l = dot(c, vec3(0.2126, 0.7152, 0.0722));
            sum += log(max(l, 1e-4));
            n++;
        }
    outLum = exp(sum / float(n));
}
