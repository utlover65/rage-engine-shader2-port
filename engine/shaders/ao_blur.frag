// Depth-aware 5 x 5 blur of the half-resolution AO. Neighbours across a depth edge (a pier against
// the sea, the headland against the sky) get little weight, so occlusion does not bleed over them.
layout(location = 0) out float outBlur;

void main() {
    vec2 uv = gl_FragCoord.xy / uPush.uTargetSize.xy;
    vec2 texel = 1.0 / uPush.uTargetSize.xy;
    float d0 = texture(uLinDepth, uv).r;
    float sum = 0.0, wsum = 0.0;
    for (int y = -2; y <= 2; y++) {
        for (int x = -2; x <= 2; x++) {
            vec2 o = vec2(float(x), float(y)) * texel;
            float g = exp(-0.5 * float(x * x + y * y) / 2.0);
            float d = texture(uLinDepth, uv + o).r;
            float depthW = exp(-abs(d - d0) / (0.03 * d0 + 0.05));
            float w = g * depthW;
            sum += w * texture(uAO, uv + o).r;
            wsum += w;
        }
    }
    outBlur = wsum > 0.0 ? sum / wsum : 1.0;
}
