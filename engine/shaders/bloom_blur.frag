// Separable 9-tap Gaussian blur, taps 2 texels apart. uPush.uTargetSize.z selects the
// direction: > 0.5 horizontal, otherwise vertical.
layout(location = 0) out vec4 outB;

void main() {
    vec2 uv = gl_FragCoord.xy / uPush.uTargetSize.xy;
    vec2 texel = 1.0 / vec2(textureSize(uHdrTex, 0));
    vec2 dir = uPush.uTargetSize.z > 0.5 ? vec2(texel.x, 0.0) : vec2(0.0, texel.y);

    const float w0 = 0.2270;
    const float w[4] = float[4](0.1946, 0.1216, 0.0541, 0.0162);
    vec3 sum = texture(uHdrTex, uv).rgb * w0;
    for (int i = 0; i < 4; i++) {
        vec2 o = dir * (2.0 * float(i + 1));
        sum += w[i] * (texture(uHdrTex, uv + o).rgb + texture(uHdrTex, uv - o).rgb);
    }
    outB = vec4(sum, 1.0);
}
