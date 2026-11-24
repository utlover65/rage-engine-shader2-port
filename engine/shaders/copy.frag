// Copies the bound colour texture (binding 4) to the target: used to keep the TAA history.
layout(location = 0) out vec4 outCopy;

void main() {
    vec2 uv = gl_FragCoord.xy / uPush.uTargetSize.xy;
    outCopy = texture(uHdrTex, uv);
}
