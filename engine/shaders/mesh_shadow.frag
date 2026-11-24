// Shadow pass: store 1 - depth, so the cleared background (0) casts no shadow.
layout(location = 0) out vec4 outDepth;

void main() {
    outDepth = vec4(clamp(1.0 - gl_FragCoord.z, 0.0, 1.0), 0.0, 0.0, 1.0);
}
