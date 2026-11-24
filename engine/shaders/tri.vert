const vec2 kPos[3] = vec2[](vec2(-0.6, 0.6), vec2(0.6, 0.6), vec2(0.0, -0.6));
const vec3 kCol[3] = vec3[](vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0), vec3(0.0, 0.0, 1.0));
layout(location = 0) out vec3 vColor;
void main() {
    gl_Position = vec4(kPos[gl_VertexIndex], 0.0, 1.0);
    vColor = kCol[gl_VertexIndex];
}
