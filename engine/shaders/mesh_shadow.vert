// Shadow pass: the scene objects, transformed by the sun's view-projection.
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNrm;
layout(location = 2) in vec3 aAlb;

void main() {
    gl_Position = uF.uSunViewProj * vec4(aPos, 1.0);
}
