// Half-resolution cloud-shadow field. Each pixel intersects the camera ray with the sea plane,
// then evaluates the cloud layer's optical depth along the sun direction. The resulting map is
// sampled by both the ocean and scene-object shaders, avoiding per-full-resolution-pixel 3-D noise.
layout(location = 0) out vec4 outShadow;

void main() {
    vec2 uv = gl_FragCoord.xy / uPush.uTargetSize.xy;
    vec3 rd = viewDirAt(uv);
    if (rd.y >= -1e-4) {
        outShadow = vec4(1.0, 1.0, 1.0, 1.0);
        return;
    }
    float t = -uF.uCamPos.y / rd.y;
    if (t <= 0.0 || t > 150000.0) {
        outShadow = vec4(1.0, 1.0, 1.0, 1.0);
        return;
    }
    vec3 seaPoint = uF.uCamPos.xyz + rd * t;
    float transmittance = cloudShadowTransmittance(seaPoint);
    outShadow = vec4(transmittance, transmittance, transmittance, 1.0);
}
