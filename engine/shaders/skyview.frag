// Sky-view LUT: pre-exposed sky radiance for every direction the camera can see, at the
// current camera height and sun. Recomputed each frame (cheap: one lookup per sample).
layout(location = 0) out vec4 outL;

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    float x01 = float(p.x) / (SKY_W - 1.0);
    float y01 = float(p.y) / (SKY_H - 1.0);
    float s = 2.0 * y01 - 1.0;
    float e = s * abs(s) * 0.5 * PI;
    float phi = x01 * 2.0 * PI;
    vec3 d = vec3(cos(e) * cos(phi), sin(e), cos(e) * sin(phi));
    float r0 = RG + uF.uCamPos.y;
    outL = vec4(skyRadiance(r0, d) * uF.uExposure.x, 1.0);
}
