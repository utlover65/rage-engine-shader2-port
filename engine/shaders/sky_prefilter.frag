// Prefiltered sky for glossy reflections. The sky-view LUT is blurred over a cone of radius
// PRE_ROUGH(row) and written as three stacked rows of the same layout as the LUT: row k blurs
// with radius 0.03 + 0.16 k radians. Atmosphere only; clouds are added per lookup in skyGlossy.
// Recomputed every frame, so it follows the sun and the exposure without a cache.
layout(location = 0) out vec4 outPre;

const int PRE_TAPS = 24;

void main() {
    float row = floor(gl_FragCoord.y / SKY_H);
    float localY = gl_FragCoord.y - row * SKY_H;
    float rough = PRE_ROUGH0 + PRE_ROUGH_STEP * row;
    // Invert the LUT parametrisation (skyViewUV) for this texel's centre.
    float x01 = (gl_FragCoord.x - 0.5) / (SKY_W - 1.0);
    float y01 = ((localY + 0.5) - 0.5) / (SKY_H - 1.0);
    float phi = x01 * 2.0 * PI;
    float s = 2.0 * y01 - 1.0;
    float e = sign(s) * s * s * 0.5 * PI;
    vec3 d = vec3(cos(e) * cos(phi), sin(e), cos(e) * sin(phi));

    vec3 helper = abs(d.y) < 0.92 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 t = normalize(cross(helper, d));
    vec3 b = cross(d, t);
    vec3 sum = vec3(0.0);
    for (int i = 0; i < PRE_TAPS; i++) {
        float r = rough * sqrt((float(i) + 0.5) / float(PRE_TAPS));
        float a = float(i) * 2.39996323;
        vec3 sd = normalize(d + r * (cos(a) * t + sin(a) * b));
        sum += skyViewLookup(vec3(sd.x, max(sd.y, 0.0005), sd.z));
    }
    outPre = vec4(sum / float(PRE_TAPS), 1.0);
}
