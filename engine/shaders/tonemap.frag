// Filmic output transform plus FXAA-style edge filtering. FXAA runs after tone mapping in
// linear display-referred light, avoiding extra render targets or a costly MSAA attachment.
layout(location = 0) out vec4 outC;

const float BLOOM_STRENGTH = 0.075;
const float VIGNETTE_STRENGTH = 0.12;
const float FILM_GRAIN_STRENGTH = 0.0022;
const vec3 SUNSET_HIGHLIGHT = vec3(1.14, 0.80, 1.14);  // magenta push in the highlights
const vec3 SUNSET_SHADOW = vec3(0.055, 0.022, 0.080);  // violet lift in the shadows

vec3 acesFit(vec3 x) {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}
float srgbEncode(float v) {
    return v <= 0.0031308 ? 12.92 * v : 1.055 * pow(max(v, 0.0), 1.0 / 2.4) - 0.055;
}
float luma(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }
float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

vec3 mappedAt(vec2 uv) {
    vec3 hdr = max(texture(uHdrTex, uv).rgb, vec3(0.0));
    hdr += BLOOM_STRENGTH * max(texture(uBloomTex, uv).rgb, vec3(0.0));
    hdr *= uF.uExposure.w;  // auto-exposure gain (1.0 when disabled)
    vec3 mapped = acesFit(hdr);
    // Sunset grade: a violet lift in the shadows and a magenta push in the highlights, the colour
    // cast of low sun through the haze over the bay. Applied after the curve so it does not change
    // the exposure of the atmosphere or the water.
    float lumaMap = luma(mapped);
    mapped = mapped * mix(vec3(1.0), SUNSET_HIGHLIGHT, smoothstep(0.15, 0.75, lumaMap));
    mapped += SUNSET_SHADOW * (1.0 - smoothstep(0.0, 0.5, lumaMap));
    // Highlight desaturation follows film response: very bright sources approach neutral white
    // instead of retaining implausibly saturated clipped channels.
    float y = luma(mapped);
    float highlight = smoothstep(0.58, 1.0, y) * 0.16;
    mapped = mix(mapped, vec3(y), highlight);
    return mapped;
}

void main() {
    vec2 uv = gl_FragCoord.xy / uPush.uTargetSize.xy;
    vec2 px = 1.0 / vec2(textureSize(uHdrTex, 0));
    vec3 m = mappedAt(uv);
    vec3 n = mappedAt(uv + vec2(0.0, -px.y));
    vec3 s = mappedAt(uv + vec2(0.0,  px.y));
    vec3 w = mappedAt(uv + vec2(-px.x, 0.0));
    vec3 e = mappedAt(uv + vec2( px.x, 0.0));
    float lm = luma(m), ln = luma(n), ls = luma(s), lw = luma(w), le = luma(e);
    float lMin = min(lm, min(min(ln, ls), min(lw, le)));
    float lMax = max(lm, max(max(ln, ls), max(lw, le)));

    vec3 result = m;
    if (lMax - lMin >= max(0.0312, lMax * 0.085)) {
        vec3 nw = mappedAt(uv + vec2(-px.x, -px.y));
        vec3 ne = mappedAt(uv + vec2( px.x, -px.y));
        vec3 sw = mappedAt(uv + vec2(-px.x,  px.y));
        vec3 se = mappedAt(uv + vec2( px.x,  px.y));
        float lNW = luma(nw), lNE = luma(ne), lSW = luma(sw), lSE = luma(se);
        vec2 dir;
        dir.x = -((lNW + lNE) - (lSW + lSE));
        dir.y =  ((lNW + lSW) - (lNE + lSE));
        float reduce = max((lNW + lNE + lSW + lSE) * (0.25 * 0.125), 1.0 / 128.0);
        float rcpDirMin = 1.0 / (min(abs(dir.x), abs(dir.y)) + reduce);
        dir = clamp(dir * rcpDirMin, vec2(-8.0), vec2(8.0)) * px;
        vec3 a = 0.5 * (mappedAt(uv + dir * (1.0 / 3.0 - 0.5)) +
                        mappedAt(uv + dir * (2.0 / 3.0 - 0.5)));
        vec3 b = a * 0.5 + 0.25 * (mappedAt(uv + dir * -0.5) + mappedAt(uv + dir * 0.5));
        float lb = luma(b);
        result = (lb < lMin || lb > lMax) ? a : b;
    }
    // A restrained lens falloff and sub-quantisation film grain avoid the sterile, perfectly
    // uniform digital finish. Grain is screen-space and intentionally very subtle.
    vec2 centred = uv * 2.0 - 1.0;
    float vignette = 1.0 - VIGNETTE_STRENGTH * smoothstep(0.28, 1.42, dot(centred, centred));
    float grain = (hash12(gl_FragCoord.xy + uF.uTime.xx) - 0.5) * FILM_GRAIN_STRENGTH;
    result = max(result * vignette + grain, vec3(0.0));
    outC = vec4(srgbEncode(result.r), srgbEncode(result.g), srgbEncode(result.b), 1.0);
}
