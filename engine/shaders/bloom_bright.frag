// Bright pass: 2x2 box downsample of the HDR image to half resolution, keeping only the energy
// above a soft threshold. The input is clamped first so that the sun disk, which is many orders
// of magnitude brighter than the sky, produces a finite halo.
layout(location = 0) out vec4 outB;

// The sky itself is above 1.0 pre-exposed, so the threshold sits well above the sky and only
// the sun's glare and the brightest sea glints bloom.
const float BLOOM_THRESHOLD = 8.0;  // pre-exposed radiance (about 12000 cd/m^2 at EV100 9)
const float BLOOM_CLAMP = 64.0;      // pre-exposed radiance

void main() {
    vec2 uv = gl_FragCoord.xy / uPush.uTargetSize.xy;
    vec2 o = 0.5 / vec2(textureSize(uHdrTex, 0));
    vec3 c = 0.25 * (texture(uHdrTex, uv + vec2(-o.x, -o.y)).rgb
                   + texture(uHdrTex, uv + vec2( o.x, -o.y)).rgb
                   + texture(uHdrTex, uv + vec2(-o.x,  o.y)).rgb
                   + texture(uHdrTex, uv + vec2( o.x,  o.y)).rgb);
    c = min(max(c, vec3(0.0)), vec3(BLOOM_CLAMP));
    float l = dot(c, vec3(0.2126, 0.7152, 0.0722));
    float w = max(l - BLOOM_THRESHOLD, 0.0) / max(l, 1e-4);
    outB = vec4(c * w, 1.0);
}
