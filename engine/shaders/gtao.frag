// Horizon-based ambient occlusion at half resolution. For each pixel, AO_DIRS screen directions
// (rotated per pixel by a hash, which the blur pass then smooths) are searched out to AO_RADIUS
// metres. The largest elevation of a sample above the surface tangent, weighted by distance,
// is the occluded horizon angle. Output: visibility, 1 = open sky.
layout(location = 0) out float outAO;

const float AO_RADIUS = 2.5;      // m
const int AO_DIRS = 4;
const int AO_STEPS = 4;
const float AO_STRENGTH = 1.1;

// World position of the full-resolution pixel at uv, from its linear depth.
vec3 aoWorldAt(vec2 uv, float d) {
    return uF.uCamPos.xyz + viewDirAt(uv) * d;
}

void main() {
    vec2 uv = gl_FragCoord.xy / uPush.uTargetSize.xy;
    float d = texture(uLinDepth, uv).r;
    if (d <= 0.0) { outAO = 1.0; return; }
    vec3 rd = viewDirAt(uv);
    vec3 P = uF.uCamPos.xyz + rd * d;
    vec3 V = -rd;

    // Normal from the depth neighbours (two full-resolution texels away).
    vec2 tx = vec2(2.0) / vec2(textureSize(uLinDepth, 0));
    vec3 px = aoWorldAt(uv + vec2(tx.x, 0.0), texture(uLinDepth, uv + vec2(tx.x, 0.0)).r) - P;
    vec3 py = aoWorldAt(uv + vec2(0.0, tx.y), texture(uLinDepth, uv + vec2(0.0, tx.y)).r) - P;
    vec3 N = normalize(cross(px, py));
    if (dot(N, V) < 0.0) N = -N;

    // Screen-space radius of AO_RADIUS metres at this depth (projected world-right offset).
    vec3 right = normalize(cross(vec3(0.0, 1.0, 0.0), V));
    vec4 c0 = uF.uViewProj * vec4(P, 1.0);
    vec4 c1 = uF.uViewProj * vec4(P + right * AO_RADIUS, 1.0);
    float radiusUV = length(c1.xy / c1.w - c0.xy / c0.w) * 0.5;

    float angle0 = 6.2831853 * fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));
    float occl = 0.0;
    for (int i = 0; i < AO_DIRS; i++) {
        float a = angle0 + 6.2831853 * float(i) / float(AO_DIRS);
        vec2 dir = vec2(cos(a), sin(a));
        float best = 0.0;
        for (int s = 1; s <= AO_STEPS; s++) {
            vec2 suv = uv + dir * radiusUV * (float(s) / float(AO_STEPS));
            if (any(lessThan(suv, vec2(0.0))) || any(greaterThan(suv, vec2(1.0)))) break;
            float sd = texture(uLinDepth, suv).r;
            if (sd <= 0.0) continue;
            vec3 v = aoWorldAt(suv, sd) - P;
            float dist = length(v);
            if (dist < 1e-3 || dist > AO_RADIUS) continue;
            float w = 1.0 - smoothstep(0.5 * AO_RADIUS, AO_RADIUS, dist);
            best = max(best, dot(v / dist, N) * w);
        }
        occl += clamp(best, 0.0, 1.0);
    }
    outAO = clamp(1.0 - AO_STRENGTH * occl / float(AO_DIRS), 0.0, 1.0);
}
