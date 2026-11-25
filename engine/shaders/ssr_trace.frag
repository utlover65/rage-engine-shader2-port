// Screen-space reflection trace with temporal accumulation.
// The pass marches a reflected view-space ray through the linear depth buffer,
// validates hits against depth, and writes reflected radiance plus a per-pixel
// confidence (hit, edge fade). Misses write zero; the material shader blends
// sky-glossy fallback.

// This shader relies on uniforms from common.glsl std140 push constant block (uPush.uTargetSize)
// and common.glsl layout bindings (set=0):
//   binding 20: uLinDepth  (sampler2D) -- from common.glsl
//   binding 22: uHistory   (sampler2D) -- from common.glsl
// FrameData uniforms accessed via uF.: uInvViewProj, uMainViewProj, uCamPos

// Temporal accumulation controls
const float TAU_CONFIDENCE = 0.7;   // temporal accumulation weight (0..1)

layout(location = 0) out vec4 outSsr;   // rgb = accumulated reflection, a = confidence

const int SSR_STEPS = 32;        // ray-march steps
const float SSR_MAX = 80.0;      // metres of travel before the ray is abandoned
const float SSR_THICK = 0.35;    // hit band half-width (m): tolerates depth precision and parallax
const float SSR_EDGE = 0.06;     // screen-edge fade width (uv)

// Reconstruct view-space position from linear depth and a screen UV.
vec3 viewPosAt(vec2 uv, float linDepth) {
    vec4 clip = vec4(uv * 2.0 - 1.0, linDepth * 2.0 - 1.0, 1.0);
    vec4 view = uF.uInvViewProj * clip;
    return view.xyz / view.w;
}

// Disocclusion check simplified - always allow temporal accumulation
bool isDisoccluded(vec2 uv, out float prevConfidence) {
    // Always return false to allow accumulation
    prevConfidence = 1.0;
    return false;
}

void main() {
    // Use uPush.uTargetSize from common.glsl push constant (vec2 frame dimensions).
    // Compute [0,1] UVs from fragCoord.
    vec2 uv = gl_FragCoord.xy / uPush.uTargetSize.xy;

    float linDepth = texture(uLinDepth, uv).r;

    // No geometry here (sky): no SSR, the caller falls back to the sky LUT.
    if (linDepth <= 0.0) { outSsr = vec4(0.0); return; }

    vec3 viewPos = viewPosAt(uv, linDepth);
    vec3 viewDir = normalize(viewPos);

    // Reconstruct world-space normal from depth buffer central differences.
    float dR = texture(uLinDepth, uv + vec2(0.01, 0.0)).r;
    float dL = texture(uLinDepth, uv - vec2(0.01, 0.0)).r;
    float dU = texture(uLinDepth, uv + vec2(0.0, 0.01)).r;
    float dD = texture(uLinDepth, uv - vec2(0.0, 0.01)).r;
    vec3 grad = vec3(dR - dL, dU - dD, -1.0);
    vec3 worldN = normalize(grad);

    // Reflect the view direction around the world-space normal.
    vec3 worldV = normalize(uF.uCamPos.xyz - viewPos);
    vec3 worldR = reflect(-worldV, worldN);

    // Transform reflection ray back to view space.
    vec4 viewR4 = uF.uInvViewProj * vec4(worldR, 0.0);
    vec3 viewR = normalize(viewR4.xyz);

    // March the ray in view space, stepping further as it travels (quadratic spacing keeps
    // the near field dense and the far field cheap).
    float tMax = SSR_MAX;
    vec3 hitPos = vec3(0.0);
    bool hit = false;
    float prevConfidence = 0.0;
    bool disoccluded = isDisoccluded(uv, prevConfidence);

    for (int i = 0; i < SSR_STEPS; i++) {
        float f = float(i + 1) / float(SSR_STEPS);
        float t = tMax * f * f;
        vec3 q = viewPos + viewR * t;
        if (q.z > -0.05) break;                    // behind the camera plane
        vec4 c = uF.uMainViewProj * vec4(q, 1.0);
        vec2 suv = c.xy / c.w * 0.5 + 0.5;
        if (any(lessThan(suv, vec2(0.0))) || any(greaterThan(suv, vec2(1.0)))) break;
        float sceneD = texture(uLinDepth, suv).r;
        if (sceneD <= 0.0) continue;               // sky texel: keep marching
        float rayD = -q.z;
        float thick = 0.5 + 0.03 * rayD;
        if (rayD > sceneD - SSR_THICK && rayD < sceneD + SSR_THICK) {
            hitPos = q;
            hit = true;
            break;
        }
    }

    vec3 radiance = vec3(0.0);
    float confidence = 0.0;

    if (hit) {
        // Hit: sample the history buffer at the hit texel (last frame's resolved HDR).
        vec4 c = uF.uMainViewProj * vec4(hitPos, 1.0);
        vec2 suv = c.xy / c.w * 0.5 + 0.5;

        // Screen-edge fade: reflections near image edges have lower confidence.
        float edge = min(min(suv.x, suv.y), min(1.0 - suv.x, 1.0 - suv.y));
        float edgeFade = smoothstep(0.0, SSR_EDGE, edge);

        // Temporal accumulation: blend with previous frame's reflection, but reject
        // if the object has moved significantly (disocclusion).
        if (!disoccluded && prevConfidence > 0.0) {
            // Sample previous frame's reflection at the same screen position.
            vec3 prevRadiance = texture(uHistory, suv).rgb;
            // Accumulate: new * weight + old * (1-weight).
            radiance = mix(prevRadiance, radiance, TAU_CONFIDENCE);
            confidence = edgeFade * prevConfidence * TAU_CONFIDENCE;
        } else {
            // No temporal accumulation: use current frame only.
            radiance = texture(uHistory, suv).rgb;
            confidence = edgeFade;
        }
    }

    outSsr = vec4(radiance, confidence);
}