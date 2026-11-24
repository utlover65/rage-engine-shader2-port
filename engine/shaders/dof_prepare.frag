// Adaptive depth-of-field: prepare pass. Reads the resolved HDR (binding 4, uHdrTex) and the
// linear depth (binding 20, uLinDepth) and writes colour plus the circle of confusion in pixels
// (alpha) for the separable bokeh blur. Focus parameters come from uF.uDof:
//   x = focal length (mm), y = f-number, z = focus distance (m), w = sensor height (mm).
layout(location = 0) out vec4 outC;

const float MAX_COC_PX = 24.0;

void main() {
    vec2 uv = gl_FragCoord.xy / uPush.uTargetSize.xy;
    vec3 col = max(texture(uHdrTex, uv).rgb, vec3(0.0));
    float depth = texture(uLinDepth, uv).r;          // metres
    float f = max(uF.uDof.x, 1.0);                  // focal length (mm)
    float aperture = max(uF.uDof.y, 0.5);            // f-number
    float focusDist = max(uF.uDof.z, 0.1);           // focus distance (m)
    float sensorH = max(uF.uDof.w, 1.0);             // sensor height (mm)
    float D = depth * 1000.0;                        // object distance (mm)
    float S = focusDist * 1000.0;                    // focus distance (mm)
    // Thin-lens circle of confusion on the sensor (mm): A * |S - D| / D * f / (S - f),
    // with A = f / N the aperture diameter. Positive behind the focus plane, negative in front.
    float A = f / aperture;
    float cocMm = A * abs(S - D) / max(D, 1e-3) * f / max(S - f, 1e-3);
    // Convert to pixels using the target height.
    float cocPx = cocMm / sensorH * uPush.uTargetSize.y;
    float coc = clamp(cocPx, -MAX_COC_PX, MAX_COC_PX);
    outC = vec4(col, coc);
}
