// Sky composite at full resolution: sky-view radiance, god rays, the sun disk, and the cloud layer
// (bilinear from the half-resolution cloud target). Writes the HDR target.
layout(location = 0) out vec4 outHdr;

void main() {
    vec2 uv = gl_FragCoord.xy / uPush.uTargetSize.xy;
    vec3 rd = viewDirAt(uv);
    // Sky, minus the in-scatter that clouds shade (negative in the god-ray target).
    vec3 Ls = max(skyViewLookup(rd) + texture(uShaftTex, uv).rgb, vec3(0.0));

    // Sun disk: antialiased edge over one pixel's angular width.
    float c = dot(rd, uF.uSunDir.xyz);
    float cosR = uF.uSunDir.w;
    float disk = smoothstep(cosR - fwidth(c), cosR, c);
    Ls += uF.uSunRadiance.xyz * disk;

    // Aureole: sunlight forward-scattered by aerosols around the disk, a few degrees across, which
    // is what makes the sun look like it is lighting the air. Relative to the disk so it scales with
    // the sun's own radiance; the disk itself is excluded.
    const float AUREOLE_K = 1.0e-3;
    float theta = acos(clamp(c, -1.0, 1.0));
    Ls += uF.uSunRadiance.xyz * (AUREOLE_K * exp(-theta / 0.035) * (1.0 - disk));

    vec4 cloud = texture(uCloudTex, uv);
    outHdr = vec4(Ls * cloud.a + cloud.rgb, 1.0);
}
