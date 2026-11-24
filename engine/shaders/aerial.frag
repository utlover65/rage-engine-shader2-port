// Aerial-perspective froxels. The atlas holds 16 depth slices side by side (each AER_TW texels
// wide, AER_TH tall, slice k at distance aerialSliceDist(k)). Each texel stores the in-scatter
// from the camera to that slice's distance (flag 0) or the transmittance over the same path
// (flag 1). Mesh and sea shaders read the atlas with aerialLookup instead of integrating per pixel.
layout(location = 0) out vec4 outA;

void main() {
    float k = floor(gl_FragCoord.x / AER_TW);
    float i = floor(gl_FragCoord.x - k * AER_TW);
    float j = floor(gl_FragCoord.y);
    // Texel (i, j) is the froxel at screen uv (i/(TW-1), j/(TH-1)): the same mapping aerialLookup uses.
    vec2 uv = vec2(i, j) / vec2(AER_TW - 1.0, AER_TH - 1.0);
    vec3 rd = viewDirAt(uv);
    float dist = aerialSliceDist(k);
    vec3 Tv;
    vec3 Lin = scatterAlong(RG + uF.uCamPos.y, rd, dist, Tv) * uF.uExposure.x;
    outA = vec4(uPush.uTargetSize.z < 0.5 ? Lin : Tv, 1.0);
}
