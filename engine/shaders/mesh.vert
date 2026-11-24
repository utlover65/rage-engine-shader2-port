// Scene objects are uploaded in world space. Material class selects a compact procedural PBR
// material in the fragment stage; it avoids a texture dependency for this reference scene.
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNrm;
layout(location = 2) in vec3 aAlb;
layout(location = 3) in float aMaterial;

layout(location = 0) out vec3 vWorld;
layout(location = 1) out vec3 vNormal;
layout(location = 2) out vec3 vAlbedo;
layout(location = 3) flat out float vMaterial;

void main() {
    vWorld = aPos;
    vNormal = aNrm;
    vAlbedo = aAlb;
    vMaterial = aMaterial;
    gl_Position = uF.uViewProj * vec4(aPos, 1.0);
}
