// Scene objects are uploaded in world space. Material class selects a compact procedural PBR
// material in the fragment stage; it avoids a texture dependency for this reference scene.
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNrm;
layout(location = 2) in vec3 aAlb;
layout(location = 3) in float aMaterial;

// Wind deformation for vegetation. Each vertex is displaced along a wind direction
// vector based on its height and the wind strength, creating a gentle swaying motion.
// The displacement is modulated by a sine wave over time for animated effect.
layout(std430, set = 0, binding = 28) readonly buffer WindData {
    vec3 windDirection;    // normalized wind direction
    float windSpeed;       // speed in radians per second
    float windFrequency;   // how quickly the vertex sways
    float windAmplitude;   // maximum displacement distance
};

layout(location = 0) out vec3 vWorld;
layout(location = 1) out vec3 vNormal;
layout(location = 2) out vec3 vAlbedo;
layout(location = 3) flat out float vMaterial;

void main() {
    // Apply wind deformation to the vertex position.
    // The displacement is along the wind direction, modulated by the vertex height
    // and a time-varying sine wave.
    float t = uF.uTime.x;  // time in seconds
    float height = aPos.y; // assume y is up axis
    float displacement = sin(t * windFrequency + height * 0.5) * windAmplitude;
    vec3 displacedPos = aPos + windDirection * displacement;

    vWorld = displacedPos;
    vNormal = aNrm; // normal is not wind-deformed (or could be rotated for simple approximation)
    vAlbedo = aAlb;
    vMaterial = aMaterial;
    gl_Position = uF.uViewProj * vec4(displacedPos, 1.0);
}
