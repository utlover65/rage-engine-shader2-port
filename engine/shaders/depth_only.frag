// Depth prepass for the scene objects and the sea: linear depth, the distance along the view ray
// from the camera (the same quantity the lighting passes use as dist).
layout(location = 0) in vec3 vWorld;
layout(location = 0) out float outDepth;

void main() {
    outDepth = distance(vWorld, uF.uCamPos.xyz);
}
