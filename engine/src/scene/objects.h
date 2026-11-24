// Procedural scene geometry. Vertex layout: position (3), normal (3), linear albedo (3),
// material class (1). All geometry is emitted as a triangle list and is shared by the
// colour and sun-shadow passes.
#pragma once

#include <cmath>
#include <vector>

namespace engine {
namespace scene {

enum Material : int {
    Wood = 0,
    CrateWood = 1,
    PaintedHull = 2,
    Concrete = 3,
    Metal = 4,
    Lantern = 5,
    Rock = 6,
    Sand = 7,
    Asphalt = 8,
    RoadPaint = 9,
    Pavers = 10,
    CarPaint = 11,
    CarGlass = 12,
    Stone = 13,
};

struct Box {
    float c[3];
    float h[3];
    float albedo[3];
    float material = Wood;
};

inline void appendVertex(std::vector<float>& v, const float* p, const float* n,
                         const float* albedo, float material) {
    v.insert(v.end(), p, p + 3);
    v.insert(v.end(), n, n + 3);
    v.insert(v.end(), albedo, albedo + 3);
    v.push_back(material);
}

inline void appendTriangle(std::vector<float>& v, const float* a, const float* b, const float* c,
                           const float* albedo, float material) {
    const float ux = b[0] - a[0], uy = b[1] - a[1], uz = b[2] - a[2];
    const float vx = c[0] - a[0], vy = c[1] - a[1], vz = c[2] - a[2];
    float n[3] = {uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx};
    const float l = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
    if (l > 1e-8f) { n[0] /= l; n[1] /= l; n[2] /= l; }
    appendVertex(v, a, n, albedo, material);
    appendVertex(v, b, n, albedo, material);
    appendVertex(v, c, n, albedo, material);
}

inline void appendQuad(std::vector<float>& v, const float* a, const float* b, const float* c,
                       const float* d, const float* albedo, float material) {
    appendTriangle(v, a, b, c, albedo, material);
    appendTriangle(v, a, c, d, albedo, material);
}

inline void appendBox(std::vector<float>& v, const Box& b) {
    const float x0 = b.c[0] - b.h[0], x1 = b.c[0] + b.h[0];
    const float y0 = b.c[1] - b.h[1], y1 = b.c[1] + b.h[1];
    const float z0 = b.c[2] - b.h[2], z1 = b.c[2] + b.h[2];
    struct Face { float n[3]; float p[4][3]; };
    const Face faces[6] = {
        {{1,0,0},  {{x1,y0,z0},{x1,y1,z0},{x1,y1,z1},{x1,y0,z1}}},
        {{-1,0,0}, {{x0,y0,z1},{x0,y1,z1},{x0,y1,z0},{x0,y0,z0}}},
        {{0,1,0},  {{x0,y1,z0},{x0,y1,z1},{x1,y1,z1},{x1,y1,z0}}},
        {{0,-1,0}, {{x0,y0,z1},{x0,y0,z0},{x1,y0,z0},{x1,y0,z1}}},
        {{0,0,1},  {{x0,y0,z1},{x1,y0,z1},{x1,y1,z1},{x0,y1,z1}}},
        {{0,0,-1}, {{x1,y0,z0},{x0,y0,z0},{x0,y1,z0},{x1,y1,z0}}},
    };
    static const int order[6] = {0, 1, 2, 0, 2, 3};
    for (const Face& f : faces)
        for (int k : order) appendVertex(v, f.p[k], f.n, b.albedo, b.material);
}

// Tapered cylinder/cone with smooth side normals and flat, closed caps.
inline void appendCylinder(std::vector<float>& v, float x, float z, float y0, float y1,
                           float r0, float r1, int segments, const float* albedo, float material) {
    const float slope = (r0 - r1) / (y1 - y0);
    for (int i = 0; i < segments; ++i) {
        const float a0 = 6.28318530718f * static_cast<float>(i) / segments;
        const float a1 = 6.28318530718f * static_cast<float>(i + 1) / segments;
        float p00[3] = {x + r0 * std::cos(a0), y0, z + r0 * std::sin(a0)};
        float p01[3] = {x + r1 * std::cos(a0), y1, z + r1 * std::sin(a0)};
        float p11[3] = {x + r1 * std::cos(a1), y1, z + r1 * std::sin(a1)};
        float p10[3] = {x + r0 * std::cos(a1), y0, z + r0 * std::sin(a1)};
        float n0[3] = {std::cos(a0), slope, std::sin(a0)};
        float n1[3] = {std::cos(a1), slope, std::sin(a1)};
        for (float* n : {n0, n1}) {
            const float len = std::sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);
            n[0] /= len; n[1] /= len; n[2] /= len;
        }
        // Two side triangles with interpolated radial normals.
        if (r1 > 1e-5f) {
            appendVertex(v, p00, n0, albedo, material); appendVertex(v, p01, n0, albedo, material); appendVertex(v, p11, n1, albedo, material);
        }
        appendVertex(v, p00, n0, albedo, material); appendVertex(v, p11, n1, albedo, material); appendVertex(v, p10, n1, albedo, material);
        float cb[3] = {x,y0,z}, ct[3] = {x,y1,z};
        if (r0 > 1e-5f) appendTriangle(v, cb, p00, p10, albedo, material);
        if (r1 > 1e-5f) appendTriangle(v, ct, p11, p01, albedo, material);
    }
}

inline void appendBoatHull(std::vector<float>& v, float cx, float cy, float cz,
                           float length, float width, const float* albedo) {
    constexpr int rings = 8, sides = 20;
    const float zf[rings] = {-0.50f,-0.40f,-0.27f,-0.10f,0.10f,0.27f,0.40f,0.50f};
    const float wf[rings] = {0.05f,0.58f,0.83f,1.00f,0.98f,0.86f,0.62f,0.05f};
    const float yf[rings] = {0.00f,0.04f,0.08f,0.10f,0.10f,0.08f,0.04f,0.00f};
    for (int r = 0; r + 1 < rings; ++r) {
        for (int s = 0; s < sides; ++s) {
            const float a0 = 6.28318530718f * static_cast<float>(s) / sides;
            const float a1 = 6.28318530718f * static_cast<float>(s + 1) / sides;
            auto point = [&](int ri, float a, float out[3]) {
                out[0] = cx + width * wf[ri] * std::cos(a);
                out[1] = cy + 0.58f + yf[ri] - 0.58f * std::sin(a);
                out[2] = cz + length * zf[ri];
            };
            float a[3], b[3], c[3], d[3];
            point(r, a0, a); point(r, a1, b); point(r + 1, a1, c); point(r + 1, a0, d);
            appendQuad(v, a, d, c, b, albedo, PaintedHull);
        }
    }
}


// A triangulated, irregular rocky islet around the lighthouse. The outer rings descend below
// sea level so the water naturally occludes the submerged slope and leaves a broken shoreline.
// Normals are derived from the same height field rather than being flat per triangle.
inline void appendLighthouseIslet(std::vector<float>& v, float cx, float cz) {
    constexpr int rings = 19;
    constexpr int segments = 112;
    constexpr float pi = 3.14159265358979323846f;
    const float rock[3] = {0.31f, 0.30f, 0.27f};
    const float sand[3] = {0.48f, 0.39f, 0.25f};
    auto heightAt = [&](float r, float a) {
        const float irregular = 0.18f * std::sin(3.0f * a + 0.4f) +
                                0.10f * std::sin(7.0f * a - 1.1f) +
                                0.07f * std::sin(0.72f * r + 4.0f * a);
        const float innerRidge = 1.55f * std::exp(-((r - 8.0f) * (r - 8.0f)) / 30.0f);
        const float centreRise = 0.82f * std::exp(-r / 5.5f);
        const float outerSlope = 0.145f * std::max(r - 16.0f, 0.0f);
        return 0.26f + innerRidge + centreRise + irregular - outerSlope;
    };
    auto emit = [&](float r, float a, float material) {
        const float da = 2.0f * pi / static_cast<float>(segments);
        const float dr = 0.12f;
        const float h = heightAt(r, a);
        const float dhdr = (heightAt(r + dr, a) - heightAt(std::max(0.05f, r - dr), a)) / (2.0f * dr);
        const float dhda = (heightAt(r, a + da * 0.25f) - heightAt(r, a - da * 0.25f)) / (0.5f * da);
        const float invR = 1.0f / std::max(r, 0.35f);
        const float dhdx = dhdr * std::cos(a) - dhda * invR * std::sin(a);
        const float dhdz = dhdr * std::sin(a) + dhda * invR * std::cos(a);
        float n[3] = {-dhdx, 1.0f, -dhdz};
        const float nl = std::sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);
        n[0] /= nl; n[1] /= nl; n[2] /= nl;
        float pos[3] = {cx + r * std::cos(a), h, cz + r * std::sin(a)};
        const float* alb = material < 0.5f ? rock : sand;
        appendVertex(v, pos, n, alb, material < 0.5f ? Rock : Sand);
    };

    // Centre fan.
    float centre[3] = {cx, heightAt(0.0f, 0.0f), cz};
    float up[3] = {0.0f, 1.0f, 0.0f};
    const float* centreAlb = rock;
    for (int seg = 0; seg < segments; ++seg) {
        const float a0 = 2.0f * pi * static_cast<float>(seg) / segments;
        const float a1 = 2.0f * pi * static_cast<float>(seg + 1) / segments;
        const float r0 = 0.65f;
        appendVertex(v, centre, up, centreAlb, Rock);
        emit(r0, a0, 0.0f);
        emit(r0, a1, 0.0f);
    }

    for (int ring = 0; ring < rings - 1; ++ring) {
        const float r0 = 0.65f + (28.0f - 0.65f) * static_cast<float>(ring) / (rings - 1);
        const float r1 = 0.65f + (28.0f - 0.65f) * static_cast<float>(ring + 1) / (rings - 1);
        for (int seg = 0; seg < segments; ++seg) {
            const float a0 = 2.0f * pi * static_cast<float>(seg) / segments;
            const float a1 = 2.0f * pi * static_cast<float>(seg + 1) / segments;
            const float midR = 0.5f * (r0 + r1);
            const float material = (midR > 13.5f && midR < 21.0f) ? 1.0f : 0.0f;
            emit(r0, a0, material); emit(r1, a0, material); emit(r1, a1, material);
            emit(r0, a0, material); emit(r1, a1, material); emit(r0, a1, material);
        }
    }
}

// A road car built from boxes: body, dark greenhouse, roof and four tyres. (cx, cz) is the centre on the ground
// at y0. The car runs along z when alongZ, otherwise along x. Length 4.4 m, width 1.84 m, top about 1.5 m above y0.
inline void appendCar(std::vector<float>& v, float cx, float y0, float cz, bool alongZ, const float* paint) {
    const float tyre[3] = {0.02f, 0.02f, 0.02f};
    const float glassC[3] = {0.05f, 0.06f, 0.07f};
    // Boxes are given in (along, across) car axes, then mapped to world x and z.
    auto box = [&](float along, float yc, float across, float hAlong, float hy, float hAcross,
                   const float* albedo, float material) {
        Box b{};
        b.c[0] = alongZ ? cx + across : cx + along;
        b.c[1] = yc;
        b.c[2] = alongZ ? cz + along : cz + across;
        b.h[0] = alongZ ? hAcross : hAlong;
        b.h[1] = hy;
        b.h[2] = alongZ ? hAlong : hAcross;
        for (int i = 0; i < 3; ++i) b.albedo[i] = albedo[i];
        b.material = material;
        appendBox(v, b);
    };
    box(0.0f, y0 + 0.62f, 0.0f, 2.2f, 0.30f, 0.92f, paint, CarPaint);
    box(-0.25f, y0 + 1.15f, 0.0f, 1.15f, 0.28f, 0.80f, glassC, CarGlass);
    box(-0.25f, y0 + 1.45f, 0.0f, 1.00f, 0.03f, 0.76f, paint, CarPaint);
    for (float sa : {-1.35f, 1.35f})
        for (float sb : {-0.98f, 0.98f})
            box(sa, y0 + 0.36f, sb, 0.36f, 0.36f, 0.14f, tyre, Asphalt);
}

inline std::vector<float> buildSceneObjects() {
    std::vector<float> v;
    const float wood[3] = {0.24f, 0.135f, 0.065f};
    const float crate[3] = {0.34f, 0.205f, 0.095f};
    const float hull[3] = {0.72f, 0.76f, 0.72f};
    const float concrete[3] = {0.58f, 0.55f, 0.48f};
    const float red[3] = {0.34f, 0.055f, 0.035f};
    const float metal[3] = {0.10f, 0.12f, 0.13f};
    const float glass[3] = {0.95f, 0.56f, 0.20f};
    const float white[3] = {0.72f, 0.70f, 0.64f};

    // The beach composition has no structures offshore; set kBuildProps to restore the pier,
    // work boat and lighthouse. Their geometry is unchanged.
    constexpr bool kBuildProps = false;
    if (kBuildProps) {
    // Weathered timber pier: individually separated deck boards, cross-beams and piles.
    for (int i = 0; i < 16; ++i) {
        const float x = -12.0f + static_cast<float>(i) * 1.5f;
        appendBox(v, {{x,3.22f,-80.0f},{0.70f,0.16f,2.85f},{wood[0],wood[1],wood[2]},Wood});
    }
    for (float x : {-10.5f,-4.5f,1.5f,7.5f}) {
        appendBox(v, {{x,2.72f,-77.2f},{0.30f,0.23f,0.25f},{wood[0],wood[1],wood[2]},Wood});
        appendBox(v, {{x,2.72f,-82.8f},{0.30f,0.23f,0.25f},{wood[0],wood[1],wood[2]},Wood});
        appendBox(v, {{x,0.55f,-77.2f},{0.25f,2.55f,0.25f},{wood[0],wood[1],wood[2]},Wood});
        appendBox(v, {{x,0.55f,-82.8f},{0.25f,2.55f,0.25f},{wood[0],wood[1],wood[2]},Wood});
    }
    appendBox(v, {{-1.0f,2.82f,-80.0f},{12.0f,0.18f,0.22f},{wood[0],wood[1],wood[2]},Wood});

    // Crate with corner battens and individual slats instead of a single featureless cube.
    appendBox(v, {{2.0f,3.88f,-80.0f},{0.95f,0.62f,0.95f},{crate[0],crate[1],crate[2]},CrateWood});
    for (float x : {1.12f,2.88f}) for (float z : {-80.88f,-79.12f})
        appendBox(v, {{x,3.88f,z},{0.07f,0.66f,0.07f},{wood[0],wood[1],wood[2]},Wood});
    for (int i = 0; i < 5; ++i) {
        const float y = 3.48f + 0.20f * i;
        appendBox(v, {{2.0f,y,-80.96f},{0.79f,0.055f,0.035f},{crate[0],crate[1],crate[2]},CrateWood});
        appendBox(v, {{2.0f,y,-79.04f},{0.79f,0.055f,0.035f},{crate[0],crate[1],crate[2]},CrateWood});
    }

    // Small work boat with a tapered hull, gunwale, cabin and mast.
    appendBoatHull(v, -20.0f, 0.15f, -60.0f, 13.0f, 2.0f, hull);
    appendBox(v, {{-20.0f,1.20f,-60.2f},{1.55f,0.12f,3.2f},{hull[0],hull[1],hull[2]},PaintedHull});
    appendBox(v, {{-20.0f,2.05f,-61.2f},{1.05f,0.78f,1.25f},{white[0],white[1],white[2]},PaintedHull});
    appendBox(v, {{-20.0f,2.95f,-61.2f},{1.10f,0.10f,1.30f},{metal[0],metal[1],metal[2]},Metal});
    appendCylinder(v, -20.0f, -57.0f, 1.25f, 5.2f, 0.075f, 0.045f, 12, metal, Metal);
    appendBox(v, {{-20.0f,4.55f,-57.0f},{0.85f,0.07f,0.07f},{metal[0],metal[1],metal[2]},Metal});

    // Lighthouse: tapered masonry tower, plinth, gallery, lantern room and roof.
    appendCylinder(v, 30.0f, -120.0f, 0.0f, 0.8f, 3.2f, 3.0f, 48, concrete, Concrete);
    appendCylinder(v, 30.0f, -120.0f, 0.8f, 24.8f, 2.85f, 1.78f, 48, white, Concrete);
    appendCylinder(v, 30.0f, -120.0f, 24.8f, 25.35f, 2.65f, 2.65f, 48, metal, Metal);
    appendCylinder(v, 30.0f, -120.0f, 25.35f, 28.8f, 1.55f, 1.55f, 40, metal, Metal);
    appendCylinder(v, 30.0f, -120.0f, 26.0f, 28.0f, 1.39f, 1.39f, 40, glass, Lantern);
    appendCylinder(v, 30.0f, -120.0f, 28.8f, 30.7f, 1.8f, 0.0f, 40, red, PaintedHull);
    // Gallery rail and evenly spaced stanchions.
    appendCylinder(v, 30.0f, -120.0f, 26.05f, 26.16f, 2.50f, 2.50f, 48, metal, Metal);
    for (int i = 0; i < 24; ++i) {
        const float a = 6.28318530718f * static_cast<float>(i) / 24.0f;
        appendCylinder(v, 30.0f + 2.48f * std::cos(a), -120.0f + 2.48f * std::sin(a),
                       26.12f, 27.05f, 0.035f, 0.035f, 8, metal, Metal);
    }
    appendCylinder(v, 30.0f, -120.0f, 27.02f, 27.10f, 2.50f, 2.50f, 48, metal, Metal);
    // Narrow seaward windows, fitted to the taper of the visible (+Z) face.
    for (float y : {7.0f,13.0f,19.0f}) {
        const float radius = 2.85f - (2.85f - 1.78f) * ((y - 0.8f) / 24.0f);
        appendBox(v, {{30.0f,y,-120.0f + radius + 0.035f},
                      {0.25f,0.72f,0.045f},{metal[0],metal[1],metal[2]},Metal});
    }
    // Warm beacon core, visible through the lantern glazing.
    appendCylinder(v, 30.0f, -120.0f, 26.55f, 27.45f, 0.34f, 0.34f, 24, glass, Lantern);

    // Irregular volcanic rock and a narrow sand band give the lighthouse a real shoreline.
    appendLighthouseIslet(v, 30.0f, -120.0f);

    }  // kBuildProps

    // Sunset skyline across the bay: towers on a deterministic jitter, set back from the far shore. Each tower
    // takes one of five facade palettes; about half carry a setback crown, and the tallest carry a mast.
    {
        const float palette[5][3] = {
            {0.60f, 0.53f, 0.44f},  // warm limestone
            {0.40f, 0.42f, 0.45f},  // grey concrete
            {0.48f, 0.29f, 0.21f},  // brick
            {0.16f, 0.19f, 0.23f},  // dark glass
            {0.70f, 0.68f, 0.63f},  // pale stone
        };
        const float mastColour[3] = {0.55f, 0.55f, 0.58f};
        for (int i = 0; i < 64; ++i) {
            const float fi = static_cast<float>(i);
            const float x = -340.0f + 10.6f * fi + 6.0f * std::sin(fi * 2.37f);
            const float depthRow = std::fabs(std::sin(fi * 3.7f));
            const float z = -430.0f - 90.0f * depthRow;
            const float half = 6.0f + 6.0f * std::fabs(std::sin(fi * 5.1f));
            const float height = 8.0f + 70.0f * std::pow(std::fabs(std::sin(fi * 78.233f)), 2.0f) * (1.0f - 0.4f * depthRow);
            const int pal = static_cast<int>(std::fabs(std::sin(fi * 2.9f + 0.7f)) * 4.999f);
            const float shade = 0.92f + 0.12f * std::fabs(std::sin(fi * 1.9f));
            const float c[3] = {palette[pal][0] * shade, palette[pal][1] * shade, palette[pal][2] * shade};
            appendBox(v, {{x, height, z}, {half, height, half}, {c[0], c[1], c[2]}, Concrete});
            const float top = 2.0f * height;  // the box spans 0 .. 2 * height
            // Three stone pilasters on the street face (+z) and a cornice band at the roofline.
            for (float k : {-0.62f, 0.0f, 0.62f})
                appendBox(v, {{x + k * half, height, z + half + 0.15f}, {0.30f, height, 0.30f},
                              {c[0] * 1.12f, c[1] * 1.12f, c[2] * 1.12f}, Stone});
            appendBox(v, {{x, top - 0.40f, z}, {half + 0.40f, 0.60f, half + 0.40f},
                          {c[0] * 1.15f, c[1] * 1.15f, c[2] * 1.15f}, Stone});
            const float crown = std::fabs(std::sin(fi * 4.3f));
            float crownTop = top;
            if (crown > 0.45f) {
                const float ch = height * (0.08f + 0.10f * crown);
                const float cw = 0.62f * half;
                appendBox(v, {{x, top + ch, z}, {cw, ch, cw}, {c[0] * 1.08f, c[1] * 1.08f, c[2] * 1.08f}, Concrete});
                crownTop = top + 2.0f * ch;
            }
            if (height > 50.0f) {
                const float mh = 10.0f + 14.0f * crown;
                appendBox(v, {{x, crownTop + mh, z}, {0.35f, mh, 0.35f},
                              {mastColour[0], mastColour[1], mastColour[2]}, Metal});
            }
            // Rooftop plant on about half the towers: a galvanised box set toward the rear corner.
            const float plant = std::fabs(std::sin(fi * 6.3f));
            if (plant > 0.5f) {
                const float pw = 1.0f + 0.8f * plant;
                const float ph = 0.8f + 1.0f * (plant - 0.5f);
                appendBox(v, {{x + 0.3f * half, crownTop + ph, z - 0.3f * half}, {pw, ph, pw},
                              {0.42f, 0.43f, 0.45f}, Metal});
            }
        }
    }
    // Quay and causeway. A paved quay with a rock seawall runs across the bay below the skyline. A four-lane causeway
    // leaves the quay toward the camera, so the road is seen in depth, where it reads; a road running across the
    // view at this camera height is squashed to under a pixel. Sidewalks top out at 1.20 m and the road at 0.90 m,
    // so the kerbs read as steps. The seawall has a gap where the causeway joins it.
    {
        const float pavers[3] = {0.46f, 0.44f, 0.40f};
        const float asphalt[3] = {0.055f, 0.055f, 0.06f};
        const float paint[3] = {0.86f, 0.85f, 0.78f};
        const float rock[3] = {0.40f, 0.38f, 0.35f};
        // Seawall in two parts, leaving x 15 .. 33 open for the causeway.
        appendBox(v, {{-172.5f, 0.0f, -394.5f}, {187.5f, 1.2f, 0.5f}, {rock[0], rock[1], rock[2]}, Rock});
        appendBox(v, {{196.5f, 0.0f, -394.5f}, {163.5f, 1.2f, 0.5f}, {rock[0], rock[1], rock[2]}, Rock});
        // Quay between the seawall and the tower fronts (z -418 .. -395).
        appendBox(v, {{0.0f, 0.45f, -406.5f}, {360.0f, 0.75f, 11.5f}, {pavers[0], pavers[1], pavers[2]}, Pavers});

        // Causeway: z -395 .. -150, x 17 .. 31 (lanes 3.5 m wide), with a sidewalk on each side.
        appendBox(v, {{24.0f, 0.40f, -272.5f}, {7.0f, 0.50f, 122.5f}, {asphalt[0], asphalt[1], asphalt[2]}, Asphalt});
        appendBox(v, {{16.0f, 0.45f, -272.5f}, {1.0f, 0.75f, 122.5f}, {pavers[0], pavers[1], pavers[2]}, Pavers});
        appendBox(v, {{32.0f, 0.45f, -272.5f}, {1.0f, 0.75f, 122.5f}, {pavers[0], pavers[1], pavers[2]}, Pavers});

        // Edge lines, a solid centre line and dashed lane dividers, 0.01 m proud of the asphalt.
        for (float x : {17.3f, 30.7f, 24.0f})
            appendBox(v, {{x, 0.91f, -272.5f}, {0.12f, 0.01f, 122.5f}, {paint[0], paint[1], paint[2]}, RoadPaint});
        for (float x : {20.5f, 27.5f})
            for (int i = 0; i < 29; ++i)
                appendBox(v, {{x, 0.91f, -390.0f + 8.0f * static_cast<float>(i) + 4.0f}, {0.09f, 0.01f, 2.0f},
                              {paint[0], paint[1], paint[2]}, RoadPaint});

        // Traffic: 24 cars in four lanes (centres x = 18.75, 22.25, 25.75, 29.25), all running along z.
        const float paints[6][3] = {
            {0.30f, 0.02f, 0.015f},  // red
            {0.03f, 0.05f, 0.12f},   // navy
            {0.55f, 0.56f, 0.58f},   // silver
            {0.02f, 0.02f, 0.025f},  // black
            {0.75f, 0.50f, 0.05f},   // taxi yellow
            {0.05f, 0.15f, 0.08f},   // dark green
        };
        for (int lane = 0; lane < 4; ++lane) {
            for (int k = 0; k < 6; ++k) {
                const float cx = 18.75f + 3.5f * static_cast<float>(lane);
                const float cz = -160.0f - 30.0f * static_cast<float>(k) - 5.0f * static_cast<float>(lane);
                appendCar(v, cx, 0.90f, cz, true, paints[(k + 2 * lane) % 6]);
            }
        }
    }

    // Ferris wheel over the bay, on a ring in the x-y plane (facing the camera): a dense rim, cabins on the
    // rim, spokes, a hub and a simple support frame. Sized so the whole ring sits inside the frame.
    {
        const float cx = -90.0f, cy = 45.0f, cz = -330.0f, R = 35.0f;
        const float rim[3] = {0.82f, 0.84f, 0.90f};
        const float cabin[3] = {0.95f, 0.45f, 0.22f};
        for (int i = 0; i < 96; ++i) {
            const float a = 6.28318530718f * static_cast<float>(i) / 96.0f;
            appendBox(v, {{cx + R * std::cos(a), cy + R * std::sin(a), cz}, {0.25f, 0.25f, 0.25f},
                          {rim[0], rim[1], rim[2]}, Metal});
        }
        for (int i = 0; i < 24; ++i) {
            const float a = 6.28318530718f * static_cast<float>(i) / 24.0f;
            appendBox(v, {{cx + R * std::cos(a), cy + R * std::sin(a), cz}, {0.9f, 1.1f, 0.8f},
                          {cabin[0], cabin[1], cabin[2]}, PaintedHull});
        }
        appendCylinder(v, cx, cz, cy - 1.5f, cy + 1.5f, 1.8f, 1.8f, 16, rim, Metal);  // hub
        for (int i = 0; i < 6; ++i) {
            const float a = 6.28318530718f * static_cast<float>(i) / 6.0f;
            appendBox(v, {{cx + 0.5f * R * std::cos(a), cy + 0.5f * R * std::sin(a), cz},
                          {0.3f, 0.3f, 0.3f}, {rim[0], rim[1], rim[2]}, Metal});
        }
        appendBox(v, {{cx - 14.0f, 0.5f * cy, cz}, {1.0f, 0.5f * cy, 1.0f}, {rim[0], rim[1], rim[2]}, Metal});
        appendBox(v, {{cx + 14.0f, 0.5f * cy, cz}, {1.0f, 0.5f * cy, 1.0f}, {rim[0], rim[1], rim[2]}, Metal});
    }

    // Palms along both shores of the bay: a trunk and three crossed frond slabs.
    const float trunk[3] = {0.36f, 0.27f, 0.18f};
    const float frond[3] = {0.12f, 0.30f, 0.12f};
    for (int i = 0; i < 14; ++i) {
        const float x = (i < 7) ? -190.0f + 20.0f * static_cast<float>(i) : 120.0f + 14.0f * static_cast<float>(i - 7);
        const float z = -170.0f - 12.0f * static_cast<float>(i % 3);
        appendCylinder(v, x, z, 0.0f, 14.0f, 0.6f, 0.35f, 8, trunk, Wood);
        appendBox(v, {{x, 14.4f, z}, {6.0f, 0.12f, 1.0f}, {frond[0], frond[1], frond[2]}, Wood});
        appendBox(v, {{x, 14.4f, z}, {1.0f, 0.12f, 6.0f}, {frond[0], frond[1], frond[2]}, Wood});
        appendBox(v, {{x, 15.5f, z}, {4.5f, 0.10f, 4.5f}, {frond[0], frond[1], frond[2]}, Wood});
    }

    // Open pavilion on the left shore: posts, a slatted roof and a deck.
    for (float x : {-110.0f, -90.0f}) for (float z : {-96.0f, -84.0f})
        appendCylinder(v, x, z, 0.0f, 5.5f, 0.35f, 0.35f, 8, white, Concrete);
    appendBox(v, {{-100.0f, 5.9f, -90.0f}, {13.0f, 0.4f, 9.0f}, {concrete[0], concrete[1], concrete[2]}, Concrete});
    appendBox(v, {{-100.0f, 0.2f, -90.0f}, {14.0f, 0.2f, 10.0f}, {wood[0], wood[1], wood[2]}, Wood});

    // Moored boats on the bay.
    appendBoatHull(v, -40.0f, 0.15f, -140.0f, 13.0f, 3.0f, hull);
    appendBoatHull(v, 60.0f, 0.15f, -190.0f, 11.0f, 2.6f, hull);
    appendBoatHull(v, -120.0f, 0.15f, -200.0f, 16.0f, 3.4f, white);
    return v;
}

}  // namespace scene
}  // namespace engine
