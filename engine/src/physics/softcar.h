// Soft-body car crash model (see README, pass 21): a 3-D lattice of point masses joined by
// compliant distance springs, solved with XPBD. Springs yield plastically and tear; internal motion is
// damped. Header-only so it can be called from game code without a new library target. Units m, s, kg.
#pragma once

#include <cmath>
#include <vector>

namespace engine {
namespace physics {

struct V3 { float x, y, z; };
inline V3 add(V3 a, V3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline V3 sub(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline V3 mul(V3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
inline float dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline float len(V3 a) { return std::sqrt(dot(a, a)); }

struct Node { V3 p, prev, v; float invMass; };
struct Spring { int a, b; float rest; float alpha; bool broken; };

struct SoftCar {
    // Lattice size and cell spacing (m): about 4.5 m long, 1.0 m high, 1.6 m wide.
    int NX = 10, NY = 3, NZ = 5;
    float dx = 0.5f, dy = 0.5f, dz = 0.4f;
    float mass = 1200.0f;
    // Material: strain past which a spring yields, rate of permanent set, strain at which it tears.
    float yieldStrain = 0.01f, plasticRate = 25.0f, breakStrain = 0.25f;
    float restitution = 0.15f, friction = 0.4f;
    // Compliance (m/N) of the crumple zone (the two front columns) and of the rest of the body.
    float frontCompliance = 1.0f / 2.0e4f, bodyCompliance = 1.0f / 5.0e6f;
    float internalDamping = 2.0f;  // 1/s, decay of deviations from the mean velocity
    std::vector<Node> node;
    std::vector<Spring> spring;
    std::vector<float> lambda;  // XPBD multipliers, reset each substep

    int idx(int i, int j, int k) const { return (k * NY + j) * NX + i; }

    void build(float x0, float rideHeight, float vx0) {
        node.assign(static_cast<size_t>(NX * NY * NZ), Node{});
        const float nodeMass = mass / static_cast<float>(node.size());
        for (int k = 0; k < NZ; k++)
            for (int j = 0; j < NY; j++)
                for (int i = 0; i < NX; i++) {
                    Node& n = node[idx(i, j, k)];
                    n.p = {x0 + i * dx, rideHeight + j * dy, (k - (NZ - 1) * 0.5f) * dz};
                    n.prev = n.p;
                    n.v = {vx0, 0.0f, 0.0f};
                    n.invMass = 1.0f / nodeMass;
                }
        // Thirteen forward neighbours give axial, face-diagonal and body-diagonal links (shear included).
        const int off[13][3] = {{1,0,0},{0,1,0},{0,0,1},{1,1,0},{1,-1,0},{0,1,1},{0,1,-1},
                                {1,0,1},{1,0,-1},{1,1,1},{1,1,-1},{1,-1,1},{1,-1,-1}};
        for (int k = 0; k < NZ; k++)
            for (int j = 0; j < NY; j++)
                for (int i = 0; i < NX; i++)
                    for (const auto& o : off) {
                        const int ii = i + o[0], jj = j + o[1], kk = k + o[2];
                        if (ii < 0 || ii >= NX || jj < 0 || jj >= NY || kk < 0 || kk >= NZ) continue;
                        const int a = idx(i, j, k), b = idx(ii, jj, kk);
                        const bool front = (i <= 1) && (ii <= 1);
                        spring.push_back({a, b, len(sub(node[b].p, node[a].p)),
                                          front ? frontCompliance : bodyCompliance, false});
                    }
    }

    // One XPBD substep: predict, solve compliant distance constraints, project contacts, update velocity.
    void substep(float h, int iterations) {
        const V3 g{0.0f, -9.81f, 0.0f};
        for (Node& n : node) {
            n.v = add(n.v, mul(g, h));
            n.prev = n.p;
            n.p = add(n.p, mul(n.v, h));
        }
        lambda.assign(spring.size(), 0.0f);
        for (int it = 0; it < iterations; it++) {
            for (size_t si = 0; si < spring.size(); si++) {
                const Spring& s = spring[si];
                if (s.broken) continue;
                Node& A = node[s.a];
                Node& B = node[s.b];
                const float wsum = A.invMass + B.invMass;
                const V3 d = sub(B.p, A.p);
                const float L = len(d);
                if (L < 1e-6f) continue;
                const V3 n = mul(d, 1.0f / L);
                const float C = L - s.rest;
                const float alphaTilde = s.alpha / (h * h);
                const float dl = (-C - alphaTilde * lambda[si]) / (wsum + alphaTilde);
                lambda[si] += dl;
                A.p = sub(A.p, mul(n, A.invMass * dl));
                B.p = add(B.p, mul(n, B.invMass * dl));
            }
        }
        for (Node& n : node) {
            // Ground at y = 0.
            if (n.p.y < 0.0f) {
                n.p.y = 0.0f;
                if (n.v.y < 0.0f) n.v.y = -restitution * n.v.y;
                n.v.x *= (1.0f - friction * 0.2f);
                n.v.z *= (1.0f - friction * 0.2f);
            }
            // Rigid wall at x = 0 (the car approaches from +x).
            if (n.p.x < 0.0f) {
                n.p.x = 0.0f;
                if (n.v.x < 0.0f) n.v.x = -restitution * n.v.x;
                n.v.y *= (1.0f - friction * 0.2f);
                n.v.z *= (1.0f - friction * 0.2f);
            }
            n.v = mul(sub(n.p, n.prev), 1.0f / h);
        }
        // Internal damping: deviations from the mean velocity decay; rigid translation is untouched.
        const V3 vm = velocityMean();
        const float k = std::exp(-internalDamping * h);
        for (Node& n : node) n.v = add(vm, mul(sub(n.v, vm), k));
    }

    // Once per frame: permanent set past the yield strain, tearing past the break strain.
    void material(float frameDt) {
        for (Spring& s : spring) {
            if (s.broken) continue;
            const float L = len(sub(node[s.b].p, node[s.a].p));
            const float strain = (L - s.rest) / s.rest;
            if (std::fabs(strain) > breakStrain) { s.broken = true; continue; }
            if (std::fabs(strain) > yieldStrain) {
                const float k = 1.0f - std::exp(-plasticRate * frameDt);
                s.rest += (L - s.rest) * k;
            }
        }
    }

    float kinetic() const {
        float e = 0.0f;
        for (const Node& n : node) e += 0.5f / n.invMass * dot(n.v, n.v);
        return e;
    }
    float frontX() const {  // mean x of the front face (i = 0)
        float x = 0.0f;
        for (int k = 0; k < NZ; k++) for (int j = 0; j < NY; j++) x += node[idx(0, j, k)].p.x;
        return x / static_cast<float>(NZ * NY);
    }
    float rearX() const {  // mean x of the rear face (i = NX - 1)
        float x = 0.0f;
        for (int k = 0; k < NZ; k++) for (int j = 0; j < NY; j++) x += node[idx(NX - 1, j, k)].p.x;
        return x / static_cast<float>(NZ * NY);
    }
    V3 centre() const {
        V3 c{0, 0, 0};
        for (const Node& n : node) c = add(c, n.p);
        return mul(c, 1.0f / static_cast<float>(node.size()));
    }
    V3 velocityMean() const {
        V3 v{0, 0, 0};
        for (const Node& n : node) v = add(v, n.v);
        return mul(v, 1.0f / static_cast<float>(node.size()));
    }
    int countBroken() const {
        int c = 0;
        for (const Spring& s : spring) c += s.broken ? 1 : 0;
        return c;
    }
    bool finite() const {
        for (const Node& n : node)
            if (!std::isfinite(n.p.x) || !std::isfinite(n.p.y) || !std::isfinite(n.p.z)) return false;
        return true;
    }
};


}  // namespace physics
}  // namespace engine
