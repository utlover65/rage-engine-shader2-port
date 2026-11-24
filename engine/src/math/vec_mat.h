// Minimal vector and matrix math for the engine host code. Matrices are column-major
// (m[column * 4 + row]), matching GLSL.
#pragma once

#include <array>
#include <cmath>
#include <utility>

namespace engine {
namespace math {

struct Vec3 {
    double x = 0, y = 0, z = 0;
};
inline Vec3 operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline Vec3 operator*(Vec3 a, double s) { return {a.x * s, a.y * s, a.z * s}; }
inline double dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline Vec3 cross(Vec3 a, Vec3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
inline Vec3 normalize(Vec3 a) {
    const double l = std::sqrt(dot(a, a));
    return a * (1.0 / l);
}

// Column-major 4x4 matrix in double precision; converted to float for the GPU.
struct Mat4 {
    std::array<double, 16> m{};
    static Mat4 identity() {
        Mat4 r;
        r.m[0] = r.m[5] = r.m[10] = r.m[15] = 1.0;
        return r;
    }
    double& at(int row, int col) { return m[col * 4 + row]; }
    double at(int row, int col) const { return m[col * 4 + row]; }
};

inline Mat4 operator*(const Mat4& a, const Mat4& b) {
    Mat4 r;
    for (int c = 0; c < 4; c++)
        for (int rr = 0; rr < 4; rr++) {
            double s = 0;
            for (int k = 0; k < 4; k++) s += a.at(rr, k) * b.at(k, c);
            r.at(rr, c) = s;
        }
    return r;
}

// View matrix looking from eye toward target (right-handed, y up).
inline Mat4 lookAt(Vec3 eye, Vec3 target, Vec3 up) {
    Vec3 f = normalize(target - eye);
    Vec3 s = normalize(cross(f, up));
    Vec3 u = cross(s, f);
    Mat4 r = Mat4::identity();
    r.at(0, 0) = s.x; r.at(0, 1) = s.y; r.at(0, 2) = s.z;
    r.at(1, 0) = u.x; r.at(1, 1) = u.y; r.at(1, 2) = u.z;
    r.at(2, 0) = -f.x; r.at(2, 1) = -f.y; r.at(2, 2) = -f.z;
    r.at(0, 3) = -dot(s, eye);
    r.at(1, 3) = -dot(u, eye);
    r.at(2, 3) = dot(f, eye);
    return r;
}

// Vulkan clip conventions: depth 0..1, and y flipped so row 0 of the framebuffer is the top
// of the image (NDC y = -1 is the top).
// Orthographic projection for Vulkan: x and y as usual, depth mapped to [0, 1] with near at 0.
inline Mat4 orthoVulkan(double l, double r, double b, double t, double n, double f) {
    Mat4 m;
    for (int i = 0; i < 16; i++) m.m[i] = 0.0;
    m.m[0] = 2.0 / (r - l);
    m.m[5] = 2.0 / (t - b);
    // View space looks down -z, so depth d = -z_view and z_ndc = (d - n) / (f - n).
    m.m[10] = -1.0 / (f - n);
    m.m[12] = -(r + l) / (r - l);
    m.m[13] = -(t + b) / (t - b);
    m.m[14] = -n / (f - n);
    m.m[15] = 1.0;
    return m;
}

inline Mat4 perspectiveVulkan(double fovYRad, double aspect, double zNear, double zFar) {
    const double f = 1.0 / std::tan(fovYRad * 0.5);
    Mat4 r;
    r.at(0, 0) = f / aspect;
    r.at(1, 1) = -f;
    r.at(2, 2) = zFar / (zNear - zFar);
    r.at(2, 3) = (zFar * zNear) / (zNear - zFar);
    r.at(3, 2) = -1.0;
    return r;
}

// General 4x4 inverse by Gauss-Jordan elimination with partial pivoting.
inline Mat4 inverse(const Mat4& in) {
    double a[4][8];
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++) {
            a[r][c] = in.at(r, c);
            a[r][c + 4] = (r == c) ? 1.0 : 0.0;
        }
    for (int col = 0; col < 4; col++) {
        int piv = col;
        for (int r = col + 1; r < 4; r++)
            if (std::fabs(a[r][col]) > std::fabs(a[piv][col])) piv = r;
        for (int c = 0; c < 8; c++) std::swap(a[col][c], a[piv][c]);
        const double d = a[col][col];
        for (int c = 0; c < 8; c++) a[col][c] /= d;
        for (int r = 0; r < 4; r++) {
            if (r == col) continue;
            const double f = a[r][col];
            for (int c = 0; c < 8; c++) a[r][c] -= f * a[col][c];
        }
    }
    Mat4 out;
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++) out.at(r, c) = a[r][c + 4];
    return out;
}

}  // namespace math
}  // namespace engine
