// Road-vehicle dynamics: a 6-DOF rigid chassis (quaternion orientation) on four hydraulic
// suspension corners, with simplified Pacejka tyres, wheel spin, a torque-curve engine and a
// five-speed automatic, rear-wheel drive. Header-only. SI units: m, s, kg, N, Nm.
// Body axes: X right, Y up, Z forward. World axes: Y up. Ground: y = 0.
//
// Model limits (stated, not calibrated): no anti-roll bar, no Ackermann, no clutch slip, a single
// friction coefficient, no tyre temperature, no load-sensitive mu. Corner dampers are a fitted
// bump/rebound linear + quadratic curve, not a measured valve map.
#pragma once

#include <algorithm>
#include <cmath>

namespace engine {
namespace vehicle {

struct Vec {
    float x = 0, y = 0, z = 0;
};
inline Vec operator+(Vec a, Vec b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline Vec operator-(Vec a, Vec b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline Vec operator*(Vec a, float s) { return {a.x * s, a.y * s, a.z * s}; }
inline float dot(Vec a, Vec b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline Vec cross(Vec a, Vec b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
inline float length(Vec a) { return std::sqrt(dot(a, a)); }

struct Quat {
    float w = 1, x = 0, y = 0, z = 0;
};
inline Quat mul(Quat a, Quat b) {
    return {a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
            a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
            a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
            a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w};
}
inline Quat conj(Quat q) { return {q.w, -q.x, -q.y, -q.z}; }
inline Vec rotate(Quat q, Vec v) {
    const Vec u{q.x, q.y, q.z};
    const Vec t = cross(u, v) * 2.0f;
    return v + t * q.w + cross(u, t);
}
inline void normalize(Quat& q) {
    const float n = std::sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
    q.w /= n; q.x /= n; q.y /= n; q.z /= n;
}

// Magic-formula shape: F(x) = sin(C atan(Bx - E (Bx - atan(Bx)))), normalised peak 1 (times D outside).
inline float magic(float x, float B, float C, float E) {
    const float bx = B * x;
    return std::sin(C * std::atan(bx - E * (bx - std::atan(bx))));
}

struct Vehicle {
    // Chassis
    float mass = 1400.0f;
    float Ixx = 2500.0f;  // pitch inertia (about body X)
    float Iyy = 2800.0f;  // yaw inertia (about body Y)
    float Izz = 600.0f;   // roll inertia (about body Z)
    static constexpr float kGravity = 9.81f;
    float aeroCdA = 0.32f * 2.2f;  // Cd x frontal area (m^2)
    float airDensity = 1.2f;
    float rollingCoef = 0.015f;

    // Suspension (per corner): linear spring, hydraulic damper (bump and rebound, linear + quadratic),
    // and a bump stop past 0.18 m of compression.
    float restLen = 0.25f;           // strut length at zero load (m), attachment to tyre contact
    float springK = 35000.0f;        // N/m
    float bumpStopK = 3.0e5f;
    float bumpStopStart = 0.18f;
    float dampBumpLin = 3000.0f;     // N s/m, compression
    float dampBumpQuad = 800.0f;     // N s^2/m^2, compression
    float dampReboundLin = 4500.0f;  // N s/m, extension
    float dampReboundQuad = 1200.0f;

    // Tyres and wheels
    float wheelR = 0.33f;
    float wheelI = 1.2f;   // kg m^2 per wheel and tyre
    float mu = 1.0f;       // friction coefficient (dry tarmac)
    float brakeTorque = 2400.0f;  // Nm per wheel at full brake
    float maxSteer = 0.45f;       // rad at full steering input (front wheels)

    // Drivetrain: torque curve, five gears, automatic shifts, rear-wheel drive
    float tMax = 280.0f;           // Nm at peak (tuned: 0-100 km/h in about 6.6 s for 1400 kg)
    float tPeakRpm = 4500.0f;
    float idleRpm = 800.0f, shiftUpRpm = 6500.0f, shiftDownRpm = 2500.0f, redlineRpm = 7000.0f;
    float gearRatio[5] = {3.5f, 2.2f, 1.5f, 1.1f, 0.9f};
    float finalDrive = 3.7f;
    float driveEff = 0.92f;

    // Hard points (body-local, relative to the COM): front left, front right, rear left, rear right
    Vec attach[4] = {{-0.8f, 0, 1.4f}, {0.8f, 0, 1.4f}, {-0.8f, 0, -1.4f}, {0.8f, 0, -1.4f}};

    // State
    Vec p{0, 0.48f, 0};  // COM, world (m); starts at static ride height
    Vec v{0, 0, 0};      // COM velocity, world (m/s)
    Vec w{0, 0, 0};      // angular velocity, body frame (rad/s)
    Quat q;              // body to world
    float omega[4] = {0, 0, 0, 0};  // wheel spin (rad/s)
    float comp[4] = {0, 0, 0, 0};   // suspension compression (m)
    float load[4] = {0, 0, 0, 0};   // normal load per tyre (N)
    int gear = 1;                   // 1..5
    float rpm = 800.0f;
    float engineTorque = 0.0f;

    // Driver inputs in [0, 1] (throttle, brake) and [-1, 1] (steer, positive right)
    float throttle = 0, brake = 0, steer = 0;

    float speed() const { return length(v); }
    float forwardSpeed() const { return dot(v, rotate(q, Vec{0, 0, 1})); }

    float engineTorqueAt(float r) const {
        const float x = (r - tPeakRpm) / (redlineRpm - idleRpm);
        return tMax * (1.0f - 0.35f * x * x);
    }

    void driverShift() {
        if (rpm > shiftUpRpm && gear < 5) gear++;
        else if (rpm < shiftDownRpm && gear > 1) gear--;
    }

    void step(float dt) {
        // Engine state from the previous wheel speeds (rear axle average).
        const float rearSpin = 0.5f * (omega[2] + omega[3]);
        rpm = std::max(idleRpm, std::fabs(rearSpin) * gearRatio[gear - 1] * finalDrive * 60.0f / (2.0f * 3.14159265f));
        driverShift();
        rpm = std::max(idleRpm, std::fabs(rearSpin) * gearRatio[gear - 1] * finalDrive * 60.0f / (2.0f * 3.14159265f));
        engineTorque = (rpm < redlineRpm) ? throttle * engineTorqueAt(rpm) : 0.0f;

        // Heading (yaw only) for the wheel axes.
        const Vec fwd3 = rotate(q, Vec{0, 0, 1});
        const float fl = std::sqrt(fwd3.x * fwd3.x + fwd3.z * fwd3.z);
        const Vec fwdH{fwd3.x / fl, 0.0f, fwd3.z / fl};
        const Vec rightH{fwdH.z, 0.0f, -fwdH.x};
        const Vec wWorld = rotate(q, w);

        Vec F{0, 0, 0}, T{0, 0, 0};
        const Vec worldUp{0, 1, 0};

        for (int i = 0; i < 4; i++) {
            const Vec r = rotate(q, attach[i]);
            const Vec aw = p + r;                  // attachment, world
            const Vec va = v + cross(wWorld, r);   // attachment velocity, world
            // Suspension: spring, bump stop and hydraulic damper. No tension, so the tyre can lift.
            const float c = restLen - (aw.y - wheelR);
            const float cr = -va.y;  // compression rate (m/s)
            float Fs = 0.0f;
            if (c > 0.0f) {
                Fs = springK * c;
                if (c > bumpStopStart) Fs += bumpStopK * (c - bumpStopStart);
                if (cr > 0.0f) Fs += dampBumpLin * cr + dampBumpQuad * cr * cr;
                else Fs -= dampReboundLin * (-cr) + dampReboundQuad * cr * cr;
            }
            comp[i] = c;
            const float N = std::max(Fs, 0.0f);
            load[i] = N;
            F = F + worldUp * N;
            T = T + cross(aw - p, worldUp * N);

            // Tyre: contact point directly below the hard point; steering on the front axle.
            const Vec cp{aw.x, 0.0f, aw.z};
            const float delta = (i < 2) ? steer * maxSteer : 0.0f;
            const float cd = std::cos(delta), sd = std::sin(delta);
            const Vec f = fwdH * cd + rightH * sd;   // wheel heading
            const Vec l = rightH * cd - fwdH * sd;   // wheel lateral (right)
            const Vec vc = v + cross(wWorld, cp - p);
            const float vLong = dot(vc, f);
            const float vLat = dot(vc, l);
            const float absLong = std::fabs(vLong);

            float Fx = 0.0f, Fy = 0.0f;
            if (N > 0.0f) {
                const float D = mu * N;
                const float alpha = std::atan2(vLat, std::max(absLong, 0.5f));
                const float kappa = (omega[i] * wheelR - vLong) / std::max(absLong, 1.0f);
                Fy = -D * magic(alpha, 8.0f, 1.3f, 0.97f);
                Fx = D * magic(kappa, 10.0f, 1.65f, 0.97f);
                // Friction circle: the combined force cannot exceed mu N.
                const float mag = std::sqrt(Fx * Fx + Fy * Fy);
                if (mag > D) { Fx *= D / mag; Fy *= D / mag; }
            }

            // Wheel spin: drive (rear axle), brake (all), reaction from the tyre force.
            const float Td = (i >= 2) ? engineTorque * gearRatio[gear - 1] * finalDrive * driveEff * 0.5f : 0.0f;
            const float dirSpin = (omega[i] > 0.0f) ? 1.0f : (omega[i] < 0.0f ? -1.0f : 0.0f);
            const float Tb = brake * brakeTorque * dirSpin;
            float wNew = omega[i] + dt * (Td - Tb - Fx * wheelR) / wheelI;
            if (brake > 0.0f && omega[i] * wNew < 0.0f) wNew = 0.0f;  // brakes stop a wheel, they do not reverse it
            omega[i] = wNew;

            const Vec Ft = f * Fx + l * Fy;
            F = F + Ft;
            T = T + cross(cp - p, Ft);
        }

        // Aerodynamic drag and rolling resistance on the COM.
        const float sp = speed();
        if (sp > 0.0f) F = F - v * (0.5f * airDensity * aeroCdA * sp);
        const Vec vH{v.x, 0.0f, v.z};
        const float vHl = length(vH);
        if (vHl > 0.1f) F = F - vH * (rollingCoef * mass * kGravity / vHl);
        F = F - worldUp * (mass * kGravity);

        // Rigid body: translation, then rotation (body-frame Euler equations, diagonal inertia).
        v = v + F * (dt / mass);
        p = p + v * dt;
        const Vec Tb_body = rotate(conj(q), T);
        const Vec Iw{Ixx * w.x, Iyy * w.y, Izz * w.z};
        const Vec gyro = cross(w, Iw);
        w.x += dt * (Tb_body.x - gyro.x) / Ixx;
        w.y += dt * (Tb_body.y - gyro.y) / Iyy;
        w.z += dt * (Tb_body.z - gyro.z) / Izz;
        const Quat qd = mul(q, Quat{0.0f, w.x, w.y, w.z});
        q.w += 0.5f * dt * qd.w;
        q.x += 0.5f * dt * qd.x;
        q.y += 0.5f * dt * qd.y;
        q.z += 0.5f * dt * qd.z;
        normalize(q);
    }
};

}  // namespace vehicle
}  // namespace engine
