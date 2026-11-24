// Headless driving test for physics/vehicle.h: settle, launch, brake, corner. Exit 0 on pass.
#include <cmath>
#include <cstdio>

#include "physics/vehicle.h"

using engine::vehicle::Vehicle;
using engine::vehicle::Vec;

static bool finiteState(const Vehicle& c) {
    return std::isfinite(c.p.x) && std::isfinite(c.p.y) && std::isfinite(c.p.z) && std::isfinite(c.v.x) &&
           std::isfinite(c.v.z) && std::isfinite(c.w.x) && std::isfinite(c.w.y) && std::isfinite(c.w.z) &&
           std::isfinite(c.q.w) && std::isfinite(c.q.x) && std::isfinite(c.q.y) && std::isfinite(c.q.z);
}

int main() {
    Vehicle car;
    const float dt = 0.001f;
    bool ok = true;

    // A: static settle. Each corner should carry about a quarter of the weight (3434 N each).
    for (int i = 0; i < 3000; i++) car.step(dt);
    float compAvg = 0.0f, loadAvg = 0.0f;
    for (int i = 0; i < 4; i++) { compAvg += car.comp[i] * 0.25f; loadAvg += car.load[i] * 0.25f; }
    const float expectedLoad = car.mass * Vehicle::kGravity * 0.25f;
    std::printf("settle: comp %.3f m, load %.0f N (weight/4 %.0f N), speed %.3f m/s, height %.3f m\n", compAvg,
                loadAvg, expectedLoad, car.speed(), car.p.y);
    if (!(std::fabs(loadAvg - expectedLoad) < 0.02f * expectedLoad && car.speed() < 0.05f && car.p.y > 0.35f)) {
        std::printf("FAIL: static settle\n");
        ok = false;
    }

    // B: full throttle from rest, timed to 100 km/h. A 1400 kg car with about 280 Nm peak torque
    // reaches it in roughly 6.5 to 8 s on dry tarmac.
    car.throttle = 1.0f;
    float t100 = -1.0f;
    for (int i = 0; i < 10000; i++) {
        car.step(dt);
        if (t100 < 0.0f && car.forwardSpeed() >= 27.78f) t100 = (i + 1) * dt;
    }
    std::printf("launch: 0-100 km/h in %.2f s, gear %d, rpm %.0f\n", t100, car.gear, car.rpm);
    if (!(t100 > 6.5f && t100 < 8.0f && finiteState(car))) { std::printf("FAIL: launch time\n"); ok = false; }

    // C: brake from the current speed to a stop.
    car.throttle = 0.0f;
    car.brake = 1.0f;
    const Vec start = car.p;
    const float v0 = car.forwardSpeed();
    float t = 0.0f;
    while (car.speed() > 0.2f && t < 60.0f) { car.step(dt); t += dt; }
    const float stopDist = length(car.p - start);
    std::printf("brake: from %.1f m/s to stop in %.1f m (%.2f s)\n", v0, stopDist, t);
    // Friction-limited stop on dry tarmac: d = v^2 / (2 mu g) with mu = 1, allowing for weight transfer
    // and aero drag. Real cars land close to this.
    const float ideal = v0 * v0 / (2.0f * 9.81f);
    std::printf("  ideal v^2/(2 mu g) = %.1f m\n", ideal);
    if (!(stopDist > 0.9f * ideal && stopDist < 1.15f * ideal && finiteState(car))) { std::printf("FAIL: braking\n"); ok = false; }
    car.brake = 0.0f;

    // D: cornering. Build speed, then hold full steering with light throttle.
    car.throttle = 0.4f;
    for (int i = 0; i < 6000 && car.forwardSpeed() < 15.0f; i++) car.step(dt);
    car.steer = 1.0f;
    float peakYaw = 0.0f;
    for (int i = 0; i < 2000; i++) {
        car.step(dt);
        peakYaw = std::max(peakYaw, std::fabs(car.w.y));
    }
    std::printf("corner: speed %.1f m/s, peak yaw rate %.2f rad/s, finite %s\n", car.forwardSpeed(), peakYaw,
                finiteState(car) ? "yes" : "no");
    // Lateral acceleration v * yaw rate cannot exceed the friction limit mu g.
    const float latAcc = car.forwardSpeed() * peakYaw;
    std::printf("  lateral acceleration %.2f m/s^2 (limit %.2f)\n", latAcc, 9.81f);
    if (!(peakYaw > 0.2f && latAcc < 1.05f * 9.81f && finiteState(car))) { std::printf("FAIL: cornering\n"); ok = false; }

    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
