// Headless crash test for physics/softcar.h: 54 km/h into a rigid wall. Exit code 0 on pass.
#include <cstdio>

#include "physics/softcar.h"

using engine::physics::SoftCar;
using engine::physics::V3;

int main() {
    SoftCar car;
    const float x0 = 6.0f, ride = 0.3f, v0 = -15.0f;  // 54 km/h towards the wall
    car.build(x0, ride, v0);
    const float kin0 = car.kinetic();
    const float sep0 = car.rearX() - car.frontX();  // rear-to-front separation: shortens when the nose crushes
    const float frameDt = 1.0f / 60.0f;
    const int substeps = 10, iterations = 4;
    const float h = frameDt / substeps;
    bool ok = true;
    std::printf("t(s)   v_cm(m/s)  KE(kJ)   front_x(m)  rear_x(m)  broken\n");
    for (int frame = 0; frame <= 120; frame++) {
        if (frame % 12 == 0) {
            const V3 v = car.velocityMean();
            std::printf("%5.2f  %8.2f  %7.1f  %9.3f  %9.3f  %6d\n", frame * frameDt, v.x, car.kinetic() / 1000.0f,
                        car.frontX(), car.rearX(), car.countBroken());
        }
        if (!car.finite()) { std::printf("FAIL: non-finite positions at frame %d\n", frame); return 1; }
        if (frame == 120) break;
        for (int s = 0; s < substeps; s++) car.substep(h, iterations);
        car.material(frameDt);
    }
    const float crumple = sep0 - (car.rearX() - car.frontX());
    std::printf("crumple (front-to-rear shortening): %.3f m, springs torn: %d of %zu\n", crumple, car.countBroken(),
                car.spring.size());
    if (!(car.kinetic() < kin0)) { std::printf("FAIL: kinetic energy did not decrease\n"); ok = false; }
    // Sanity bound only: the model crumples but is not calibrated to crash data (real front crush is ~0.4 m+).
    if (!(crumple > 0.15f)) { std::printf("FAIL: front face did not crumple\n"); ok = false; }
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
