#include <cstdio>
#include "pid.h"

int main() {
    float target, Kp, Ki, Kd;
    float state, dt;
    int step_time, nsteps;

    state = target = 0.f;
    step_time = 20;
    nsteps = 200;

    dt = 0.1;
    Kp = 3.f;
    Ki = 2.f;
    Kd = 3.f;
    
    PidControl<float> pid(target-state, Kp, Ki, Kd);

    for (int i = 0; i < nsteps; ++i) {
        target = i < step_time ? 0.f : 1.f;
        state += dt * pid.update(target-state);
        printf("%g\n", state);
    }
    
    return 0;
}
