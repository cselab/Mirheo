#include <mirheo/core/logger.h>
#include <mirheo/core/utils/helper_math.h>
#include <mirheo/plugins/utils/pid.h>

#include <cmath>
#include <cstdio>
#include <gtest/gtest.h>

using namespace mirheo;

TEST (PID, PidEquilibrates)
{
    float target, Kp, Ki, Kd;
    float state, dt;
    int step_time, nsteps;

    const float target_start = 0.f;
    const float target_end   = 1.f;
    const float tolerance = 1e-5;

    state = target = 0.f;
    step_time = 20;
    nsteps = 200;

    dt = 0.1;
    Kp = 3.f;
    Ki = 2.f;
    Kd = 3.f;

    PidControl<float> pid(target-state, Kp, Ki, Kd);

    for (int i = 0; i < nsteps; ++i) {
        target = i < step_time ? target_start : target_end;
        state += dt * pid.update(target-state);
        // printf("%g\n", state);
    }

    ASSERT_LE(math::abs(state - target_end), tolerance);
}

int main(int argc, char **argv)
{
    logger.init(MPI_COMM_NULL, "pid.log", 0);
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
