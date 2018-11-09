#include <vector>
#include <core/logger.h>
#include <core/utils/helper_math.h>

Logger logger;

static void clear(std::vector<float3>& vect)
{
    for (auto& v : vect) v = make_float3(0, 0, 0);
}

static void initialFlagellum(float a, int n, std::vector<float3>& positions)
{
    positions.resize(5 * n + 1);

    for (int i = 0; i < n; ++i) {
        float3 r{i * a, 0, 0};
        float3 r0{(i + 0.5f) * a,   0.5f * a, 0};
        float3 r1{(i + 0.5f) * a, 0,   0.5f * a};
        float3 r2{(i + 0.5f) * a, - 0.5f * a, 0};
        float3 r3{(i + 0.5f) * a, 0, - 0.5f * a};

        positions[i * 5 + 0] = r;
        positions[i * 5 + 1] = r0;
        positions[i * 5 + 2] = r1;
        positions[i * 5 + 3] = r2;
        positions[i * 5 + 4] = r3;
    }

    positions[5*n] = make_float3(n * a, 0, 0);
}

static void addBoundForces(float a, float K, int n, const float3 *positions, float3 *forces)
{
    
}

static void addTorsionForces(float a, float K, int n, const float3 *positions, float3 *forces)
{
    
}

static void forcesFlagellum(float a, float K, int n, const float3 *positions, float3 *forces)
{
    addBoundForces(a, K, n, positions, forces);
    addTorsionForces(a, K, n, positions, forces);
}

static void advanceFlagellum(float dt, int n, const float3 *forces, float3 *positions, float3 *velocities)
{
    // assume m = 1
    int np = 5 * n + 1;
    for (int i = 0; i < np; ++i) {
        velocities[i] += dt * forces[i];
        positions[i] += dt * velocities[i];
    }
}


static void run()
{
    int n = 4;
    float a = 0.1;
    float K = 1.0;
    float dt = 0.01;
    int nsteps = 20;

    std::vector<float3> positions, velocities, forces;
    
    initialFlagellum(a, n, positions);
    velocities.resize(positions.size());
    forces    .resize(positions.size());
    clear(velocities);
    
    for (int i = 0; i < nsteps; ++i) {
        clear(forces);
        forcesFlagellum(a, K, n, positions.data(), forces.data());
        advanceFlagellum(dt, n, forces.data(), positions.data(), velocities.data());
    }
}

int main(int argc, char **argv)
{
    run();
    return 0;
}
