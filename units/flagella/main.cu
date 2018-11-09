#include <vector>
#include <core/logger.h>
#include <core/utils/helper_math.h>
#include <plugins/utils/xyz.h>

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

inline float3 spring(float K, float leq, float3 r1, float3 r2)
{
    float3 dr = r2 - r1;
    float l = length(dr);
    return dr * (K * (l - leq) / l);
}

static void addBondForces(float a, float K, int n, const float3 *positions, float3 *forces)
{
    const float ledge = a / sqrt(2);
    const float ldiag = a;

    auto bond = [&](int offset, int i, int j, float leq) {
        i += offset;
        j += offset;
        auto f = spring(K, leq, positions[i], positions[j]);
        forces[i] += f;
        forces[j] -= f;
    };
    
    for (int i = 0; i < n; ++i) {
        int offset = 5 * i;

        bond(offset, 0, 1, ledge);
        bond(offset, 0, 2, ledge);
        bond(offset, 0, 3, ledge);
        bond(offset, 0, 4, ledge);

        bond(offset, 5, 1, ledge);
        bond(offset, 5, 2, ledge);
        bond(offset, 5, 3, ledge);
        bond(offset, 5, 4, ledge);

        bond(offset, 0, 1, ledge);
        bond(offset, 1, 2, ledge);
        bond(offset, 2, 3, ledge);
        bond(offset, 3, 4, ledge);

        bond(offset, 0, 5, ldiag);
        bond(offset, 0, 2, ldiag);
        bond(offset, 1, 3, ldiag);
    }
}

static void addTorsionForces(float a, float K, int n, const float3 *positions, float3 *forces)
{
    
}

static void forcesFlagellum(float a, float K, int n, const float3 *positions, float3 *forces)
{
    addBondForces(a, K, n, positions, forces);
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

static void dump(MPI_Comm comm, int id, int np, const float3 *positions)
{
    std::vector<Particle> particles(np);
    for (int i = 0; i < np; ++i) {
        Particle p;
        float3 r = positions[i];
        p.r.x = r.x;
        p.r.y = r.y;
        p.r.z = r.z;
        particles[i] = p;
    }
    std::string tstr = std::to_string(id);
    std::string name = "flagella_" + std::string(5 - tstr.length(), '0') + tstr + ".xyz";

    writeXYZ(comm, name, particles.data(), particles.size());
}

static void run(MPI_Comm comm)
{
    int n = 30;
    float a = 0.1;
    float K = 1.0;
    float dt = 0.01;
    int nsteps = 20;
    int dumpEvery = 1;

    std::vector<float3> positions, velocities, forces;
    
    initialFlagellum(a, n, positions);
    velocities.resize(positions.size());
    forces    .resize(positions.size());
    clear(velocities);
    
    for (int i = 0; i < nsteps; ++i) {
        clear(forces);
        forcesFlagellum(a, K, n, positions.data(), forces.data());
        advanceFlagellum(dt, n, forces.data(), positions.data(), velocities.data());

        if (i % dumpEvery == 0) {
            int id = i / dumpEvery;
            dump(comm, id, positions.size(), positions.data());
        }
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    run(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}
