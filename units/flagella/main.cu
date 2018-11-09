#include <vector>
#include <core/logger.h>
#include <core/utils/helper_math.h>
#include <plugins/utils/xyz.h>

Logger logger;

struct Params
{
    float Ke, Kb, theta0, a, gamma;
};

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

static void addBondForces(const Params& p, int n, const float3 *positions, float3 *forces)
{
    const float ledge = p.a / sqrt(2);
    const float ldiag = p.a;

    auto bond = [&](int offset, int i, int j, float leq) {
        i += offset;
        j += offset;
        auto f = spring(p.Ke, leq, positions[i], positions[j]);
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
        bond(offset, 1, 3, ldiag);
        bond(offset, 2, 4, ldiag);
    }
}

inline float3 visc(float g, float3 r1, float3 r2, float3 v1, float3 v2)
{
    float3 dr = r2 - r1;
    float3 dv = v2 - v1;
    float l = length(dr);    
    return dr * (g * dot(dv, dr) / dot(dr, dr));
}

static void addBondDissipativeForce(const Params& p, int n, const float3 *positions, const float3 *velocities, float3 *forces)
{
    auto bond = [&](int offset, int i, int j) {
        i += offset;
        j += offset;
        auto f = visc(p.gamma, positions[i], positions[j], velocities[i], velocities[j]);
        forces[i] += f;
        forces[j] -= f;
    };
    
    for (int i = 0; i < n; ++i) {
        int offset = 5 * i;

        bond(offset, 0, 1);
        bond(offset, 0, 2);
        bond(offset, 0, 3);
        bond(offset, 0, 4);

        bond(offset, 5, 1);
        bond(offset, 5, 2);
        bond(offset, 5, 3);
        bond(offset, 5, 4);

        bond(offset, 0, 1);
        bond(offset, 1, 2);
        bond(offset, 2, 3);
        bond(offset, 3, 4);

        bond(offset, 0, 5);
        bond(offset, 1, 3);
        bond(offset, 2, 4);
    }    
}

inline float3 normalised(float3 a)
{
    float l = length(a);
    if (l > 1e-6f) return a / l;
    return make_float3(0, 0, 0);
}

static void addBendingForces(const Params& p, int n, const float3 *positions, float3 *forces)
{
    for (int i = 0; i < n - 1; ++i) {
        float3 r0 = positions[5*(i + 0)];
        float3 r1 = positions[5*(i + 1)];
        float3 r2 = positions[5*(i + 2)];

        float3 d0 = r1 - r0;
        float3 d1 = r2 - r1;

        float d0_inv = 1.0 / length(d0);
        float d1_inv = 1.0 / length(d1);

        d0 *= d0_inv;
        d1 *= d1_inv;

        float dot01 = dot(d0, d1);
        if (dot01 >  1) dot01 =  1;
        if (dot01 < -1) dot01 = -1;
        float theta = acos(dot01);

        float3 f01 = normalised(d0 * dot01 - d1) * d0_inv;
        float3 f12 = normalised(d1 * dot01 - d0) * d1_inv;

        float factor = -(p.theta0 - theta) * p.Kb;

        forces[5 * (i + 0)] += factor * f01;
        forces[5 * (i + 1)] += factor * (f12 - f01);
        forces[5 * (i + 2)] -= factor * f12;
    }
}

static void addTorsionForces(const Params& p, int n, const float3 *positions, float3 *forces)
{
    
}

static void forcesFlagellum(const Params& p, int n, const float3 *positions, const float3 *velocities, float3 *forces)
{
    addBondForces(p, n, positions, forces);
    addBendingForces(p, n, positions, forces);
    addTorsionForces(p, n, positions, forces);
    addBondDissipativeForce(p, n, positions, velocities, forces);
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

    Params p;
    p.a = 0.1;
    p.gamma = 10;
    p.Ke = 60;
    p.Kb = 1.0;
    p.theta0 = 0;

    float dt = 0.01;
    int nsteps = 20000;
    int dumpEvery = 100;

    std::vector<float3> positions, velocities, forces;
    
    initialFlagellum(p.a, n, positions);
    velocities.resize(positions.size());
    forces    .resize(positions.size());
    clear(velocities);

    for (auto& v : velocities) {
        v = 0.1 * make_float3(drand48() - 0.5f,
                              drand48() - 0.5f,
                              drand48() - 0.5f);
    }
    
    for (int i = 0; i < nsteps; ++i) {
        clear(forces);
        forcesFlagellum(p, n, positions.data(), velocities.data(), forces.data());
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
