#pragma once

#include <mirheo/core/analytical_shapes/api.h>
#include <mirheo/core/containers.h>
#include <mirheo/core/initial_conditions/rigid.h>
#include <mirheo/core/initial_conditions/rod.h>
#include <mirheo/core/initial_conditions/uniform.h>
#include <mirheo/core/pvs/rigid_ashape_object_vector.h>
#include <mirheo/core/pvs/rod_vector.h>

#include <memory>
#include <random>
#include <vector>

using namespace mirheo;

std::unique_ptr<ParticleVector>
initializeRandomPV(const MPI_Comm& comm, const MirState *state, real density)
{
    real mass = 1;
    auto pv = std::make_unique<ParticleVector> (state, "pv", mass);
    UniformIC ic(density);
    ic.exec(comm, pv.get(), defaultStream);
    return pv;
}

// rejection sampling for particles inside ellipsoid
static auto generateUniformEllipsoid(size_t n, real3 axes, long seed = 424242)
{
    std::vector<real3> pos;
    pos.reserve(n);

    Ellipsoid ell(axes);

    std::mt19937 gen(seed);
    std::uniform_real_distribution<real> dx(-axes.x, axes.x);
    std::uniform_real_distribution<real> dy(-axes.y, axes.y);
    std::uniform_real_distribution<real> dz(-axes.z, axes.z);

    while (pos.size() < n)
    {
        const real3 r {dx(gen), dy(gen), dz(gen)};
        if (ell.inOutFunction(r) < 0._r)
            pos.push_back(r);
    }
    return pos;
}

static auto generateObjectComQ(int n, real3 L, long seed=12345)
{
    std::vector<ComQ> com_q;
    com_q.reserve(n);

    std::mt19937 gen(seed);
    std::uniform_real_distribution<real> dx(0._r, L.x);
    std::uniform_real_distribution<real> dy(0._r, L.y);
    std::uniform_real_distribution<real> dz(0._r, L.z);

    for (int i = 0; i < n; ++i)
    {
        const real3 r {dx(gen), dy(gen), dz(gen)};
        const real4 q {1._r, 0._r, 0._r, 0._r};
        com_q.push_back({r, q});
    }

    return com_q;
}

std::unique_ptr<RigidShapedObjectVector<Ellipsoid>>
initializeRandomREV(const MPI_Comm& comm, const MirState *state, int nObjs, int objSize)
{
    real mass = 1;
    real3 axes {1._r, 1._r, 1._r};
    Ellipsoid ellipsoid(axes);

    auto rev = std::make_unique<RigidShapedObjectVector<Ellipsoid>>
        (state, "rev", mass, objSize, ellipsoid);

    auto com_q  = generateObjectComQ(nObjs, state->domain.globalSize);
    auto coords = generateUniformEllipsoid(objSize, axes);

    RigidIC ic(com_q, coords);
    ic.exec(comm, rev.get(), defaultStream);

    return rev;
}

std::unique_ptr<RodVector>
initializeRandomRods(const MPI_Comm& comm, const MirState *state, int nObjs, int numSegments)
{
    real mass = 1._r;
    real a = 0.1_r;
    real L = 4._r;

    auto centerLine = [&](real s)
    {
        return real3 {0._r, 0._r, L * (s-0.5_r)};
    };

    auto torsion = [](__UNUSED real s) {return 0._r;};

    auto rv = std::make_unique<RodVector> (state, "rv", mass, numSegments);

    auto com_q  = generateObjectComQ(nObjs, state->domain.globalSize);

    RodIC ic(com_q, centerLine, torsion, a);
    ic.exec(comm, rv.get(), defaultStream);

    return rv;
}

inline bool areEquals(float3 a, float3 b)
{
    return
        a.x == b.x &&
        a.y == b.y &&
        a.z == b.z;
}

inline bool areEquals(float4 a, float4 b)
{
    return
        a.x == b.x &&
        a.y == b.y &&
        a.z == b.z &&
        a.w == b.w;
}

inline bool areEquals(double3 a, double3 b)
{
    return
        a.x == b.x &&
        a.y == b.y &&
        a.z == b.z;
}

inline bool areEquals(double4 a, double4 b)
{
    return
        a.x == b.x &&
        a.y == b.y &&
        a.z == b.z &&
        a.w == b.w;
}

inline bool areEquals(RigidMotion a, RigidMotion b)
{
    return
        areEquals(a.r, b.r) &&
        areEquals(static_cast<RigidReal4>(a.q), static_cast<RigidReal4>(b.q)) &&
        areEquals(a.vel, b.vel) &&
        areEquals(a.omega, b.omega) &&
        areEquals(a.force, b.force) &&
        areEquals(a.torque, b.torque);
}
