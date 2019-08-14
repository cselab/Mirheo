#pragma once

#include <core/analytical_shapes/api.h>
#include <core/containers.h>
#include <core/initial_conditions/rigid.h>
#include <core/initial_conditions/rod.h>
#include <core/initial_conditions/uniform.h>
#include <core/pvs/rigid_ashape_object_vector.h>
#include <core/pvs/rod_vector.h>

#include <memory>
#include <random>
#include <vector>

std::unique_ptr<ParticleVector>
initializeRandomPV(const MPI_Comm& comm, const MirState *state, float density)
{
    float mass = 1;
    auto pv = std::make_unique<ParticleVector> (state, "pv", mass);
    UniformIC ic(density);
    ic.exec(comm, pv.get(), defaultStream);
    return pv;
}

// rejection sampling for particles inside ellipsoid
static auto generateUniformEllipsoid(size_t n, float3 axes, long seed = 424242)
{
    PyTypes::VectorOfFloat3 pos;
    pos.reserve(n);

    Ellipsoid ell(axes);
    
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dx(-axes.x, axes.x);
    std::uniform_real_distribution<float> dy(-axes.y, axes.y);
    std::uniform_real_distribution<float> dz(-axes.z, axes.z);
    
    while (pos.size() < n)
    {
        float3 r {dx(gen), dy(gen), dz(gen)};
        if (ell.inOutFunction(r) < 0.f)
            pos.push_back({r.x, r.y, r.z});
    }
    return pos;
}

static auto generateObjectComQ(int n, float3 L, long seed=12345)
{
    PyTypes::VectorOfFloat7 com_q;
    com_q.reserve(n);

    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dx(0.f, L.x);
    std::uniform_real_distribution<float> dy(0.f, L.y);
    std::uniform_real_distribution<float> dz(0.f, L.z);

    for (int i = 0; i < n; ++i)
    {
        float3 r {dx(gen), dy(gen), dz(gen)};
        com_q.push_back({r.x, r.y, r.z, 1.f, 0.f, 0.f, 0.f});
    }
    
    return com_q;
}

std::unique_ptr<RigidShapedObjectVector<Ellipsoid>>
initializeRandomREV(const MPI_Comm& comm, const MirState *state, int nObjs, int objSize)
{
    float mass = 1;
    float3 axes {1.f, 1.f, 1.f};
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
    float mass = 1.f;
    float a = 0.1f;
    float L = 4.f;
    
    auto centerLine = [&](float s) -> PyTypes::float3
    {
        return {0.f, 0.f, L * (s-0.5f)};
    };

    auto torsion = [](__UNUSED float s) {return 0.f;};
    
    auto rv = std::make_unique<RodVector>
        (state, "rv", mass, numSegments);

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
        areEquals(a.q, b.q) &&
        areEquals(a.vel, b.vel) &&
        areEquals(a.omega, b.omega) &&
        areEquals(a.force, b.force) &&
        areEquals(a.torque, b.torque);
}               
