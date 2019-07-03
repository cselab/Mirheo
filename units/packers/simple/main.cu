#include <core/analytical_shapes/api.h>
#include <core/containers.h>
#include <core/initial_conditions/rigid.h>
#include <core/initial_conditions/uniform.h>
#include <core/logger.h>
#include <core/pvs/packers/packers.h>
#include <core/pvs/particle_vector.h>
#include <core/pvs/rigid_ashape_object_vector.h>
#include <core/utils/cuda_common.h>
#include <core/utils/kernel_launch.h>

#include <cmath>
#include <cstdio>
#include <gtest/gtest.h>
#include <random>
#include <vector>

Logger logger;

__global__ void packParticlesIdentityMap(int n, ParticlePackerHandler packer, char *buffer)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;

    int srcId = i;
    int dstId = i;
    
    packer.particles.pack(srcId, dstId, buffer, n);
}

__global__ void unpackParticlesIdentityMap(int n, const char *buffer, ParticlePackerHandler packer)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;

    int srcId = i;
    int dstId = i;
    
    packer.particles.unpack(srcId, dstId, buffer, n);
}


__global__ void packObjectsIdentityMap(int nObjects, int objSize, ObjectPackerHandler packer, char *buffer)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int nParticles = nObjects * objSize;

    int srcId = i;
    int dstId = i;

    // assume nParticles > nObjects
    // so that buffer is updated for conserned threads

    if (i < nParticles)
        buffer += packer.particles.pack(srcId, dstId, buffer, nParticles);

    if ( i < nObjects)
        packer.objects.pack(srcId, dstId, buffer, nObjects);
}

__global__ void unpackObjectsIdentityMap(int nObjects, int objSize, const char *buffer, ObjectPackerHandler packer)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int nParticles = nObjects * objSize;

    int srcId = i;
    int dstId = i;

    // assume nParticles > nObjects
    // so that buffer is updated for conserned threads

    if (i < nParticles)
        buffer += packer.particles.unpack(srcId, dstId, buffer, nParticles);

    if (i < nObjects)
        packer.objects.unpack(srcId, dstId, buffer, nObjects);
}


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
auto generateUniformEllipsoid(int n, float3 axes, long seed = 424242)
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

auto generateObjectComQ(int n, float3 L, long seed=12345)
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

TEST (PACKERS_SIMPLE, particles)
{
    float dt = 0.f;
    float L = 8.f;
    float density = 4.f;
    DomainInfo domain;
    domain.globalSize  = {L, L, L};
    domain.globalStart = {0.f, 0.f, 0.f};
    domain.localSize   = {L, L, L};
    MirState state(domain, dt);
    auto pv = initializeRandomPV(MPI_COMM_WORLD, &state, density);
    auto lpv = pv->local();

    auto& pos = lpv->positions();
    auto& vel = lpv->velocities();
    
    const std::vector<float4> pos_cpy(pos.begin(), pos.end());
    const std::vector<float4> vel_cpy(vel.begin(), vel.end());

    int n = lpv->size();

    ParticlePacker packer;
    PackPredicate predicate = [](const DataManager::NamedChannelDesc&) {return true;};
    packer.update(lpv, predicate, defaultStream);
    
    size_t sizeBuff = packer.getSizeBytes(n);
    DeviceBuffer<char> buffer(sizeBuff);

    const int nthreads = 128;
    const int nblocks  = getNblocks(n, nthreads);

    SAFE_KERNEL_LAUNCH(
        packParticlesIdentityMap,
        nblocks, nthreads, 0, defaultStream,
        n, packer.handler(), buffer.devPtr());

    // make sure we actually copy back the stuff
    pos.clearDevice(defaultStream);
    vel.clearDevice(defaultStream);

    SAFE_KERNEL_LAUNCH(
        unpackParticlesIdentityMap,
        nblocks, nthreads, 0, defaultStream,
        n, buffer.devPtr(), packer.handler());

    pos.downloadFromDevice(defaultStream);
    vel.downloadFromDevice(defaultStream);

    for (int i = 0; i < n; ++i)
    {
        ASSERT_TRUE(areEquals(pos[i], pos_cpy[i])) << "failed for position with id " << i;
        ASSERT_TRUE(areEquals(vel[i], vel_cpy[i])) << "failed for velocity with id " << i;
    }
}

TEST (PACKERS_SIMPLE, objects)
{
    float dt = 0.f;
    float L   = 64.f;
    int nObjs = 128;
    int objSize = 666;

    DomainInfo domain;
    domain.globalSize  = {L, L, L};
    domain.globalStart = {0.f, 0.f, 0.f};
    domain.localSize   = {L, L, L};
    MirState state(domain, dt);
    auto rev = initializeRandomREV(MPI_COMM_WORLD, &state, nObjs, objSize);
    auto lrev = rev->local();

    auto& pos = lrev->positions();
    auto& vel = lrev->velocities();
    auto  mot = lrev->dataPerObject.getData<RigidMotion>(ChannelNames::motions);

    pos .downloadFromDevice(defaultStream);
    vel .downloadFromDevice(defaultStream);
    mot->downloadFromDevice(defaultStream);

    const std::vector<float4> pos_cpy(pos.begin(), pos.end());
    const std::vector<float4> vel_cpy(vel.begin(), vel.end());
    const std::vector<RigidMotion> mot_cpy(mot->begin(), mot->end());

    int n = lrev->size();

    ObjectPacker packer;
    PackPredicate predicate = [](const DataManager::NamedChannelDesc&) {return true;};
    packer.update(lrev, predicate, defaultStream);
    
    size_t sizeBuff = packer.getSizeBytes(nObjs, objSize);
    DeviceBuffer<char> buffer(sizeBuff);

    const int nthreads = 128;
    const int nblocks  = getNblocks(n, nthreads);

    SAFE_KERNEL_LAUNCH(
        packObjectsIdentityMap,
        nblocks, nthreads, 0, defaultStream,
        nObjs, objSize, packer.handler(), buffer.devPtr());

    // make sure we actually copy back the stuff
    pos .clearDevice(defaultStream);
    vel .clearDevice(defaultStream);
    mot->clearDevice(defaultStream);
    
    SAFE_KERNEL_LAUNCH(
        unpackObjectsIdentityMap,
        nblocks, nthreads, 0, defaultStream,
        nObjs, objSize, buffer.devPtr(), packer.handler());

    pos .downloadFromDevice(defaultStream);
    vel .downloadFromDevice(defaultStream);
    mot->downloadFromDevice(defaultStream);

    for (int i = 0; i < n; ++i)
    {
        ASSERT_TRUE(areEquals(pos[i], pos_cpy[i])) << "failed for position with id " << i;
        ASSERT_TRUE(areEquals(vel[i], vel_cpy[i])) << "failed for velocity with id " << i;
    }

    for (int i = 0; i < nObjs; ++i)
    {
        ASSERT_TRUE(areEquals((*mot)[i], mot_cpy[i])) << "failed for motion with id " << i;
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    logger.init(MPI_COMM_WORLD, "packers_simple.log", 3);    

    testing::InitGoogleTest(&argc, argv);
    auto retval = RUN_ALL_TESTS();

    MPI_Finalize();
    return retval;
}
