#include "../common.h"

#include <mirheo/core/analytical_shapes/api.h>
#include <mirheo/core/containers.h>
#include <mirheo/core/logger.h>
#include <mirheo/core/pvs/packers/objects.h>
#include <mirheo/core/pvs/particle_vector.h>
#include <mirheo/core/pvs/rigid_ashape_object_vector.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/utils/kernel_launch.h>

#include <cmath>
#include <cstdio>
#include <gtest/gtest.h>
#include <random>
#include <vector>

__global__ void packParticlesIdentityMap(int n, ParticlePackerHandler packer, char *buffer)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;

    int srcId = i;
    int dstId = i;
    
    packer.particles.pack(srcId, dstId, buffer, n);
}

__global__ void packShiftParticlesIdentityMap(int n, real3 shift, ParticlePackerHandler packer, char *buffer)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;

    int srcId = i;
    int dstId = i;
    
    packer.particles.packShift(srcId, dstId, buffer, n, shift);
}

__global__ void unpackParticlesIdentityMap(int n, const char *buffer, ParticlePackerHandler packer)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;

    int srcId = i;
    int dstId = i;
    
    packer.particles.unpack(srcId, dstId, buffer, n);
}


__global__ void packObjectsIdentityMap(int nObjects, ObjectPackerHandler packer, char *buffer)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int nParticles = nObjects * packer.objSize;

    int srcId = i;
    int dstId = i;

    // assume nParticles > nObjects
    // so that buffer is updated for conserned threads

    if (i < nParticles)
        buffer += packer.particles.pack(srcId, dstId, buffer, nParticles);

    if ( i < nObjects)
        packer.objects.pack(srcId, dstId, buffer, nObjects);
}

__global__ void unpackObjectsIdentityMap(int nObjects, const char *buffer, ObjectPackerHandler packer)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int nParticles = nObjects * packer.objSize;

    int srcId = i;
    int dstId = i;

    // assume nParticles > nObjects
    // so that buffer is updated for conserned threads

    if (i < nParticles)
        buffer += packer.particles.unpack(srcId, dstId, buffer, nParticles);

    if (i < nObjects)
        packer.objects.unpack(srcId, dstId, buffer, nObjects);
}

TEST (PACKERS_SIMPLE, particles)
{
    real dt = 0.f;
    real L = 8.f;
    real density = 4.f;
    DomainInfo domain;
    domain.globalSize  = {L, L, L};
    domain.globalStart = {0.f, 0.f, 0.f};
    domain.localSize   = {L, L, L};
    MirState state(domain, dt);
    auto pv = initializeRandomPV(MPI_COMM_WORLD, &state, density);
    auto lpv = pv->local();

    auto& pos = lpv->positions();
    auto& vel = lpv->velocities();
    
    const std::vector<real4> pos_cpy(pos.begin(), pos.end());
    const std::vector<real4> vel_cpy(vel.begin(), vel.end());

    int n = lpv->size();

    PackPredicate predicate = [](const DataManager::NamedChannelDesc&) {return true;};
    ParticlePacker packer(predicate);
    packer.update(lpv, defaultStream);
    
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

TEST (PACKERS_SIMPLE, particlesShift)
{
    real dt = 0.f;
    real L = 8.f;
    real density = 4.f;
    real3 shift {42.0f, 43.0f, 44.0f};
    DomainInfo domain;
    domain.globalSize  = {L, L, L};
    domain.globalStart = {0.f, 0.f, 0.f};
    domain.localSize   = {L, L, L};
    MirState state(domain, dt);
    auto pv = initializeRandomPV(MPI_COMM_WORLD, &state, density);
    auto lpv = pv->local();

    auto& pos = lpv->positions();
    auto& vel = lpv->velocities();
    
    std::vector<real4> pos_cpy(pos.begin(), pos.end());
    std::vector<real4> vel_cpy(vel.begin(), vel.end());

    for (auto& r : pos_cpy)
    {
        r.x += shift.x;
        r.y += shift.y;
        r.z += shift.z;
    }
    
    int n = lpv->size();

    PackPredicate predicate = [](const DataManager::NamedChannelDesc&) {return true;};
    ParticlePacker packer(predicate);
    packer.update(lpv, defaultStream);
    
    size_t sizeBuff = packer.getSizeBytes(n);
    DeviceBuffer<char> buffer(sizeBuff);

    const int nthreads = 128;
    const int nblocks  = getNblocks(n, nthreads);

    SAFE_KERNEL_LAUNCH(
        packShiftParticlesIdentityMap,
        nblocks, nthreads, 0, defaultStream,
        n, shift, packer.handler(), buffer.devPtr());

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
    real dt = 0.f;
    real L   = 64.f;
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

    const std::vector<real4> pos_cpy(pos.begin(), pos.end());
    const std::vector<real4> vel_cpy(vel.begin(), vel.end());
    const std::vector<RigidMotion> mot_cpy(mot->begin(), mot->end());

    int n = lrev->size();

    PackPredicate predicate = [](const DataManager::NamedChannelDesc&) {return true;};
    ObjectPacker packer(predicate);
    packer.update(lrev, defaultStream);
    
    size_t sizeBuff = packer.getSizeBytes(nObjs);
    DeviceBuffer<char> buffer(sizeBuff);

    const int nthreads = 128;
    const int nblocks  = getNblocks(n, nthreads);

    SAFE_KERNEL_LAUNCH(
        packObjectsIdentityMap,
        nblocks, nthreads, 0, defaultStream,
        nObjs, packer.handler(), buffer.devPtr());

    // make sure we actually copy back the stuff
    pos .clearDevice(defaultStream);
    vel .clearDevice(defaultStream);
    mot->clearDevice(defaultStream);
    
    SAFE_KERNEL_LAUNCH(
        unpackObjectsIdentityMap,
        nblocks, nthreads, 0, defaultStream,
        nObjs, buffer.devPtr(), packer.handler());

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
