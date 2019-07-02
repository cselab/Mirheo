#include <core/containers.h>
#include <core/initial_conditions/uniform.h>
#include <core/logger.h>
#include <core/pvs/data_packers/data_packers.h>
#include <core/pvs/particle_vector.h>
#include <core/utils/cuda_common.h>
#include <core/utils/kernel_launch.h>

#include <cmath>
#include <cstdio>
#include <gtest/gtest.h>
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


std::unique_ptr<ParticleVector>
initializeRandom(const MPI_Comm& comm, const MirState *state, float density)
{
    float mass = 1;
    auto pv = std::make_unique<ParticleVector> (state, "pv", mass);
    UniformIC ic(density);
    ic.exec(comm, pv.get(), defaultStream);
    return pv;
}
inline bool areEquals(float4 a, float4 b)
{
    return
        a.x == b.x &&
        a.y == b.y &&
        a.z == b.z &&
        a.w == b.w;
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
    auto pv = initializeRandom(MPI_COMM_WORLD, &state, density);
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

    SAFE_KERNEL_LAUNCH(
        unpackParticlesIdentityMap,
        nblocks, nthreads, 0, defaultStream,
        n, buffer.devPtr(), packer.handler());

    pos.downloadFromDevice(defaultStream);
    vel.downloadFromDevice(defaultStream);

    for (int i = 0; i < n; ++i)
    {
        ASSERT_TRUE(areEquals(pos[i], pos_cpy[i]));
        ASSERT_TRUE(areEquals(vel[i], vel_cpy[i]));
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
