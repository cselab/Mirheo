#include "../common.h"

#include <core/analytical_shapes/api.h>
#include <core/celllist.h>
#include <core/containers.h>
#include <core/exchangers/api.h>
#include <core/logger.h>
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

// move particles no more than rc in random direction
void moveParticles(float rc, PinnedBuffer<float4>& pos, long seed = 80085)
{
    float frac = 0.9f;
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> distr(-frac*rc, frac*rc);

    for (auto& r : pos)
    {
        r.x += distr(gen);
        r.y += distr(gen);
        r.z += distr(gen);
    }
    pos.uploadToDevice(defaultStream);
}

void checkInside(const PinnedBuffer<float4>& pos, float3 L)
{
    for (const auto& r : pos)
    {
        ASSERT_LT(r.x, 0.5f * L.x);
        ASSERT_LT(r.y, 0.5f * L.y);
        ASSERT_LT(r.z, 0.5f * L.z);

        ASSERT_GE(r.x, -0.5f * L.x);
        ASSERT_GE(r.y, -0.5f * L.y);
        ASSERT_GE(r.z, -0.5f * L.z);
    }
}

TEST (PACKERS_REDISTRIBUTE, particles)
{
    float dt = 0.f;
    float rc = 1.f;
    float L  = 8.f;
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

    moveParticles(rc, pos);

    int n = lpv->size();

    auto cl = std::make_unique<PrimaryCellList>(pv.get(), rc, domain.localSize);
    cl->build(defaultStream);
    
    auto redistr = std::make_unique<ParticleRedistributor>();
    redistr->attach(pv.get(), cl.get());

    auto engine = std::make_unique<SingleNodeEngine>(std::move(redistr));

    engine->init(defaultStream);
    engine->finalize(defaultStream);

    cl->build(defaultStream);

    pos.downloadFromDevice(defaultStream);
    vel.downloadFromDevice(defaultStream);

    checkInside(pos, domain.localSize);
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
