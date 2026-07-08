#include "../common.h"

#include <mirheo/core/analytical_shapes/api.h>
#include <mirheo/core/celllist.h>
#include <mirheo/core/containers.h>
#include <mirheo/core/exchangers/api.h>
#include <mirheo/core/logger.h>
#include <mirheo/core/pvs/particle_vector.h>
#include <mirheo/core/pvs/rigid_ashape_object_vector.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/utils/kernel_launch.h>

#include <cmath>
#include <cstdio>
#include <gtest/gtest.h>
#include <random>
#include <vector>

// move particles no more than rc in random direction
static void moveParticles(real rc, PinnedBuffer<real4>& pos, long seed = 80085)
{
    real frac = 0.9f;
    std::mt19937 gen(seed);
    std::uniform_real_distribution<real> distr(-frac*rc, frac*rc);

    for (auto& r : pos)
    {
        r.x += distr(gen);
        r.y += distr(gen);
        r.z += distr(gen);
    }
    pos.uploadToDevice(defaultStream);
}

// move objects no more than domainSize in random direction
static void moveObjects(real3 L, PinnedBuffer<real4>& pos,
                        PinnedBuffer<RigidMotion>& mot, long seed = 80085)
{
    real frac = 0.5f * 0.9f;
    std::mt19937 gen(seed);
    std::uniform_real_distribution<real> dx(-frac * L.x, frac * L.x);
    std::uniform_real_distribution<real> dy(-frac * L.y, frac * L.y);
    std::uniform_real_distribution<real> dz(-frac * L.z, frac * L.z);

    int objSize = pos.size() / mot.size();

    for (size_t i = 0; i < mot.size(); ++i)
    {
        real3 shift {dx(gen), dy(gen), dz(gen)};
        mot[i].r += shift;

        for (int j = 0; j < objSize; ++j)
        {
            int id = i * objSize + j;
            pos[id].x += shift.x;
            pos[id].y += shift.y;
            pos[id].z += shift.z;
        }
    }
    pos.uploadToDevice(defaultStream);
    mot.uploadToDevice(defaultStream);
}

// move rods no more than domainSize in random direction
static void moveRods(int nObjs, real3 L, PinnedBuffer<real4>& pos, long seed = 80085)
{
    real frac = 0.5f * 0.9f;
    std::mt19937 gen(seed);
    std::uniform_real_distribution<real> dx(-frac * L.x, frac * L.x);
    std::uniform_real_distribution<real> dy(-frac * L.y, frac * L.y);
    std::uniform_real_distribution<real> dz(-frac * L.z, frac * L.z);

    int objSize = pos.size() / nObjs;

    for (int i = 0; i < nObjs; ++i)
    {
        real3 shift {dx(gen), dy(gen), dz(gen)};
        for (int j = 0; j < objSize; ++j)
        {
            int id = i * objSize + j;
            pos[id].x += shift.x;
            pos[id].y += shift.y;
            pos[id].z += shift.z;
        }
    }
    pos.uploadToDevice(defaultStream);
}

inline void backToDomain(real& x, real L)
{
    if      (x < -0.5f * L) x += L;
    else if (x >= 0.5f * L) x -= L;
}

// create the reference data
// here we test 2 things:
// - correct reordering
// - correct shift
// by copying corrected positions to velocities
static void createRefParticles(const PinnedBuffer<real4>& pos,
                               PinnedBuffer<real4>& vel, real3 L)
{
    for (size_t i = 0; i < pos.size(); ++i)
    {
        auto v = pos[i];
        backToDomain(v.x, L.x);
        backToDomain(v.y, L.y);
        backToDomain(v.z, L.z);
        vel[i] = v;
    }

    vel.uploadToDevice(defaultStream);
}


// create the reference data
// here we test 2 things:
// - correct reordering
// - correct shift
// by copying corrected positions to velocities
static void createRefObjects(int objSize, const PinnedBuffer<real4>& pos,
                             PinnedBuffer<real4>& vel, real3 L)
{
    int nObj = pos.size() / objSize;

    for (int i = 0; i < nObj; ++i)
    {
        real3 com {0.f, 0.f, 0.f};

        for (int j = 0; j < objSize; ++j)
        {
            int id = i * objSize + j;
            com.x += pos[id].x;
            com.y += pos[id].y;
            com.z += pos[id].z;
        }
        com *= (1.0 / objSize);
        real3 scom = com;

        backToDomain(scom.x, L.x);
        backToDomain(scom.y, L.y);
        backToDomain(scom.z, L.z);

        real3 shift = scom - com;

        for (int j = 0; j < objSize; ++j)
        {
            int id = i * objSize + j;
            vel[id].x = pos[id].x + shift.x;
            vel[id].y = pos[id].y + shift.y;
            vel[id].z = pos[id].z + shift.z;
        }
    }

    vel.uploadToDevice(defaultStream);
}

// create the reference data
// here we test 3 things:
// - correct reordering
// - correct shift
// - correct reordering of bisegment data
// by copying corrected positions to velocities
static void createRefRods(int objSize,
                          const PinnedBuffer<real4>& pos,
                          PinnedBuffer<real4>& vel, PinnedBuffer<int64_t>& data,
                          real3 L)
{
    const int nObj = pos.size() / objSize;
    const int numSegments = (objSize - 1) / 5;
    const int numBiSegments = numSegments - 1;


    for (int i = 0; i < nObj; ++i)
    {
        real3 com {0.f, 0.f, 0.f};

        for (int j = 0; j < objSize; ++j)
        {
            int id = i * objSize + j;
            com.x += pos[id].x;
            com.y += pos[id].y;
            com.z += pos[id].z;
        }
        com *= (1.0 / objSize);
        real3 scom = com;

        backToDomain(scom.x, L.x);
        backToDomain(scom.y, L.y);
        backToDomain(scom.z, L.z);

        real3 shift = scom - com;

        for (int j = 0; j < objSize; ++j)
        {
            int id = i * objSize + j;
            vel[id].x = pos[id].x + shift.x;
            vel[id].y = pos[id].y + shift.y;
            vel[id].z = pos[id].z + shift.z;
        }

        Particle p(pos[i*objSize],
                   vel[i*objSize]);

        for (int j = 0; j < numBiSegments; ++j)
            data[i * numBiSegments + j] = p.getId();
    }

    vel.uploadToDevice(defaultStream);
    data.uploadToDevice(defaultStream);
}

static void checkInsideParticles(const PinnedBuffer<real4>& pos, real3 L)
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

static void checkRefParticles(const PinnedBuffer<real4>& pos,
                                  const PinnedBuffer<real4>& vel)
{
    for (size_t i = 0; i < pos.size(); ++i)
    {
        auto r = pos[i];
        auto v = vel[i];

        ASSERT_EQ(r.x, v.x);
        ASSERT_EQ(r.y, v.y);
        ASSERT_EQ(r.z, v.z);
    }
}

static void checkInsideObjects(int objSize, const PinnedBuffer<real4>& pos, real3 L)
{
    int nObj = pos.size() / objSize;

    for (int i = 0; i < nObj; ++i)
    {
        real3 com {0.f, 0.f, 0.f};

        for (int j = 0; j < objSize; ++j)
        {
            int id = i * objSize + j;
            com.x += pos[id].x;
            com.y += pos[id].y;
            com.z += pos[id].z;
        }
        com *= (1.0 / objSize);

        ASSERT_LT(com.x, 0.5f * L.x);
        ASSERT_LT(com.y, 0.5f * L.y);
        ASSERT_LT(com.z, 0.5f * L.z);

        ASSERT_GE(com.x, -0.5f * L.x);
        ASSERT_GE(com.y, -0.5f * L.y);
        ASSERT_GE(com.z, -0.5f * L.z);
    }
}

static void checkRefObjects(const PinnedBuffer<real4>& pos,
                            const PinnedBuffer<real4>& vel)
{
    for (size_t i = 0; i < pos.size(); ++i)
    {
        auto r = pos[i];
        auto v = vel[i];

        ASSERT_EQ(r.x, v.x);
        ASSERT_EQ(r.y, v.y);
        ASSERT_EQ(r.z, v.z);
    }
}

static void checkRefRods(int numBiSegments, int objSize,
                         const PinnedBuffer<real4>& pos,
                         const PinnedBuffer<real4>& vel,
                         const PinnedBuffer<int64_t>& data)
{
    int nObjs = data.size() / numBiSegments;

    for (size_t i = 0; i < pos.size(); ++i)
    {
        auto r = pos[i];
        auto v = vel[i];

        ASSERT_EQ(r.x, v.x);
        ASSERT_EQ(r.y, v.y);
        ASSERT_EQ(r.z, v.z);
    }

    for (int i = 0; i < nObjs; ++i)
    {
        Particle p(pos[i*objSize],
                   vel[i*objSize]);

        for (int j = 0; j < numBiSegments; ++j)
            ASSERT_EQ(data[i * numBiSegments + j], p.getId());
    }
}

TEST (PACKERS_REDISTRIBUTE, particles)
{
    real dt = 0.f;
    real rc = 1.f;
    real L  = 48.f;
    real density = 8.f;
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
    createRefParticles(pos, vel, domain.localSize);

    auto cl = std::make_unique<PrimaryCellList>(pv.get(), rc, domain.localSize);
    cl->build(defaultStream);

    auto redistr = std::make_unique<ParticleRedistributor>();
    redistr->attach(pv.get(), cl.get());

    auto engine = std::make_unique<SingleNodeExchangeEngine>(std::move(redistr));

    engine->init(defaultStream);
    engine->finalize(defaultStream);

    cl->build(defaultStream);

    pos.downloadFromDevice(defaultStream);
    vel.downloadFromDevice(defaultStream);

    checkInsideParticles(pos, domain.localSize);
    checkRefParticles(pos, vel);
}


TEST (PACKERS_REDISTRIBUTE, objects)
{
    real dt = 0.f;
    real L  = 64.f;
    int nObjs = 128;
    int objSize = 555;

    DomainInfo domain;
    domain.globalSize  = {L, L, L};
    domain.globalStart = {0.f, 0.f, 0.f};
    domain.localSize   = {L, L, L};
    MirState state(domain, dt);
    auto rev = initializeRandomREV(MPI_COMM_WORLD, &state, nObjs, objSize);
    auto lrev = rev->local();

    auto& pos = lrev->positions();
    auto& vel = lrev->velocities();
    auto& mot = *lrev->dataPerObject.getData<RigidMotion>(channel_names::motions);

    moveObjects(domain.localSize, pos, mot);
    createRefObjects(objSize, pos, vel, domain.localSize);

    auto redistr = std::make_unique<ObjectRedistributor>();
    redistr->attach(rev.get());

    auto engine = std::make_unique<SingleNodeExchangeEngine>(std::move(redistr));

    engine->init(defaultStream);
    engine->finalize(defaultStream);

    pos.downloadFromDevice(defaultStream);
    vel.downloadFromDevice(defaultStream);
    mot.downloadFromDevice(defaultStream);

    checkInsideObjects(objSize, pos, domain.localSize);
    checkRefObjects(pos, vel);
}

TEST (PACKERS_REDISTRIBUTE, rods)
{
    const std::string channelName = "my_extra_data";
    const real dt = 0.f;
    const real L  = 64.f;
    const int nObjs = 128;
    const int numSegments = 42;
    const int numBisegments = numSegments - 1;
    const int objSize = 5 * numSegments + 1;

    DomainInfo domain;
    domain.globalSize  = {L, L, L};
    domain.globalStart = {0.f, 0.f, 0.f};
    domain.localSize   = {L, L, L};
    MirState state(domain, dt);

    auto rv = initializeRandomRods(MPI_COMM_WORLD, &state, nObjs, numSegments);

    // add persistent extra data
    rv->requireDataPerBisegment<int64_t>(channelName, DataManager::PersistenceMode::Active);
    auto lrv = rv->local();
    lrv->resize(lrv->size(), defaultStream);

    auto& pos = lrv->positions();
    auto& vel = lrv->velocities();
    auto& data = *lrv->dataPerBisegment.getData<int64_t>(channelName);

    moveRods(nObjs, domain.localSize, pos);
    createRefRods(objSize, pos, vel, data, domain.localSize);

    auto redistr = std::make_unique<ObjectRedistributor>();
    redistr->attach(rv.get());

    auto engine = std::make_unique<SingleNodeExchangeEngine>(std::move(redistr));

    engine->init(defaultStream);
    engine->finalize(defaultStream);

    pos.downloadFromDevice(defaultStream);
    vel.downloadFromDevice(defaultStream);
    data.downloadFromDevice(defaultStream);

    checkInsideObjects(objSize, pos, domain.localSize);
    checkRefRods(numBisegments, objSize, pos, vel, data);
}


int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    logger.init(MPI_COMM_WORLD, "packers_redistribute.log", 9);

    testing::InitGoogleTest(&argc, argv);
    auto retval = RUN_ALL_TESTS();

    MPI_Finalize();
    return retval;
}
