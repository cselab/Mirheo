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

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <gtest/gtest.h>
#include <random>
#include <vector>

Logger logger;

// create the reference data
// will let us test 2 things:
// - correct reordering
// - correct shift
// by copying corrected positions to velocities
static void createRef(const PinnedBuffer<float4>& pos, PinnedBuffer<float4>& vel)
{
    for (int i = 0; i < pos.size(); ++i)
        vel[i] = pos[i];
    
    vel.uploadToDevice(defaultStream);
}

// check that ref pos and vel are exactly L apart
static void checkRef(const PinnedBuffer<float4>& pos,
                     const PinnedBuffer<float4>& vel,
                     float3 L)
{
    constexpr float eps = 1e-6f;
    for (int i = 0; i < pos.size(); ++i)
    {
        auto r = make_float3(pos[i]);
        auto v = make_float3(vel[i]);
        auto e = fabs(r-v) - L;

        auto minErr = std::min(e.x, std::min(e.y, e.z));

        ASSERT_LE(minErr, eps);
    }
}

struct Comp // for sorting particles
{
    bool inline operator()(float4 a_, float4 b_) const
    {
        Float3_int a(a_), b(b_);
        return
            (a.i < b.i) ||
            (a.i == b.i && a.v.x  < b.v.x) ||
            (a.i == b.i && a.v.x == b.v.x && a.v.y  < b.v.y) ||
            (a.i == b.i && a.v.x == b.v.x && a.v.y == b.v.y && a.v.z == b.v.z);
    } 
};

static void checkHalo(const PinnedBuffer<float4>& lpos,
                      const PinnedBuffer<float4>& hpos,
                      float3 L, float rc)
{
    std::vector<float4> hposSorted(hpos.begin(), hpos.end());
    std::sort(hposSorted.begin(), hposSorted.end(), Comp());

    for (int i = 0; i < lpos.size(); ++i)
    {
        const auto r0 = lpos[i];

        int dx = -1 + (r0.x >= -0.5f * L.x + rc) + (r0.x >= 0.5f * L.x - rc);
        int dy = -1 + (r0.y >= -0.5f * L.y + rc) + (r0.y >= 0.5f * L.y - rc);
        int dz = -1 + (r0.z >= -0.5f * L.z + rc) + (r0.z >= 0.5f * L.z - rc);
        
        for (int iz = min(dz,0); iz < max(dz,0); ++iz)
        for (int iy = min(dy,0); iy < max(dy,0); ++iy)
        for (int ix = min(dx,0); ix < max(dx,0); ++ix)
        {
            if (ix == 0 && iy == 0 && iz == 0) continue;

            float4 r = {r0.x - ix * L.x,
                        r0.y - iy * L.y,
                        r0.z - iz * L.z,
                        r0.w};

            auto less = [](float4 a_, float4 b_)
            {
                Float3_int a(a_), b(b_);
                return a.i < b.i;
            };

            auto lessEq = [](float4 a_, float4 b_)
            {
                Float3_int a(a_), b(b_);
                return a.i <= b.i;
            };

            auto lb = std::lower_bound(hposSorted.begin(),
                                       hposSorted.end(),
                                       r, less);

            float err = 1e9f;
            
            while (lb != hposSorted.end() && lessEq(*lb, r))
            {
                float errx = fabs(lb->x - r.x);
                float erry = fabs(lb->y - r.y);
                float errz = fabs(lb->z - r.z);
                float curr = std::min(errx, std::min(erry, errz));
                err = std::min(err, curr);
                ++lb;
            };
            
            ASSERT_LE(err, 1e-6f);
        }
    }
}

TEST (PACKERS_EXCHANGE, particles)
{
    float dt = 0.f;
    float rc = 1.f;
    float L  = 48.f;
    float density = 8.f;
    DomainInfo domain;
    domain.globalSize  = {L, L, L};
    domain.globalStart = {0.f, 0.f, 0.f};
    domain.localSize   = {L, L, L};
    MirState state(domain, dt);
    auto pv = initializeRandomPV(MPI_COMM_WORLD, &state, density);
    auto lpv = pv->local();
    auto hpv = pv->halo();

    auto& lpos = lpv->positions();
    auto& lvel = lpv->velocities();

    createRef(lpos, lvel);    

    int n = lpv->size();

    auto cl = std::make_unique<PrimaryCellList>(pv.get(), rc, domain.localSize);
    cl->build(defaultStream);
    
    auto exch = std::make_unique<ParticleHaloExchanger>();
    exch->attach(pv.get(), cl.get(), {});

    auto engine = std::make_unique<SingleNodeEngine>(std::move(exch));

    engine->init(defaultStream);
    engine->finalize(defaultStream);

    lpos.downloadFromDevice(defaultStream);
    lvel.downloadFromDevice(defaultStream);

    auto& hpos = hpv->positions();
    auto& hvel = hpv->velocities();

    hpos.downloadFromDevice(defaultStream);
    hvel.downloadFromDevice(defaultStream);

    checkHalo(lpos, hpos, domain.localSize, rc);
    checkRef(hpos, hvel, domain.localSize);
}

/*
 * tests for object exchange:
 * apply a field to the object particles in 2 manners:
 * - periodic, local one
 * - local ; exchange + halo
 * then compare some unique reduced variable
 */

template <class ForceCont>
static void clearForces(ForceCont& forces)
{
    for (auto& f : forces)
        f.f = make_float3(0.f);
}

inline bool isInside(float3 r, float3 L)
{
    return
        (r.x >= - 0.5f * L.x && r.x < 0.5f * L.x) &&
        (r.y >= - 0.5f * L.y && r.y < 0.5f * L.y) &&
        (r.z >= - 0.5f * L.z && r.z < 0.5f * L.z);
}

inline Force getField(float3 r, float3 L)
{
    Force f;
    f.f = length(r) * r;
    return f;
}

template <class ForceCont>
static void applyFieldLocal(const PinnedBuffer<float4>& pos,
                            ForceCont& force, float3 L)
{
    for (size_t i = 0; i < pos.size(); ++i)
    {
        auto r = make_float3(pos[i]);
        if (isInside(r, L))
            force[i] += getField(r, L);
    }
}

template <class ForceCont>
static void applyFieldPeriodic(const PinnedBuffer<float4>& pos,
                               ForceCont& forces, float3 L)
{
    for (size_t i = 0; i < pos.size(); ++i)
    {
        const auto r0 = make_float3(pos[i]);

        for (int ix = -1; ix < 2; ++ix)
        for (int iy = -1; iy < 2; ++iy)
        for (int iz = -1; iz < 2; ++iz)
        {
            float3 r {r0.x + ix * L.x,
                      r0.y + iy * L.y,
                      r0.z + iz * L.z};
        
            if (isInside(r, L))
                forces[i] += getField(r, L);
        }
    }
}

template <class ForceCont>
static void applyFieldUnbounded(const PinnedBuffer<float4>& pos,
                               ForceCont& forces, float3 L)
{
    for (size_t i = 0; i < pos.size(); ++i)
    {
        const auto r = make_float3(pos[i]);
        forces[i] += getField(r, L);
    }
}


inline float linf(float3 a, float3 b)
{
    auto d = fabs(a-b);
    return std::min(d.x, std::min(d.y, d.z));
}

static void checkForces(const PinnedBuffer<float4>& pos,
                        const PinnedBuffer<Force>& forces,
                        float3 L)
{
    for (size_t i = 0; i < pos.size(); ++i)
    {
        const auto r0 = make_float3(pos[i]);
        const auto f0 = forces[i].f;
        float err = 1e9f;

        for (int ix = -1; ix < 2; ++ix)
        for (int iy = -1; iy < 2; ++iy)
        for (int iz = -1; iz < 2; ++iz)
        {
            if (ix == 0 && iy == 0 && iz == 0) continue;

            float3 r {r0.x + ix * L.x,
                      r0.y + iy * L.y,
                      r0.z + iz * L.z};
            auto f = getField(r, L).f;
            
            err = std::min(err, linf(f0, f));
        }
        ASSERT_LE(err, 1e-6f);
    }
}

static void compareForces(const PinnedBuffer<Force>& forcesA,
                          const std::vector<Force>& forcesB)
{
    for (size_t i = 0; i < forcesA.size(); ++i)
    {
        auto fA = forcesA[i].f;
        auto fB = forcesB[i].f;
        
        auto err = linf(fA, fB);
        ASSERT_LE(err, 1e-6f);

        // ASSERT_TRUE(areEquals(fA, fB));
    }
}



TEST (PACKERS_EXCHANGE, objects_exchange)
{
    float dt = 0.f;
    float rc = 1.f;
    float L  = 48.f;
    int nObjs = 1024;
    int objSize = 555;

    DomainInfo domain;
    domain.globalSize  = {L, L, L};
    domain.globalStart = {0.f, 0.f, 0.f};
    domain.localSize   = {L, L, L};
    MirState state(domain, dt);
    auto rev = initializeRandomREV(MPI_COMM_WORLD, &state, nObjs, objSize);
    auto lrev = rev->local();
    auto hrev = rev->halo();

    auto& lpos = lrev->positions();
    auto& lvel = lrev->velocities();
    auto& lforces = lrev->forces();

    auto& hpos = hrev->positions();
    auto& hvel = hrev->velocities();
    auto& hforces = hrev->forces();

    lpos.downloadFromDevice(defaultStream);

    // will send the forces computed from periodic field
    // and then compare to what it should be
    std::vector<std::string> extraExchangeChannels = {ChannelNames::forces};

    auto exchanger = std::make_unique<ObjectHaloExchanger>();

    exchanger->attach(rev.get(), rc, extraExchangeChannels);
        
    auto engineExchange = std::make_unique<SingleNodeEngine>(std::move(exchanger));

    clearForces(lforces);
    applyFieldUnbounded(lpos, lforces, domain.localSize);
    lforces.uploadToDevice(defaultStream);

    engineExchange->init(defaultStream);
    engineExchange->finalize(defaultStream);

    hpos   .downloadFromDevice(defaultStream);
    hforces.downloadFromDevice(defaultStream);
    
    checkForces(hpos, hforces, domain.localSize);
}


TEST (PACKERS_EXCHANGE, objects_reverse_exchange)
{
    float dt = 0.f;
    float rc = 1.f;
    float L  = 48.f;
    int nObjs = 1024;
    int objSize = 555;

    DomainInfo domain;
    domain.globalSize  = {L, L, L};
    domain.globalStart = {0.f, 0.f, 0.f};
    domain.localSize   = {L, L, L};
    MirState state(domain, dt);
    auto rev = initializeRandomREV(MPI_COMM_WORLD, &state, nObjs, objSize);
    auto lrev = rev->local();
    auto hrev = rev->halo();

    auto& lpos = lrev->positions();
    auto& lvel = lrev->velocities();
    auto& lforces = lrev->forces();

    auto& hpos = hrev->positions();
    auto& hvel = hrev->velocities();
    auto& hforces = hrev->forces();

    lpos.downloadFromDevice(defaultStream);

    clearForces(lforces);
    std::vector<Force> refForces(lforces.begin(), lforces.end());

    std::vector<std::string>   extraExchangeChannels = {};
    std::vector<std::string> reverseExchangeChannels = {ChannelNames::forces};

    auto exchanger        = std::make_unique<ObjectHaloExchanger>();
    auto reverseExchanger = std::make_unique<ObjectReverseExchanger>(exchanger.get());

    exchanger       ->attach(rev.get(), rc, extraExchangeChannels);
    reverseExchanger->attach(rev.get(),   reverseExchangeChannels);
        
    auto engineExchange        = std::make_unique<SingleNodeEngine>(std::move(exchanger));
    auto engineReverseExchange = std::make_unique<SingleNodeEngine>(std::move(reverseExchanger));

    engineExchange->init(defaultStream);
    engineExchange->finalize(defaultStream);

    applyFieldPeriodic(lpos, refForces, domain.localSize);

    hpos   .downloadFromDevice(defaultStream);
    hforces.downloadFromDevice(defaultStream);
    clearForces(hforces);
    
    applyFieldLocal(lpos, lforces, domain.localSize);
    applyFieldLocal(hpos, hforces, domain.localSize);

    lforces.uploadToDevice(defaultStream);
    hforces.uploadToDevice(defaultStream);

    engineReverseExchange->init(defaultStream);
    engineReverseExchange->finalize(defaultStream);

    lforces.downloadFromDevice(defaultStream);

    compareForces(lforces, refForces);
}

TEST (PACKERS_EXCHANGE, objects_extra_exchange)
{
    float dt = 0.f;
    float rc = 1.f;
    float L  = 48.f;
    int nObjs = 1024;
    int objSize = 555;

    DomainInfo domain;
    domain.globalSize  = {L, L, L};
    domain.globalStart = {0.f, 0.f, 0.f};
    domain.localSize   = {L, L, L};
    MirState state(domain, dt);
    auto rev = initializeRandomREV(MPI_COMM_WORLD, &state, nObjs, objSize);
    auto lrev = rev->local();
    auto hrev = rev->halo();

    const std::string extraChannelName = "single_float_field";
    
    rev->requireDataPerParticle<float>(extraChannelName,
                                       DataManager::PersistenceMode::None,
                                       DataManager::ShiftMode::None);

    auto& lpos = lrev->positions();
    auto& lvel = lrev->velocities();
    auto& lforces = lrev->forces();
    auto& lfield = *lrev->dataPerParticle.getData<float>(extraChannelName);

    auto& hpos = hrev->positions();
    auto& hvel = hrev->velocities();
    auto& hforces = hrev->forces();
    auto& hfield = *hrev->dataPerParticle.getData<float>(extraChannelName);

    auto fieldTransform = [](Force f){return length(f.f);};
    
    std::vector<std::string> exchangeChannels = {ChannelNames::forces};
    std::vector<std::string> extraExchangeChannels = {extraChannelName};

    lpos.downloadFromDevice(defaultStream);
    clearForces(lforces);
    applyFieldUnbounded(lpos, lforces, domain.localSize);
    std::transform(lforces.begin(), lforces.end(), lfield.begin(), fieldTransform);
                   
    lforces.uploadToDevice(defaultStream);
    lfield.uploadToDevice(defaultStream);
    
    auto exchanger      = std::make_unique<ObjectHaloExchanger>();
    auto extraExchanger = std::make_unique<ObjectExtraExchanger>(exchanger.get());

    exchanger     ->attach(rev.get(), rc, exchangeChannels);
    extraExchanger->attach(rev.get(), extraExchangeChannels);
        
    auto engineExchange      = std::make_unique<SingleNodeEngine>(std::move(exchanger));
    auto engineExtraExchange = std::make_unique<SingleNodeEngine>(std::move(extraExchanger));

    engineExchange->init(defaultStream);
    engineExchange->finalize(defaultStream);

    hforces.downloadFromDevice(defaultStream);

    engineExtraExchange->init(defaultStream);
    engineExtraExchange->finalize(defaultStream);

    hfield.downloadFromDevice(defaultStream);

    for (size_t i = 0; i < hforces.size(); ++i)
    {
        auto ref = fieldTransform(hforces[i]);
        auto val = hfield[i];
        ASSERT_EQ(ref, val) << "wrong value for index " << i;
    }
}


int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    logger.init(MPI_COMM_WORLD, "packers_exchange.log", 3);    

    testing::InitGoogleTest(&argc, argv);
    auto retval = RUN_ALL_TESTS();

    MPI_Finalize();
    return retval;
}
