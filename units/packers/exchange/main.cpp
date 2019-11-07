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

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <gtest/gtest.h>
#include <random>
#include <vector>

namespace mirheo { Logger logger; }

// create the reference data
// will let us test 2 things:
// - correct reordering
// - correct shift
// by copying corrected positions to velocities
static void createRef(const PinnedBuffer<real4>& pos, PinnedBuffer<real4>& vel)
{
    for (size_t i = 0; i < pos.size(); ++i)
        vel[i] = pos[i];
    
    vel.uploadToDevice(defaultStream);
}

// check that ref pos and vel are exactly L apart
static void checkRef(const PinnedBuffer<real4>& pos,
                     const PinnedBuffer<real4>& vel,
                     real3 L)
{
    constexpr real eps = 1e-6_r;
    for (size_t i = 0; i < pos.size(); ++i)
    {
        auto r = make_real3(pos[i]);
        auto v = make_real3(vel[i]);
        auto e = math::abs(r-v) - L;

        auto minErr = std::min(e.x, std::min(e.y, e.z));

        ASSERT_LE(minErr, eps);
    }
}

struct Comp // for sorting particles
{
    bool inline operator()(real4 a_, real4 b_) const
    {
        Real3_int a(a_), b(b_);
        return
            (a.i < b.i) ||
            (a.i == b.i && a.v.x  < b.v.x) ||
            (a.i == b.i && a.v.x == b.v.x && a.v.y  < b.v.y) ||
            (a.i == b.i && a.v.x == b.v.x && a.v.y == b.v.y && a.v.z == b.v.z);
    } 
};

static void checkHalo(const PinnedBuffer<real4>& lpos,
                      const PinnedBuffer<real4>& hpos,
                      real3 L, real rc)
{
    std::vector<real4> hposSorted(hpos.begin(), hpos.end());
    std::sort(hposSorted.begin(), hposSorted.end(), Comp());

    for (size_t i = 0; i < lpos.size(); ++i)
    {
        const auto r0 = lpos[i];

        int dx = -1 + (r0.x >= -0.5f * L.x + rc) + (r0.x >= 0.5f * L.x - rc);
        int dy = -1 + (r0.y >= -0.5f * L.y + rc) + (r0.y >= 0.5f * L.y - rc);
        int dz = -1 + (r0.z >= -0.5f * L.z + rc) + (r0.z >= 0.5f * L.z - rc);
        
        for (int iz = math::min(dz,0); iz < math::max(dz,0); ++iz)
        for (int iy = math::min(dy,0); iy < math::max(dy,0); ++iy)
        for (int ix = math::min(dx,0); ix < math::max(dx,0); ++ix)
        {
            if (ix == 0 && iy == 0 && iz == 0) continue;

            real4 r = {r0.x - ix * L.x,
                       r0.y - iy * L.y,
                       r0.z - iz * L.z,
                       r0.w};

            auto less = [](real4 a_, real4 b_)
            {
                Real3_int a(a_), b(b_);
                return a.i < b.i;
            };

            auto lessEq = [](real4 a_, real4 b_)
            {
                Real3_int a(a_), b(b_);
                return a.i <= b.i;
            };

            auto lb = std::lower_bound(hposSorted.begin(),
                                       hposSorted.end(),
                                       r, less);

            real err = 1e9f;
            
            while (lb != hposSorted.end() && lessEq(*lb, r))
            {
                real errx = math::abs(lb->x - r.x);
                real erry = math::abs(lb->y - r.y);
                real errz = math::abs(lb->z - r.z);
                real curr = std::min(errx, std::min(erry, errz));
                err = std::min(err, curr);
                ++lb;
            };
            
            ASSERT_LE(err, 1e-6f);
        }
    }
}

TEST (PACKERS_EXCHANGE, particles)
{
    real dt = 0.0_r;
    real rc = 1.0_r;
    real L  = 48.0_r;
    real density = 8.0_r;
    DomainInfo domain;
    domain.globalSize  = {L, L, L};
    domain.globalStart = {0.0_r, 0.0_r, 0.0_r};
    domain.localSize   = {L, L, L};
    MirState state(domain, dt);
    auto pv = initializeRandomPV(MPI_COMM_WORLD, &state, density);
    auto lpv = pv->local();
    auto hpv = pv->halo();

    auto& lpos = lpv->positions();
    auto& lvel = lpv->velocities();

    createRef(lpos, lvel);    

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
        f.f = make_real3(0.0_r);
}

inline bool isInside(real3 r, real3 L)
{
    return
        (r.x >= - 0.5_r * L.x && r.x < 0.5_r * L.x) &&
        (r.y >= - 0.5_r * L.y && r.y < 0.5_r * L.y) &&
        (r.z >= - 0.5_r * L.z && r.z < 0.5_r * L.z);
}

inline Force getField(real3 r)
{
    Force f;
    f.f = length(r) * r;
    return f;
}

template <class ForceCont>
static void applyFieldLocal(const PinnedBuffer<real4>& pos,
                            ForceCont& force, real3 L)
{
    for (size_t i = 0; i < pos.size(); ++i)
    {
        auto r = make_real3(pos[i]);
        if (isInside(r, L))
            force[i] += getField(r);
    }
}

template <class ForceCont>
static void applyFieldPeriodic(const PinnedBuffer<real4>& pos,
                               ForceCont& forces, real3 L)
{
    for (size_t i = 0; i < pos.size(); ++i)
    {
        const auto r0 = make_real3(pos[i]);

        for (int ix = -1; ix < 2; ++ix)
        for (int iy = -1; iy < 2; ++iy)
        for (int iz = -1; iz < 2; ++iz)
        {
            real3 r {r0.x + ix * L.x,
                     r0.y + iy * L.y,
                     r0.z + iz * L.z};
        
            if (isInside(r, L))
                forces[i] += getField(r);
        }
    }
}

template <class ForceCont>
static void applyFieldUnbounded(const PinnedBuffer<real4>& pos,
                               ForceCont& forces)
{
    for (size_t i = 0; i < pos.size(); ++i)
    {
        const auto r = make_real3(pos[i]);
        forces[i] += getField(r);
    }
}


inline real linf(real3 a, real3 b)
{
    auto d = math::abs(a-b);
    return std::min(d.x, std::min(d.y, d.z));
}

static void checkForces(const PinnedBuffer<real4>& pos,
                        const PinnedBuffer<Force>& forces,
                        real3 L)
{
    for (size_t i = 0; i < pos.size(); ++i)
    {
        const auto r0 = make_real3(pos[i]);
        const auto f0 = forces[i].f;
        real err = 1e9_r;

        for (int ix = -1; ix < 2; ++ix)
        for (int iy = -1; iy < 2; ++iy)
        for (int iz = -1; iz < 2; ++iz)
        {
            if (ix == 0 && iy == 0 && iz == 0) continue;

            const real3 r {r0.x + ix * L.x,
                           r0.y + iy * L.y,
                           r0.z + iz * L.z};
            const auto f = getField(r).f;
            
            err = std::min(err, linf(f0, f));
        }
        ASSERT_LE(err, 1e-6_r);
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
        ASSERT_LE(err, 1e-6_r);

        // ASSERT_TRUE(areEquals(fA, fB));
    }
}



TEST (PACKERS_EXCHANGE, objects_exchange)
{
    real dt = 0.0_r;
    real rc = 1.0_r;
    real L  = 48.0_r;
    int nObjs = 1024;
    int objSize = 555;

    DomainInfo domain;
    domain.globalSize  = {L, L, L};
    domain.globalStart = {0.0_r, 0.0_r, 0.0_r};
    domain.localSize   = {L, L, L};
    MirState state(domain, dt);
    auto rev = initializeRandomREV(MPI_COMM_WORLD, &state, nObjs, objSize);
    auto lrev = rev->local();
    auto hrev = rev->halo();

    auto& lpos = lrev->positions();
    auto& lforces = lrev->forces();

    auto& hpos = hrev->positions();
    auto& hforces = hrev->forces();

    lpos.downloadFromDevice(defaultStream);

    // will send the forces computed from periodic field
    // and then compare to what it should be
    std::vector<std::string> extraExchangeChannels = {ChannelNames::forces};

    auto exchanger = std::make_unique<ObjectHaloExchanger>();

    exchanger->attach(rev.get(), rc, extraExchangeChannels);
        
    auto engineExchange = std::make_unique<SingleNodeEngine>(std::move(exchanger));

    clearForces(lforces);
    applyFieldUnbounded(lpos, lforces);
    lforces.uploadToDevice(defaultStream);

    engineExchange->init(defaultStream);
    engineExchange->finalize(defaultStream);

    hpos   .downloadFromDevice(defaultStream);
    hforces.downloadFromDevice(defaultStream);
    
    checkForces(hpos, hforces, domain.localSize);
}


TEST (PACKERS_EXCHANGE, objects_reverse_exchange)
{
    real dt = 0.0_r;
    real rc = 1.0_r;
    real L  = 48.0_r;
    int nObjs = 1024;
    int objSize = 555;

    DomainInfo domain;
    domain.globalSize  = {L, L, L};
    domain.globalStart = {0.0_r, 0.0_r, 0.0_r};
    domain.localSize   = {L, L, L};
    MirState state(domain, dt);
    auto rev = initializeRandomREV(MPI_COMM_WORLD, &state, nObjs, objSize);
    auto lrev = rev->local();
    auto hrev = rev->halo();

    auto& lpos = lrev->positions();
    auto& lforces = lrev->forces();

    auto& hpos = hrev->positions();
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
    real dt = 0.0_r;
    real rc = 1.0_r;
    real L  = 48.0_r;
    int nObjs = 1024;
    int objSize = 555;

    DomainInfo domain;
    domain.globalSize  = {L, L, L};
    domain.globalStart = {0.0_r, 0.0_r, 0.0_r};
    domain.localSize   = {L, L, L};
    MirState state(domain, dt);
    auto rev = initializeRandomREV(MPI_COMM_WORLD, &state, nObjs, objSize);
    auto lrev = rev->local();
    auto hrev = rev->halo();

    const std::string extraChannelName = "single_real_field";
    
    rev->requireDataPerParticle<real>(extraChannelName,
                                       DataManager::PersistenceMode::None,
                                       DataManager::ShiftMode::None);

    auto& lpos = lrev->positions();
    auto& lforces = lrev->forces();
    auto& lfield = *lrev->dataPerParticle.getData<real>(extraChannelName);

    auto& hforces = hrev->forces();
    auto& hfield = *hrev->dataPerParticle.getData<real>(extraChannelName);

    auto fieldTransform = [](Force f){return length(f.f);};
    
    std::vector<std::string> exchangeChannels = {ChannelNames::forces};
    std::vector<std::string> extraExchangeChannels = {extraChannelName};

    lpos.downloadFromDevice(defaultStream);
    clearForces(lforces);
    applyFieldUnbounded(lpos, lforces);
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
