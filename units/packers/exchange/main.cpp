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



struct Comp // for sorting
{
    bool operator()(float4 a_, float4 b_) const
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


int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    logger.init(MPI_COMM_WORLD, "packers_simple.log", 3);    

    testing::InitGoogleTest(&argc, argv);
    auto retval = RUN_ALL_TESTS();

    MPI_Finalize();
    return retval;
}
