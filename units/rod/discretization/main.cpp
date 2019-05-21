#include <core/logger.h>
#include <core/utils/helper_math.h>
#include <core/utils/cuda_common.h>
#include <core/initial_conditions/rod.h>
#include <core/pvs/rod_vector.h>

#include <vector>
#include <functional>
#include <gtest/gtest.h>

Logger logger;

using real = double;
using real2 = double2;
using real3 = double3;
using real4 = double4;

static real2 make_real2(float2 v) { return {(real) v.x, (real) v.y}; }
static real3 make_real3(float3 v) { return {(real) v.x, (real) v.y, (real) v.z}; }

using CenterLineFunc = std::function<real3(real)>;
using CurvatureFunc  = std::function<real(real)>;

constexpr float a = 0.05f;
constexpr float dt = 0.f;

static std::vector<real> computeCurvatures(const float4 *positions, int nSegments)
{
    std::vector<real> curvatures;
    curvatures.reserve(nSegments-1);

    for (int i = 0; i < nSegments - 1; ++i)
    {
        auto r0  = make_float3(positions[5*(i+0)]);
        auto r1  = make_float3(positions[5*(i+1)]);
        auto r2  = make_float3(positions[5*(i+2)]);

        auto e0 = r1 - r0;
        auto e1 = r2 - r1;

        auto t0 = normalize(e0);
        auto t1 = normalize(e1);

        real le0 = length(e0);
        real le1 = length(e1);
        real l = 0.5 * (le0 + le1);
        auto bicurFactor = 1.0 / (le0 * le1 + dot(e0, e1));
        auto bicur = (2.0 * bicurFactor) * cross(e0, e1);
        
        curvatures.push_back(length(bicur) / l);
    }
    return curvatures;
}


static real checkCurvature(const MPI_Comm& comm, CenterLineFunc centerLine, int nSegments, CurvatureFunc ref)
{
    RodIC::MappingFunc3D ymrCenterLine = [&](float s)
    {
        auto r = centerLine(s);
        return PyTypes::float3({(float) r.x, (float) r.y, (float) r.z});
    };
    
    RodIC::MappingFunc1D ymrTorsion = [&](float s)
    {
        return 0.f;
    };

    DomainInfo domain;
    float L = 32.f;
    domain.globalSize  = {L, L, L};
    domain.globalStart = {0.f, 0.f, 0.f};
    domain.localSize   = {L, L, L};
    float mass = 1.f;
    YmrState state(domain, dt);
    RodVector rv(&state, "rod", mass, nSegments);
    
    RodIC ic({{L/2, L/2, L/2, 1.0f, 0.0f, 0.0f}},
             ymrCenterLine, ymrTorsion, a);
    
    ic.exec(comm, &rv, defaultStream);

    auto& pos = rv.local()->positions();
    
    auto curvatures = computeCurvatures(pos.data(), nSegments);

    real h = 1.0 / nSegments;
    real err = 0;
    
    for (int i = 0; i < nSegments - 1; ++i)
    {
        real s = (i+1) * h;
        auto curvRef = ref(s);
        auto curvSim = curvatures[i];
        auto dcurv = curvSim - curvRef;
        // printf("%g %g\n", curvRef, curvSim);
        err += dcurv * dcurv;
    }

    return sqrt(err / nSegments);
}


TEST (ROD, curvature)
{
    real radius = 1.2;
    
    auto centerLine = [&](real s) -> real3
    {
        real coss = cos(2*M_PI*s);
        real sins = sin(2*M_PI*s);
        return {radius * coss, radius * sins, 0.};
    };

    auto analyticCurv = [&](real s) -> real
    {
        return 1 / radius;
    };

    std::vector<int> nsegs = {8, 16, 32, 64, 128};
    std::vector<real> errors;
    for (auto n : nsegs)
        errors.push_back( checkCurvature(MPI_COMM_WORLD, centerLine, n, analyticCurv) );

    // check convergence rate
    const real rateTh = 2;

    for (int i = 0; i < nsegs.size() - 1; ++i)
    {
        real e0 = errors[i], e1 = errors[i+1];
        int  n0 =  nsegs[i], n1 =  nsegs[i+1];

        real rate = (log(e0) - log(e1)) / (log(n1) - log(n0));

        ASSERT_LE(fabs(rate-rateTh), 1e-1);
    }
}



int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    logger.init(MPI_COMM_WORLD, "flagella.log", 9);
    
    testing::InitGoogleTest(&argc, argv);
    auto ret = RUN_ALL_TESTS();

    MPI_Finalize();
    return ret;
}
