#include <core/logger.h>
#include <core/utils/helper_math.h>
#include <core/utils/cuda_common.h>
#include <core/initial_conditions/rod.h>
#include <core/pvs/rod_vector.h>
#include <core/utils/quaternion.h>

#include <vector>
#include <functional>
#include <gtest/gtest.h>

Logger logger;

using real  = double;
using real2 = double2;
using real3 = double3;
using real4 = double4;

inline real2 make_real2(float2 v) { return {(real) v.x, (real) v.y}; }
inline real3 make_real3(float3 v) { return {(real) v.x, (real) v.y, (real) v.z}; }
inline real3 make_real3(float4 v) { return {(real) v.x, (real) v.y, (real) v.z}; }

using CenterLineFunc = std::function<real3(real)>;
using CurvatureFunc  = std::function<real(real)>;
using TorsionFunc    = std::function<real(real)>;

constexpr float a = 0.05f;
constexpr float dt = 0.f;

static std::vector<real> computeCurvatures(const float4 *positions, int nSegments)
{
    std::vector<real> curvatures;
    curvatures.reserve(nSegments-1);

    for (int i = 0; i < nSegments - 1; ++i)
    {
        auto r0  = make_real3(positions[5*(i+0)]);
        auto r1  = make_real3(positions[5*(i+1)]);
        auto r2  = make_real3(positions[5*(i+2)]);

        auto e0 = r1 - r0;
        auto e1 = r2 - r1;

        real le0 = length(e0);
        real le1 = length(e1);
        real l = 0.5 * (le0 + le1);
        auto bicurFactor = 1.0 / (le0 * le1 + dot(e0, e1));
        auto bicur = (2.0 * bicurFactor) * cross(e0, e1);
        auto kappa = (1/l) * bicur;
        
        curvatures.push_back(length(kappa));
    }
    return curvatures;
}

inline real safeDiffTheta(real t0, real t1)
{
    auto dth = t1 - t0;
    if (dth >  M_PI) dth -= 2.0 * M_PI;
    if (dth < -M_PI) dth += 2.0 * M_PI;
    return dth;
}

static std::vector<real> computeTorsions(const float4 *positions, int nSegments)
{
    std::vector<real> torsions;
    torsions.reserve(nSegments-1);

    for (int i = 0; i < nSegments - 1; ++i)
    {
        auto r0  = make_real3(positions[5*(i+0)]);
        auto r1  = make_real3(positions[5*(i+1)]);
        auto r2  = make_real3(positions[5*(i+2)]);

        auto pm0 = make_real3(positions[5*i + 1]);
        auto pp0 = make_real3(positions[5*i + 2]);
        auto pm1 = make_real3(positions[5*i + 6]);
        auto pp1 = make_real3(positions[5*i + 7]);

        auto e0 = r1 - r0;
        auto e1 = r2 - r1;

        auto t0 = normalize(e0);
        auto t1 = normalize(e1);
        
        real4  Q = Quaternion::getFromVectorPair(t0, t1);
        real3 u0 = normalize(anyOrthogonal(t0));
        real3 u1 = normalize(Quaternion::rotate(u0, Q));

        auto dp0 = pp0 - pm0;
        auto dp1 = pp1 - pm1;

        real le0 = length(e0);
        real le1 = length(e1);
        auto linv = 2.0 / (le0 + le1);

        auto v0 = cross(t0, u0);
        auto v1 = cross(t1, u1);

        real dpu0 = dot(dp0, u0);
        real dpv0 = dot(dp0, v0);

        real dpu1 = dot(dp1, u1);
        real dpv1 = dot(dp1, v1);

        real theta0 = atan2(dpv0, dpu0);
        real theta1 = atan2(dpv1, dpu1);
    
        real tau = safeDiffTheta(theta0, theta1) * linv;
        
        torsions.push_back(tau);
    }
    return torsions;
}


static real checkCurvature(const MPI_Comm& comm, CenterLineFunc centerLine, int nSegments, CurvatureFunc ref)
{
    RodIC::MappingFunc3D mirCenterLine = [&](float s)
    {
        auto r = centerLine(s);
        return float3({(float) r.x, (float) r.y, (float) r.z});
    };
    
    RodIC::MappingFunc1D mirTorsion = [&](__UNUSED float s)
    {
        return 0.f;
    };

    DomainInfo domain;
    float L = 32.f;
    domain.globalSize  = {L, L, L};
    domain.globalStart = {0.f, 0.f, 0.f};
    domain.localSize   = {L, L, L};
    float mass = 1.f;
    MirState state(domain, dt);
    RodVector rv(&state, "rod", mass, nSegments);

    ComQ comq = {{L/2, L/2, L/2}, {1.0f, 0.0f, 0.0f, 0.0f}};
    RodIC ic({comq}, mirCenterLine, mirTorsion, a);
    
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

static real checkTorsion(const MPI_Comm& comm, CenterLineFunc centerLine, TorsionFunc torsion, int nSegments)
{
    RodIC::MappingFunc3D mirCenterLine = [&](float s)
    {
        auto r = centerLine(s);
        return float3({(float) r.x, (float) r.y, (float) r.z});
    };
    
    RodIC::MappingFunc1D mirTorsion = [&](float s)
    {
        return (float) torsion(s);
    };

    DomainInfo domain;
    float L = 32.f;
    domain.globalSize  = {L, L, L};
    domain.globalStart = {0.f, 0.f, 0.f};
    domain.localSize   = {L, L, L};
    float mass = 1.f;
    MirState state(domain, dt);
    RodVector rv(&state, "rod", mass, nSegments);

    ComQ comq = {{L/2, L/2, L/2}, {1.0f, 0.0f, 0.0f, 0.0f}};
    RodIC ic({comq}, mirCenterLine, mirTorsion, a);
    
    ic.exec(comm, &rv, defaultStream);

    auto& pos = rv.local()->positions();
    
    auto torsions = computeTorsions(pos.data(), nSegments);

    real h = 1.0 / nSegments;
    real err = 0;
    
    for (int i = 0; i < nSegments - 1; ++i)
    {
        real s = (i+1) * h;
        auto tauRef = torsion(s);
        auto tauSim = torsions[i];
        auto dtau = tauSim - tauRef;
        // printf("%g %g\n", tauRef, tauSim);
        err += dtau * dtau;
    }

    return sqrt(err / nSegments);
}


TEST (ROD, curvature_straight)
{
    real L = 5.0;
    
    auto centerLine = [&](real s) -> real3
    {
        return {(s-0.5) * L, 0., 0.};
    };

    auto analyticCurv = [&](__UNUSED real s) -> real
    {
        return 0.;
    };

    std::vector<int> nsegs = {8, 16, 32, 64, 128};

    for (auto n : nsegs)
    {
        auto err = checkCurvature(MPI_COMM_WORLD, centerLine, n, analyticCurv);
        ASSERT_LE(err, 1e-6);
    }
}

TEST (ROD, curvature_circle)
{
    real radius = 1.2;
    
    auto centerLine = [&](real s) -> real3
    {
        real coss = cos(2*M_PI*s);
        real sins = sin(2*M_PI*s);
        return {radius * coss, radius * sins, 0.};
    };

    auto analyticCurv = [&](__UNUSED real s) -> real
    {
        return 1 / radius;
    };

    std::vector<int> nsegs = {8, 16, 32, 64, 128};
    std::vector<real> errors;
    for (auto n : nsegs)
        errors.push_back( checkCurvature(MPI_COMM_WORLD, centerLine, n, analyticCurv) );

    // check convergence rate
    const real rateTh = 2;

    for (int i = 0; i < static_cast<int>(nsegs.size()) - 1; ++i)
    {
        real e0 = errors[i], e1 = errors[i+1];
        int  n0 =  nsegs[i], n1 =  nsegs[i+1];

        real rate = (log(e0) - log(e1)) / (log(n1) - log(n0));

        ASSERT_LE(fabs(rate-rateTh), 1e-1);
    }
}

TEST (ROD, curvature_helix)
{
    real a = 1.2;
    real b = 2.32;
    
    auto centerLine = [&](real s) -> real3
    {
        real t = 2 * M_PI * s;
        return {a * cos(t), a * sin(t), b * t};
    };

    auto analyticCurv = [&](__UNUSED real s) -> real
    {
        return fabs(a) / (a*a + b*b);
    };

    std::vector<int> nsegs = {8, 16, 32, 64, 128};
    std::vector<real> errors;
    for (auto n : nsegs)
        errors.push_back( checkCurvature(MPI_COMM_WORLD, centerLine, n, analyticCurv) );

    // check convergence rate
    const real rateTh = 2;

    for (int i = 0; i < static_cast<int>(nsegs.size()) - 1; ++i)
    {
        real e0 = errors[i], e1 = errors[i+1];
        int  n0 =  nsegs[i], n1 =  nsegs[i+1];

        real rate = (log(e0) - log(e1)) / (log(n1) - log(n0));

        ASSERT_LE(fabs(rate-rateTh), 1e-1);
    }
}


TEST (ROD, torsion_straight_const)
{
    real L = 5.0;
    real tau = 0.5;
    
    auto centerLine = [&](real s) -> real3
    {
        return {(s-0.5) * L, 0., 0.};
    };

    auto torsion = [&](__UNUSED real s)
    {
        return tau;
    };

    std::vector<int> nsegs = {8, 16, 32, 64, 128};

    for (auto n : nsegs)
    {
        auto err = checkTorsion(MPI_COMM_WORLD, centerLine, torsion, n);
        ASSERT_LE(err, 1e-6);
    }
}

TEST (ROD, torsion_straight_vary)
{
    real L = 5.0;
    real tauA = 1.5;
    
    auto centerLine = [&](real s) -> real3
    {
        return {(s-0.5) * L, 0., 0.};
    };

    auto torsion = [&](real s)
    {
        return s * tauA;
    };

    std::vector<int> nsegs = {8, 16, 32, 64, 128};

    for (auto n : nsegs)
    {
        auto err = checkTorsion(MPI_COMM_WORLD, centerLine, torsion, n);
        ASSERT_LE(err, 1e-6);
    }
}

TEST (ROD, torsion_circle_vary)
{
    real radius = 1.2;
    real tauA = 1.5;
    
    auto centerLine = [&](real s) -> real3
    {
        real coss = cos(2*M_PI*s);
        real sins = sin(2*M_PI*s);
        return {radius * coss, radius * sins, 0.};
    };
    
    auto torsion = [&](real s)
    {
        return s * tauA;
    };

    std::vector<int> nsegs = {8, 16, 32, 64, 128};

    for (auto n : nsegs)
    {
        auto err = checkTorsion(MPI_COMM_WORLD, centerLine, torsion, n);
        ASSERT_LE(err, 5e-5);
    }
}

TEST (ROD, torsion_helix)
{
    real a = 1.2;
    real b = 2.32;
    
    auto centerLine = [&](real s) -> real3
    {
        real t = 2 * M_PI * s;
        return {a * cos(t), a * sin(t), b * t};
    };

    auto torsion = [&](__UNUSED real s)
    {
        return b / (a*a + b*b);
    };

    std::vector<int> nsegs = {8, 16, 32, 64};
    for (auto n : nsegs)
    {
        auto err = checkTorsion(MPI_COMM_WORLD, centerLine, torsion, n);
        // printf("%d %g\n", n, err);
        ASSERT_LE(err, 1e-5);
    }
}


int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    logger.init(MPI_COMM_WORLD, "rod_discretization.log", 9);
    
    testing::InitGoogleTest(&argc, argv);
    auto ret = RUN_ALL_TESTS();

    MPI_Finalize();
    return ret;
}
