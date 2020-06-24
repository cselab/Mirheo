#include <mirheo/core/logger.h>
#include <mirheo/core/utils/helper_math.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/initial_conditions/rod.h>
#include <mirheo/core/pvs/rod_vector.h>
#include <mirheo/core/utils/quaternion.h>

#include <vector>
#include <functional>
#include <gtest/gtest.h>

using namespace mirheo;

using Real  = double;
using Real2 = double2;
using Real3 = double3;
using Real4 = double4;

inline Real2 make_Real2(real2 v) { return {(Real) v.x, (Real) v.y}; }
inline Real3 make_Real3(real3 v) { return {(Real) v.x, (Real) v.y, (Real) v.z}; }
inline Real3 make_Real3(real4 v) { return {(Real) v.x, (Real) v.y, (Real) v.z}; }

using CenterLineFunc = std::function<Real3(Real)>;
using CurvatureFunc  = std::function<Real(Real)>;
using TorsionFunc    = std::function<Real(Real)>;

constexpr real a = 0.05f;
constexpr real dt = 0.f;

static std::vector<Real> computeCurvatures(const real4 *positions, int nSegments)
{
    std::vector<Real> curvatures;
    curvatures.reserve(nSegments-1);

    for (int i = 0; i < nSegments - 1; ++i)
    {
        auto r0  = make_Real3(positions[5*(i+0)]);
        auto r1  = make_Real3(positions[5*(i+1)]);
        auto r2  = make_Real3(positions[5*(i+2)]);

        auto e0 = r1 - r0;
        auto e1 = r2 - r1;

        Real le0 = length(e0);
        Real le1 = length(e1);
        Real l = 0.5 * (le0 + le1);
        auto bicurFactor = 1.0 / (le0 * le1 + dot(e0, e1));
        auto bicur = (2.0 * bicurFactor) * cross(e0, e1);
        auto kappa = (1/l) * bicur;

        curvatures.push_back(length(kappa));
    }
    return curvatures;
}

inline Real safeDiffTheta(Real t0, Real t1)
{
    auto dth = t1 - t0;
    if (dth >  M_PI) dth -= 2.0 * M_PI;
    if (dth < -M_PI) dth += 2.0 * M_PI;
    return dth;
}

static std::vector<Real> computeTorsions(const real4 *positions, int nSegments)
{
    std::vector<Real> torsions;
    torsions.reserve(nSegments-1);

    for (int i = 0; i < nSegments - 1; ++i)
    {
        auto r0  = make_Real3(positions[5*(i+0)]);
        auto r1  = make_Real3(positions[5*(i+1)]);
        auto r2  = make_Real3(positions[5*(i+2)]);

        auto pm0 = make_Real3(positions[5*i + 1]);
        auto pp0 = make_Real3(positions[5*i + 2]);
        auto pm1 = make_Real3(positions[5*i + 6]);
        auto pp1 = make_Real3(positions[5*i + 7]);

        auto e0 = r1 - r0;
        auto e1 = r2 - r1;

        auto t0 = normalize(e0);
        auto t1 = normalize(e1);

        const auto Q = Quaternion<Real>::createFromVectors(t0, t1);
        Real3 u0 = normalize(anyOrthogonal(t0));
        Real3 u1 = normalize(Q.rotate(u0));

        auto dp0 = pp0 - pm0;
        auto dp1 = pp1 - pm1;

        Real le0 = length(e0);
        Real le1 = length(e1);
        auto linv = 2.0 / (le0 + le1);

        auto v0 = cross(t0, u0);
        auto v1 = cross(t1, u1);

        Real dpu0 = dot(dp0, u0);
        Real dpv0 = dot(dp0, v0);

        Real dpu1 = dot(dp1, u1);
        Real dpv1 = dot(dp1, v1);

        Real theta0 = atan2(dpv0, dpu0);
        Real theta1 = atan2(dpv1, dpu1);

        Real tau = safeDiffTheta(theta0, theta1) * linv;

        torsions.push_back(tau);
    }
    return torsions;
}


static Real checkCurvature(const MPI_Comm& comm, CenterLineFunc centerLine, int nSegments, CurvatureFunc ref)
{
    RodIC::MappingFunc3D mirCenterLine = [&](real s)
    {
        auto r = centerLine(s);
        return real3({(real) r.x, (real) r.y, (real) r.z});
    };

    RodIC::MappingFunc1D mirTorsion = [&](__UNUSED real s)
    {
        return 0.f;
    };

    DomainInfo domain;
    real L = 32._r;
    domain.globalSize  = {L, L, L};
    domain.globalStart = {0._r, 0._r, 0._r};
    domain.localSize   = {L, L, L};
    real mass = 1._r;
    MirState state(domain, dt, UnitConversion{});
    RodVector rv(&state, "rod", mass, nSegments);

    ComQ comq = {{L/2, L/2, L/2}, {1.0_r, 0.0_r, 0.0_r, 0.0_r}};
    RodIC ic({comq}, mirCenterLine, mirTorsion, a);

    ic.exec(comm, &rv, defaultStream);

    auto& pos = rv.local()->positions();

    auto curvatures = computeCurvatures(pos.data(), nSegments);

    Real h = 1.0 / nSegments;
    Real err = 0;

    for (int i = 0; i < nSegments - 1; ++i)
    {
        Real s = (i+1) * h;
        auto curvRef = ref(s);
        auto curvSim = curvatures[i];
        auto dcurv = curvSim - curvRef;
        // printf("%g %g\n", curvRef, curvSim);
        err += dcurv * dcurv;
    }

    return math::sqrt(err / nSegments);
}

static Real checkTorsion(const MPI_Comm& comm, CenterLineFunc centerLine, TorsionFunc torsion, int nSegments)
{
    RodIC::MappingFunc3D mirCenterLine = [&](real s)
    {
        auto r = centerLine(s);
        return real3({(real) r.x, (real) r.y, (real) r.z});
    };

    RodIC::MappingFunc1D mirTorsion = [&](real s)
    {
        return (real) torsion(s);
    };

    DomainInfo domain;
    real L = 32._r;
    domain.globalSize  = {L, L, L};
    domain.globalStart = {0._r, 0._r, 0._r};
    domain.localSize   = {L, L, L};
    real mass = 1.f;
    MirState state(domain, dt, UnitConversion{});
    RodVector rv(&state, "rod", mass, nSegments);

    ComQ comq = {{L/2, L/2, L/2}, {1.0_r, 0.0_r, 0.0_r, 0.0_r}};
    RodIC ic({comq}, mirCenterLine, mirTorsion, a);

    ic.exec(comm, &rv, defaultStream);

    auto& pos = rv.local()->positions();

    auto torsions = computeTorsions(pos.data(), nSegments);

    Real h = 1.0 / nSegments;
    Real err = 0;

    for (int i = 0; i < nSegments - 1; ++i)
    {
        Real s = (i+1) * h;
        auto tauRef = torsion(s);
        auto tauSim = torsions[i];
        auto dtau = tauSim - tauRef;
        // printf("%g %g\n", tauRef, tauSim);
        err += dtau * dtau;
    }

    return math::sqrt(err / nSegments);
}


TEST (ROD, curvature_straight)
{
    Real L = 5.0;

    auto centerLine = [&](Real s) -> Real3
    {
        return {(s-0.5) * L, 0., 0.};
    };

    auto analyticCurv = [&](__UNUSED Real s) -> Real
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
    Real radius = 1.2;

    auto centerLine = [&](Real s) -> Real3
    {
        Real coss = cos(2*M_PI*s);
        Real sins = sin(2*M_PI*s);
        return {radius * coss, radius * sins, 0.};
    };

    auto analyticCurv = [&](__UNUSED Real s) -> Real
    {
        return 1 / radius;
    };

    std::vector<int> nsegs = {8, 16, 32, 64, 128};
    std::vector<Real> errors;
    for (auto n : nsegs)
        errors.push_back( checkCurvature(MPI_COMM_WORLD, centerLine, n, analyticCurv) );

    // check convergence rate
    const Real rateTh = 2;

    for (int i = 0; i < static_cast<int>(nsegs.size()) - 1; ++i)
    {
        Real e0 = errors[i], e1 = errors[i+1];
        int  n0 =  nsegs[i], n1 =  nsegs[i+1];

        Real rate = (log(e0) - log(e1)) / (log(n1) - log(n0));

        ASSERT_LE(math::abs(rate-rateTh), 1e-1);
    }
}

TEST (ROD, curvature_helix)
{
    Real a = 1.2;
    Real b = 2.32;

    auto centerLine = [&](Real s) -> Real3
    {
        Real t = 2 * M_PI * s;
        return {a * cos(t), a * sin(t), b * t};
    };

    auto analyticCurv = [&](__UNUSED Real s) -> Real
    {
        return math::abs(a) / (a*a + b*b);
    };

    std::vector<int> nsegs = {8, 16, 32, 64, 128};
    std::vector<Real> errors;
    for (auto n : nsegs)
        errors.push_back( checkCurvature(MPI_COMM_WORLD, centerLine, n, analyticCurv) );

    // check convergence rate
    const Real rateTh = 2;

    for (int i = 0; i < static_cast<int>(nsegs.size()) - 1; ++i)
    {
        Real e0 = errors[i], e1 = errors[i+1];
        int  n0 =  nsegs[i], n1 =  nsegs[i+1];

        Real rate = (log(e0) - log(e1)) / (log(n1) - log(n0));

        ASSERT_LE(math::abs(rate-rateTh), 1e-1);
    }
}


TEST (ROD, torsion_straight_const)
{
    Real L = 5.0;
    Real tau = 0.5;

    auto centerLine = [&](Real s) -> Real3
    {
        return {(s-0.5) * L, 0., 0.};
    };

    auto torsion = [&](__UNUSED Real s)
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
    Real L = 5.0;
    Real tauA = 1.5;

    auto centerLine = [&](Real s) -> Real3
    {
        return {(s-0.5) * L, 0., 0.};
    };

    auto torsion = [&](Real s)
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
    Real radius = 1.2;
    Real tauA = 1.5;

    auto centerLine = [&](Real s) -> Real3
    {
        Real coss = cos(2*M_PI*s);
        Real sins = sin(2*M_PI*s);
        return {radius * coss, radius * sins, 0.};
    };

    auto torsion = [&](Real s)
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
    Real a = 1.2;
    Real b = 2.32;

    auto centerLine = [&](Real s) -> Real3
    {
        Real t = 2 * M_PI * s;
        return {a * cos(t), a * sin(t), b * t};
    };

    auto torsion = [&](__UNUSED Real s)
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
