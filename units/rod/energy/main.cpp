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
static real3 make_real3(float4 v) { return {(real) v.x, (real) v.y, (real) v.z}; }

using CenterLineFunc = std::function<real3(real)>;
using EnergyFunc     = std::function<real(real)>;
using TorsionFunc    = std::function<real(real)>;

constexpr float a = 0.05f;
constexpr float dt = 0.f;

inline real2 symmetricMatMult(const real3& A, const real2& x)
{
    return {A.x * x.x + A.y * x.y,
            A.y * x.x + A.z * x.y};
}

static std::vector<real> computeBendingEnergies(const float4 *positions, int nSegments, real3 kBending, real2 omegaEq)
{
    std::vector<real> energies;
    energies.reserve(nSegments-1);

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

        auto dp0 = pp0 - pm0;
        auto dp1 = pp1 - pm1;

        real le0 = length(e0);
        real le1 = length(e1);
        real linv = 2.0 / (le0 + le1);
        auto bicurFactor = 1.0 / (le0 * le1 + dot(e0, e1));
        auto bicur = (2.0 * bicurFactor) * cross(e0, e1);

        real dpt0 = dot(dp0, t0);
        real dpt1 = dot(dp1, t1);

        real3 t0_dp0 = cross(t0, dp0);
        real3 t1_dp1 = cross(t1, dp1);
    
        real3 dpPerp0 = dp0 - dpt0 * t0;
        real3 dpPerp1 = dp1 - dpt1 * t1;

        real dpPerp0inv = rsqrtf(dot(dpPerp0, dpPerp0));
        real dpPerp1inv = rsqrtf(dot(dpPerp1, dpPerp1));
    
        real2 omega0 { +dpPerp0inv * dot(bicur, t0_dp0),
                       -dpPerp0inv * dot(bicur,    dp0)};

        real2 omega1 { +dpPerp1inv * dot(bicur, t1_dp1),
                       -dpPerp1inv * dot(bicur,    dp1)};

        real2 domega0 = omega0 - omegaEq;
        real2 domega1 = omega1 - omegaEq;

        real2 Bomega0 = symmetricMatMult(kBending, domega0);
        real2 Bomega1 = symmetricMatMult(kBending, domega1);

        real Eb = 0.25 * linv * (dot(domega0, Bomega0) + dot(domega1, Bomega1));
        
        energies.push_back(Eb / linv);
    }
    return energies;
}

inline real safeDiffTheta(real t0, real t1)
{
    auto dth = t1 - t0;
    if (dth >  M_PI) dth -= 2.0 * M_PI;
    if (dth < -M_PI) dth += 2.0 * M_PI;
    return dth;
}

// static std::vector<real> computeTwistEnergies(const float4 *positions, const float3 *bishopFrames, int nSegments)
// {
//     std::vector<real> torsions;
//     torsions.reserve(nSegments-1);

//     for (int i = 0; i < nSegments - 1; ++i)
//     {
//         auto r0  = make_real3(positions[5*(i+0)]);
//         auto r1  = make_real3(positions[5*(i+1)]);
//         auto r2  = make_real3(positions[5*(i+2)]);

//         auto pm0 = make_real3(positions[5*i + 1]);
//         auto pp0 = make_real3(positions[5*i + 2]);
//         auto pm1 = make_real3(positions[5*i + 6]);
//         auto pp1 = make_real3(positions[5*i + 7]);

//         const auto u0 = make_real3(bishopFrames[i+0]);
//         const auto u1 = make_real3(bishopFrames[i+1]);

//         auto e0 = r1 - r0;
//         auto e1 = r2 - r1;

//         auto t0 = normalize(e0);
//         auto t1 = normalize(e1);

//         auto dp0 = pp0 - pm0;
//         auto dp1 = pp1 - pm1;

//         real le0 = length(e0);
//         real le1 = length(e1);
//         auto linv = 2.0 / (le0 + le1);

//         auto v0 = cross(t0, u0);
//         auto v1 = cross(t1, u1);

//         real dpu0 = dot(dp0, u0);
//         real dpv0 = dot(dp0, v0);

//         real dpu1 = dot(dp1, u1);
//         real dpv1 = dot(dp1, v1);

//         real theta0 = atan2(dpv0, dpu0);
//         real theta1 = atan2(dpv1, dpu1);
    
//         real tau = safeDiffTheta(theta0, theta1) * linv;
        
//         torsions.push_back(tau);
//     }
//     return torsions;
// }


static real checkBendingEnergy(const MPI_Comm& comm, CenterLineFunc centerLine, TorsionFunc torsion, int nSegments,
                               real3 kBending, real2 omegaEq, EnergyFunc ref)
{
    RodIC::MappingFunc3D ymrCenterLine = [&](float s)
    {
        auto r = centerLine(s);
        return PyTypes::float3({(float) r.x, (float) r.y, (float) r.z});
    };
    
    RodIC::MappingFunc1D ymrTorsion = [&](float s)
    {
        return (float) torsion(s);;
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
    
    auto energies = computeBendingEnergies(pos.data(), nSegments, kBending, omegaEq);

    real h = 1.0 / nSegments;
    real err = 0;
    
    for (int i = 0; i < nSegments - 1; ++i)
    {
        real s = (i+1) * h;
        auto eRef = ref(s);
        auto eSim = energies[i];
        auto de = eSim - eRef;
        // printf("%g %g\n", eRef, eSim);
        err += de * de;
    }

    return sqrt(err / nSegments);
}


TEST (ROD, energies_bending)
{
    real L = 5.0;

    real3 kBending {1.0, 0.0, 1.0};
    real2 omegaEq {0.1, 0.0};
    
    auto centerLine = [&](real s) -> real3
    {
        return {(s-0.5) * L, 0., 0.};
    };

    auto torsion = [](real s) -> real {return 0.0;};
    
    auto analyticEnergy = [&](real s) -> real
    {
        real2 Bo = symmetricMatMult(kBending, omegaEq);
        return 0.5 * dot(Bo, omegaEq);
    };

    std::vector<int> nsegs = {8, 16, 32, 64, 128};
    
    for (auto n : nsegs)
    {
        auto err = checkBendingEnergy(MPI_COMM_WORLD, centerLine, torsion, n,
                                      kBending, omegaEq, analyticEnergy);

        //printf("%d %g\n", n, err);
        ASSERT_LE(err, 1e-6);
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
