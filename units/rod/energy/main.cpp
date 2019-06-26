#include <core/initial_conditions/rod.h>
#include <core/interactions/rod.h>
#include <core/logger.h>
#include <core/pvs/rod_vector.h>
#include <core/utils/cuda_common.h>
#include <core/utils/helper_math.h>
#include <core/utils/quaternion.h>

#include <vector>
#include <functional>
#include <numeric>
#include <gtest/gtest.h>

Logger logger;

using real  = double;
using real2 = double2;
using real3 = double3;
using real4 = double4;

static real2 make_real2(float2 v) { return {(real) v.x, (real) v.y}; }
static real3 make_real3(float3 v) { return {(real) v.x, (real) v.y, (real) v.z}; }
static real3 make_real3(float4 v) { return {(real) v.x, (real) v.y, (real) v.z}; }

static float2 make_float2(real2 v) {return {(float) v.x, (float) v.y}; }

using CenterLineFunc = std::function<real3(real)>;
using EnergyFunc     = std::function<real(real)>;
using TorsionFunc    = std::function<real(real)>;

constexpr float a = 0.05f;
constexpr float dt = 0.f;

enum class EnergyMode {Density, Absolute};
enum class CheckMode {Detail, Total};

inline real2 symmetricMatMult(const real3& A, const real2& x)
{
    return {A.x * x.x + A.y * x.y,
            A.y * x.x + A.z * x.y};
}

template <EnergyMode Emode>
static std::vector<real> computeBendingEnergies(const float4 *positions, int nSegments, real3 kBending, real2 kappaEq)
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
        real l = 0.5 * (le0 + le1);
        real linv = 1.0 / l;
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
    
        real2 kappa0 { +dpPerp0inv * dot(bicur, t0_dp0),
                       -dpPerp0inv * dot(bicur,    dp0)};

        real2 kappa1 { +dpPerp1inv * dot(bicur, t1_dp1),
                       -dpPerp1inv * dot(bicur,    dp1)};

        real2 dkappa0 = kappa0 * linv - kappaEq;
        real2 dkappa1 = kappa1 * linv - kappaEq;

        real2 Bkappa0 = symmetricMatMult(kBending, dkappa0);
        real2 Bkappa1 = symmetricMatMult(kBending, dkappa1);

        // integrated energy
        // 0.25: in Bergou & al, l = e1 + e2; here l = (e1 + e2) / 2
        real Eb = 0.25 * l * (dot(dkappa0, Bkappa0) + dot(dkappa1, Bkappa1));

        if (Emode == EnergyMode::Density)
            Eb *= linv;
        
        energies.push_back(Eb);
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

template <EnergyMode Emode>
static std::vector<real> computeTwistEnergies(const float4 *positions, int nSegments, real kTwist, real tauEq)
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

        real4  Q = getQfrom(t0, t1);
        real3 u0 = normalize(anyOrthogonal(t0));
        real3 u1 = normalize(rotate(u0, Q));

        auto dp0 = pp0 - pm0;
        auto dp1 = pp1 - pm1;

        real le0 = length(e0);
        real le1 = length(e1);
        auto l    = 0.5 * (le0 + le1);
        auto linv = 1.0 / l;

        auto v0 = cross(t0, u0);
        auto v1 = cross(t1, u1);

        real dpu0 = dot(dp0, u0);
        real dpv0 = dot(dp0, v0);

        real dpu1 = dot(dp1, u1);
        real dpv1 = dot(dp1, v1);

        real theta0 = atan2(dpv0, dpu0);
        real theta1 = atan2(dpv1, dpu1);
    
        real tau  = safeDiffTheta(theta0, theta1) * linv;
        real dTau = tau - tauEq;

        // integrated twist energy
        // 0.5: in Bergou & al, l = e1 + e2; here l = (e1 + e2) / 2
        real Et = 0.5 * l * dTau * dTau * kTwist;

        if (Emode == EnergyMode::Density)
            Et *= linv;

        energies.push_back(Et);
    }
    return energies;
}

template <CheckMode checkMode>
static real checkBendingEnergy(const MPI_Comm& comm, CenterLineFunc centerLine, TorsionFunc torsion, int nSegments,
                               real3 kBending, real2 kappaEq, EnergyFunc ref, real EtotRef)
{
    RodIC::MappingFunc3D mirCenterLine = [&](float s)
    {
        auto r = centerLine(s);
        return PyTypes::float3({(float) r.x, (float) r.y, (float) r.z});
    };
    
    RodIC::MappingFunc1D mirTorsion = [&](float s)
    {
        return (float) torsion(s);;
    };

    DomainInfo domain;
    float L = 32.f;
    domain.globalSize  = {L, L, L};
    domain.globalStart = {0.f, 0.f, 0.f};
    domain.localSize   = {L, L, L};
    float mass = 1.f;
    MirState state(domain, dt);
    RodVector rv(&state, "rod", mass, nSegments);
    
    RodIC ic({{L/2, L/2, L/2, 1.0f, 0.0f, 0.0f}},
             mirCenterLine, mirTorsion, a);
    
    ic.exec(comm, &rv, defaultStream);

    auto& pos = rv.local()->positions();

    real err = 0;
    if (checkMode == CheckMode::Detail)
    {
        auto energies = computeBendingEnergies<EnergyMode::Density>
            (pos.data(), nSegments, kBending, kappaEq);

        real h = 1.0 / nSegments;
    
        for (int i = 0; i < nSegments - 1; ++i)
        {
            real s = (i+1) * h;
            auto eRef = ref(s);
            auto eSim = energies[i];
            auto de = eSim - eRef;
            // printf("%g %g\n", eRef, eSim);
            err += de * de;
        }
        err = sqrt(err / nSegments);
    }
    else
    {
        auto energies = computeBendingEnergies<EnergyMode::Absolute>
            (pos.data(), nSegments, kBending, kappaEq);

        auto EtotSim = std::accumulate(energies.begin(), energies.end(), 0.0);
        // printf("%g %g\n", EtotRef, EtotSim);
        err = fabs(EtotSim - EtotRef);
    }

    return err;
}

static real checkGPUBendingEnergy(const MPI_Comm& comm, CenterLineFunc centerLine, TorsionFunc torsion, int nSegments,
                                  real3 kBending, real2 kappaEq)
{
    RodIC::MappingFunc3D mirCenterLine = [&](float s)
    {
        auto r = centerLine(s);
        return PyTypes::float3({(float) r.x, (float) r.y, (float) r.z});
    };
    
    RodIC::MappingFunc1D mirTorsion = [&](float s)
    {
        return (float) torsion(s);;
    };

    DomainInfo domain;
    float L = 32.f;
    domain.globalSize  = {L, L, L};
    domain.globalStart = {0.f, 0.f, 0.f};
    domain.localSize   = {L, L, L};
    float mass = 1.f;
    MirState state(domain, dt);
    RodVector rv(&state, "rod", mass, nSegments);
    
    RodIC ic({{L/2, L/2, L/2, 1.0f, 0.0f, 0.0f}},
             mirCenterLine, mirTorsion, a);

    RodParameters params;
    params.kBending = make_float3(kBending);
    params.kappaEq  = {make_float2(kappaEq)};
    params.kTwist   = 0.f;
    params.tauEq    = {0.f};
    params.groundE  = {0.f};
    params.l0       = 0.f;
    params.a0       = 0.f;
    params.ksCenter = 0.f;
    params.ksFrame  = 0.f;
    InteractionRod gpuInt(&state, "twist_forces", params, StatesParametersNone{}, true);
    gpuInt.setPrerequisites(&rv, &rv, nullptr, nullptr);
    ic.exec(comm, &rv, defaultStream);
    
    auto& pos = rv.local()->positions();

    rv.local()->forces().clear(defaultStream);
    gpuInt.local(&rv, &rv, nullptr, nullptr, defaultStream);

    auto& gpuEnergies = *rv.local()->dataPerBisegment.getData<float>(ChannelNames::energies);
    gpuEnergies.downloadFromDevice(defaultStream);

    auto  cpuEnergies = computeBendingEnergies<EnergyMode::Absolute>
        (pos.data(), nSegments, kBending, kappaEq);

    real err = 0;

    for (int i = 0; i < nSegments - 1; ++i)
    {
        real gpuE = gpuEnergies[i];
        real cpuE = cpuEnergies[i];
        // printf("%g %g\n", cpuE, gpuE);
        auto dE = fabs(cpuE - gpuE);
        err = std::max(err, dE);
    }
    
    return err;
}



template <CheckMode checkMode>
static real checkTwistEnergy(const MPI_Comm& comm, CenterLineFunc centerLine, TorsionFunc torsion, int nSegments,
                             real kTwist, real tauEq, EnergyFunc ref, real EtotRef)
{
    RodIC::MappingFunc3D mirCenterLine = [&](float s)
    {
        auto r = centerLine(s);
        return PyTypes::float3({(float) r.x, (float) r.y, (float) r.z});
    };
    
    RodIC::MappingFunc1D mirTorsion = [&](float s)
    {
        return (float) torsion(s);;
    };

    DomainInfo domain;
    float L = 32.f;
    domain.globalSize  = {L, L, L};
    domain.globalStart = {0.f, 0.f, 0.f};
    domain.localSize   = {L, L, L};
    float mass = 1.f;
    MirState state(domain, dt);
    RodVector rv(&state, "rod", mass, nSegments);
    
    RodIC ic({{L/2, L/2, L/2, 1.0f, 0.0f, 0.0f}},
             mirCenterLine, mirTorsion, a);
    
    ic.exec(comm, &rv, defaultStream);

    auto& pos = rv.local()->positions();

    real err = 0;
    if (checkMode == CheckMode::Detail)
    {
        auto energies = computeTwistEnergies<EnergyMode::Density>
            (pos.data(), nSegments, kTwist, tauEq);

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

        err = sqrt(err / nSegments);
    }
    else
    {
        auto energies = computeTwistEnergies<EnergyMode::Absolute>
            (pos.data(), nSegments, kTwist, tauEq);

        auto EtotSim = std::accumulate(energies.begin(), energies.end(), 0.0);
        err = fabs(EtotRef - EtotSim);
    }

    return err;
}

static real checkGPUTwistEnergy(const MPI_Comm& comm, CenterLineFunc centerLine, TorsionFunc torsion, int nSegments,
                                real kTwist, real tauEq)
{
    RodIC::MappingFunc3D mirCenterLine = [&](float s)
    {
        auto r = centerLine(s);
        return PyTypes::float3({(float) r.x, (float) r.y, (float) r.z});
    };
    
    RodIC::MappingFunc1D mirTorsion = [&](float s)
    {
        return (float) torsion(s);;
    };

    DomainInfo domain;
    float L = 32.f;
    domain.globalSize  = {L, L, L};
    domain.globalStart = {0.f, 0.f, 0.f};
    domain.localSize   = {L, L, L};
    float mass = 1.f;
    MirState state(domain, dt);
    RodVector rv(&state, "rod", mass, nSegments);
    
    RodIC ic({{L/2, L/2, L/2, 1.0f, 0.0f, 0.0f}},
             mirCenterLine, mirTorsion, a);
    
    RodParameters params;
    params.kBending = make_float3(0.f);
    params.kappaEq  = {make_float2(0.f)};
    params.kTwist   = (float) kTwist;
    params.tauEq    = {(float) tauEq};
    params.groundE  = {0.f};
    params.l0       = 0.f;
    params.a0       = 0.f;
    params.ksCenter = 0.f;
    params.ksFrame  = 0.f;
    InteractionRod gpuInt(&state, "twist_forces", params, StatesParametersNone{}, true);
    gpuInt.setPrerequisites(&rv, &rv, nullptr, nullptr);
    ic.exec(comm, &rv, defaultStream);

    auto& pos = rv.local()->positions();

    rv.local()->forces().clear(defaultStream);
    gpuInt.local(&rv, &rv, nullptr, nullptr, defaultStream);

    auto& gpuEnergies = *rv.local()->dataPerBisegment.getData<float>(ChannelNames::energies);
    gpuEnergies.downloadFromDevice(defaultStream);

    auto cpuEnergies = computeTwistEnergies<EnergyMode::Absolute>
        (pos.data(), nSegments, kTwist, tauEq);

    real err = 0;

    for (int i = 0; i < nSegments - 1; ++i)
    {
        real gpuE = gpuEnergies[i];
        real cpuE = cpuEnergies[i];
        // printf("%g %g\n", cpuE, gpuE);
        auto dE = fabs(cpuE - gpuE);
        err = std::max(err, dE);
    }
    
    return err;
}



TEST (ROD, cpu_energies_bending)
{
    real L = 5.0;

    real3 kBending {1.0, 0.0, 1.0};
    real2 kappaEq {0.1, 0.0};
    
    auto centerLine = [&](real s) -> real3
    {
        return {(s-0.5) * L, 0., 0.};
    };

    auto torsion = [](real s) -> real {return 0.0;};
    
    auto analyticEnergy = [&](real s) -> real
    {
        real2 Bo = symmetricMatMult(kBending, kappaEq);
        return 0.5 * dot(Bo, kappaEq);
    };

    real2 Bo = symmetricMatMult(kBending, kappaEq);
    real EtotRef = 0.5 * dot(Bo, kappaEq) * L;
    
    std::vector<int> nsegs = {8, 16, 32, 64, 128, 256, 512};
    
    for (auto n : nsegs)
    {
        auto err = checkBendingEnergy<CheckMode::Detail>(MPI_COMM_WORLD, centerLine, torsion, n,
                                                         kBending, kappaEq, analyticEnergy, EtotRef);

        // printf("%d %g\n", n, err);
        ASSERT_LE(err, 1e-6);
    }
}

TEST (ROD, cpu_energies_bending_circle)
{
    real R = 1.5;
    
    real3 kBending {1.0, 0.0, 1.0};
    real2 kappaEq {0., 0.};
    
    auto centerLine = [&](real s) -> real3
    {
        real t = 2 * M_PI * s;
        return {R * cos(t), R * sin(t), 0.0};
    };

    auto torsion = [](real s) -> real {return 0.0;};
    
    auto analyticEnergy = [&](real s) -> real
    {
        real2 dOm = real2{1/R, 0.0} - kappaEq;
        real2 Bo = symmetricMatMult(kBending, dOm);
        return 0.5 * dot(Bo, dOm);
    };

    real2 dOm = real2{1/R, 0.0} - kappaEq;
    real2 Bo = symmetricMatMult(kBending, dOm);
    real EtotRef = 0.5 * dot(Bo, dOm) * 2 * M_PI * R;
    
    std::vector<int> nsegs = {8, 16, 32, 64, 128};
    std::vector<real> errors;
    
    for (auto n : nsegs)
        errors.push_back(checkBendingEnergy<CheckMode::Detail>(MPI_COMM_WORLD, centerLine, torsion, n,
                                                               kBending, kappaEq, analyticEnergy, EtotRef));

    // check convergence rate
    const real rateTh = 2;

    for (int i = 0; i < nsegs.size() - 1; ++i)
    {
        real e0 = errors[i], e1 = errors[i+1];
        int  n0 =  nsegs[i], n1 =  nsegs[i+1];

        real rate = (log(e0) - log(e1)) / (log(n1) - log(n0));

        // printf("%g\n", rate);
        ASSERT_LE(fabs(rate-rateTh), 1e-1);
    }
}

TEST (ROD, gpu_energies_bending_circle)
{
    real R = 1.5;
    
    real3 kBending {1.0, 0.0, 1.0};
    real2 kappaEq {0., 0.};
    
    auto centerLine = [&](real s) -> real3
    {
        real t = 2 * M_PI * s;
        return {R * cos(t), R * sin(t), 0.0};
    };

    auto torsion = [](real s) -> real {return 0.0;};
        
    std::vector<int> nsegs = {8, 16, 32, 64, 128};
    // std::vector<int> nsegs = {32};
    
    for (auto n : nsegs)
    {
        auto err = checkGPUBendingEnergy(MPI_COMM_WORLD, centerLine, torsion, n,
                                         kBending, kappaEq);
        // printf("%d %g\n", n, err);
        ASSERT_LE(err, 1e-6);
    }
}

TEST (ROD, cpu_energies_twist)
{
    real L = 5.0;

    real tau0  {0.1};
    real kTwist {1.0};
    real tauEq  {0.3};
    
    auto centerLine = [&](real s) -> real3
    {
        return {(s-0.5) * L, 0., 0.};
    };

    auto torsion = [&](real s) -> real {return tau0;};
    
    auto analyticEnergy = [&](real s) -> real
    {
        real dTau = tau0 - tauEq;
        return 0.5 * kTwist * dTau * dTau;
    };

    real dTau = tau0 - tauEq;
    real EtotRef = L * 0.5 * kTwist * dTau * dTau;
    
    std::vector<int> nsegs = {8, 16, 32, 64, 128};

    for (auto n : nsegs)
    {
        auto err = checkTwistEnergy<CheckMode::Detail>(MPI_COMM_WORLD, centerLine, torsion, n,
                                                       kTwist, tauEq, analyticEnergy, EtotRef);

        // printf("%d %g\n", n, err);
        ASSERT_LE(err, 1e-6);
    }
}

TEST (ROD, gpu_energies_twist)
{
    real L = 5.0;

    real tau0  {0.1};
    real kTwist {1.0};
    real tauEq  {0.3};
    
    auto centerLine = [&](real s) -> real3
    {
        return {(s-0.5) * L, 0., 0.};
    };

    auto torsion = [&](real s) -> real {return tau0;};
    
    
    std::vector<int> nsegs = {8, 16, 32, 64, 128};

    for (auto n : nsegs)
    {
        auto err = checkGPUTwistEnergy(MPI_COMM_WORLD, centerLine, torsion, n,
                                       kTwist, tauEq);

        // printf("%d %g\n", n, err);
        ASSERT_LE(err, 1e-6);
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    logger.init(MPI_COMM_WORLD, "rod_energy.log", 9);
    
    testing::InitGoogleTest(&argc, argv);
    auto ret = RUN_ALL_TESTS();

    MPI_Finalize();
    return ret;
}
