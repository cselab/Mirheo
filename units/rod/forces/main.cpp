#include <mirheo/core/interactions/rod/base_rod.h>
#include <mirheo/core/interactions/rod/factory.h>
#include <mirheo/core/logger.h>
#include <mirheo/core/pvs/rod_vector.h>
#include <mirheo/core/utils/helper_math.h>
#include <mirheo/core/utils/quaternion.h>

#include <vector>
#include <functional>
#include <gtest/gtest.h>

using namespace mirheo;

#define FMT "%+6e"
#define SEP "\t"
#define EXPAND(v) v.x, v.y, v.z

using Real  = double;
using Real2 = double2;
using Real3 = double3;
using Real4 = double4;

static Real2 make_Real2(real2 v) { return {(Real) v.x, (Real) v.y}; }
static Real3 make_Real3(real3 v) { return {(Real) v.x, (Real) v.y, (Real) v.z}; }

using CenterLineFunc = std::function<Real3(Real)>;

static void initialFlagellum(int n, std::vector<Real3>& positions, CenterLineFunc centerLine)
{
    positions.resize(5 * n + 1);
    Real h = 1.0 / n;

    for (int i = 0; i < n; ++i) {
        Real3 r = centerLine(i*h);

        positions[i * 5 + 0] = r;
        positions[i * 5 + 1] = r;
        positions[i * 5 + 2] = r;
        positions[i * 5 + 3] = r;
        positions[i * 5 + 4] = r;
    }

    positions[5*n] = centerLine(1.f);
}

static void getTransformation(Real3 t0, Real3 t1, Quaternion<Real>& Q)
{
    Q = Quaternion<Real>::createFromVectors(t0, t1);
    auto t0t1 = cross(t0, t1);
    if (length(t0t1) > 1e-6)
        t0t1 = normalize(t0t1);

    Real err_t0_t1   = length(t1 - Q.rotate(t0));
    Real err_t01_t01 = length(t0t1 - Q.rotate(t0t1));

    ASSERT_LE(err_t01_t01, 1e-6f);
    ASSERT_LE(err_t0_t1, 1e-6);
}

static void initialFrame(Real3 t0, Real3& u, Real3& v)
{
    t0 = normalize(t0);
    u = anyOrthogonal(t0);
    u = normalize(u);
    v = normalize(cross(t0, u));
}

static void transportBishopFrame(const std::vector<Real3>& positions, std::vector<Real3>& frames)
{
    int n = (positions.size() - 1) / 5;

    for (int i = 1; i < n; ++i)
    {
        auto r0 = positions[5*(i-1)];
        auto r1 = positions[5*(i)];
        auto r2 = positions[5*(i+1)];

        auto t0 = normalize(r1-r0);
        auto t1 = normalize(r2-r1);

        Quaternion<Real> Q;
        getTransformation(t0, t1, Q);
        auto u0 = frames[2*(i-1) + 0];
        auto u1 = Q.rotate(u0);
        auto v1 = cross(t1, u1);
        frames[2*i + 0] = u1;
        frames[2*i + 1] = v1;
    }
}

static Real bendingEnergy(const std::vector<Real3>& positions, const real2 B[2], real2 omega_eq)
{
    int n = (positions.size() - 1) / 5;

    Real Etot = 0;

    for (int i = 1; i < n; ++i)
    {
        auto r0 = positions[5*(i-1)];
        auto r1 = positions[5*(i)];
        auto r2 = positions[5*(i+1)];

        auto e0 = r1-r0;
        auto e1 = r2-r1;

        auto t0 = normalize(e0);
        auto t1 = normalize(e1);

        auto dp0 = positions[5*(i-1) + 2] - positions[5*(i-1) + 1];
        auto dp1 = positions[5*i     + 2] - positions[5*i     + 1];

        auto dp0Perp = dp0 - dot(dp0, t0) * t0;
        auto dp1Perp = dp1 - dot(dp1, t1) * t1;

        Real denom = length(e0) * length(e1) + dot(e0, e1);
        auto bicur = (2.f / denom) * cross(e0, e1);

        Real dp0Perpinv = 1.0 / length(dp0Perp);
        Real dp1Perpinv = 1.0 / length(dp1Perp);

        Real l = 0.5 * (length(e0) + length(e1));
        Real linv = 1.0 / l;

        Real2 om0 = {+ linv * dp0Perpinv * dot(bicur, cross(t0, dp0)),
                     - linv * dp0Perpinv * dot(bicur, dp0)};
        Real2 om1 = {+ linv * dp1Perpinv * dot(bicur, cross(t1, dp1)),
                     - linv * dp1Perpinv * dot(bicur, dp1)};


        om0 -= make_Real2(omega_eq);
        om1 -= make_Real2(omega_eq);

        Real2 Bom0 {dot(om0, make_Real2(B[0])),
                    dot(om0, make_Real2(B[1]))};

        Real2 Bom1 {dot(om1, make_Real2(B[0])),
                    dot(om1, make_Real2(B[1]))};

        Real E = 0.25 * l * (dot(Bom0, om0) + dot(Bom1, om1));
        Etot += E;
    }

    return Etot;
}

inline Real safeDiffTheta(Real t0, Real t1)
{
    auto dth = t1 - t0;
    if (dth >  M_PI) dth -= 2.0 * M_PI;
    if (dth < -M_PI) dth += 2.0 * M_PI;
    return dth;
}

static Real twistEnergy(const std::vector<Real3>& positions, Real kTwist, Real tau0)
{
    int n = (positions.size() - 1) / 5;

    Real Etot = 0;

    for (int i = 1; i < n; ++i)
    {
        auto r0 = positions[5*(i-1)];
        auto r1 = positions[5*(i)];
        auto r2 = positions[5*(i+1)];

        auto dp0 = positions[5*(i-1) + 2] - positions[5*(i-1) + 1];
        auto dp1 = positions[5*i     + 2] - positions[5*i     + 1];

        auto e0 = r1-r0;
        auto e1 = r2-r1;

        auto t0 = normalize(e0);
        auto t1 = normalize(e1);

        auto  Q = Quaternion<Real>::createFromVectors(t0, t1);
        auto u0 = normalize(anyOrthogonal(t0));
        auto u1 = normalize(Q.rotate(u0));

        auto v0 = cross(t0, u0);
        auto v1 = cross(t1, u1);

        auto l = 0.5 * (length(e0) + length(e1));

        auto theta0 = atan2(dot(dp0, v0), dot(dp0, u0));
        auto theta1 = atan2(dot(dp1, v1), dot(dp1, u1));

        auto tau = safeDiffTheta(theta0, theta1) / l;
        auto dtau = tau - tau0;

        auto E = 0.5 * kTwist * l * dtau * dtau;

        Etot += E;
    }

    return Etot;
}

static Real smoothingEnergy(const std::vector<Real3>& positions, Real kSmoothing)
{
    int n = (positions.size() - 1) / 5;
    int nBisegments = n - 1;

    std::vector<Real>  taus  (nBisegments);
    std::vector<Real2> omegas(nBisegments);

    for (int i = 1; i < n; ++i)
    {
        auto r0 = positions[5*(i-1)];
        auto r1 = positions[5*(i)];
        auto r2 = positions[5*(i+1)];

        auto e0 = r1-r0;
        auto e1 = r2-r1;

        auto t0 = normalize(e0);
        auto t1 = normalize(e1);

        auto dp0 = positions[5*(i-1) + 2] - positions[5*(i-1) + 1];
        auto dp1 = positions[5*i     + 2] - positions[5*i     + 1];

        auto dp0Perp = dp0 - dot(dp0, t0) * t0;
        auto dp1Perp = dp1 - dot(dp1, t1) * t1;

        Real denom = length(e0) * length(e1) + dot(e0, e1);
        auto bicur = (2.f / denom) * cross(e0, e1);

        Real dp0Perpinv = 1.0 / length(dp0Perp);
        Real dp1Perpinv = 1.0 / length(dp1Perp);

        auto  Q = Quaternion<Real>::createFromVectors(t0, t1);
        auto u0 = normalize(anyOrthogonal(t0));
        auto u1 = normalize(Q.rotate(u0));

        auto v0 = cross(t0, u0);
        auto v1 = cross(t1, u1);

        auto theta0 = atan2(dot(dp0, v0), dot(dp0, u0));
        auto theta1 = atan2(dot(dp1, v1), dot(dp1, u1));

        Real l = 0.5 * (length(e0) + length(e1));
        Real linv = 1.0 / l;

        Real2 om0 = {+ linv * dp0Perpinv * dot(bicur, cross(t0, dp0)),
                     - linv * dp0Perpinv * dot(bicur, dp0)};
        Real2 om1 = {+ linv * dp1Perpinv * dot(bicur, cross(t1, dp1)),
                     - linv * dp1Perpinv * dot(bicur, dp1)};

        auto tau = safeDiffTheta(theta0, theta1) / l;

        omegas[i-1] = 0.5 * (om0 + om1);
        taus  [i-1] = tau;
    }

    Real Etot = 0;

    for (int i = 1; i < n-1; ++i)
    {
        auto r0 = positions[5*(i-1)];
        auto r1 = positions[5*(i  )];
        auto l = length(r1-r0);

        auto dtau   = taus  [i] - taus  [i-1];
        auto domega = omegas[i] - omegas[i-1];

        auto E = 0.5 * kSmoothing * l * (domega.x * domega.x +
                                         domega.y * domega.y +
                                         dtau     * dtau);

        // auto E = 0.5 * kSmoothing * l * (dtau     * dtau);

        Etot += E;
    }

    return Etot;
}

template <typename EnergyComp>
inline void computeForces(const std::vector<Real3>& positions, std::vector<Real3>& forces, Real h,
                          EnergyComp computeEnergy)
{
    auto perturbed = positions;

    for (size_t i = 0; i < positions.size(); ++i)
    {
        auto computeForce = [&](Real3 dir) {
            const auto r = positions[i];
            perturbed[i] = r + (h/2) * dir;
            auto Ep = computeEnergy(perturbed);
            perturbed[i] = r - (h/2) * dir;
            auto Em = computeEnergy(perturbed);
            perturbed[i] = r;
            return - (Ep - Em) / h;
        };

        forces[i].x = computeForce({1.0, 0.0, 0.0});
        forces[i].y = computeForce({0.0, 1.0, 0.0});
        forces[i].z = computeForce({0.0, 0.0, 1.0});
    }
}

static void bendingForces(Real h, const real2 B[2], real2 omega_eq,
                          const std::vector<Real3>& positions,
                          std::vector<Real3>& forces)
{
    return computeForces(positions, forces, h, [&](const auto& positions)
    {
        return bendingEnergy(positions, B, omega_eq);
    });
}

static void twistForces(Real h, real kt, real tau0,
                        const std::vector<Real3>& positions,
                        std::vector<Real3>& forces)
{
    return computeForces(positions, forces, h, [&](const auto& positions)
    {
        return twistEnergy(positions, kt, tau0);
    });
}

inline void smoothingForces(Real h, real kbi,
                            const std::vector<Real3>& positions,
                            std::vector<Real3>& forces)
{
    return computeForces(positions, forces, h, [&](const auto& positions)
    {
        return smoothingEnergy(positions, kbi);
    });
}

static void setCrosses(const std::vector<Real3>& frames, std::vector<Real3>& positions)
{
    int n = (positions.size() - 1) / 5;
    for (int i = 0; i < n; ++i)
    {
        auto u = frames[2*i+0];
        auto v = frames[2*i+1];
        auto r0 = positions[5*i+0];
        auto r1 = positions[5*i+5];
        auto dr = 0.5f * (r1 - r0);
        Real a = length(dr);
        auto c = 0.5f * (r0 + r1);

        positions[i*5+1] = c - a * u;
        positions[i*5+2] = c + a * u;
        positions[i*5+3] = c - a * v;
        positions[i*5+4] = c + a * v;
    }
}

template <class CenterLine>
static void initializeRef(CenterLine centerLine, int nSegments, std::vector<Real3>& positions, std::vector<Real3>& frames)
{
    initialFlagellum(nSegments, positions, centerLine);

    frames.resize(2*nSegments);
    initialFrame(positions[5]-positions[0],
                 frames[0], frames[1]);

    transportBishopFrame(positions, frames);
    setCrosses(frames, positions);
}

static void copyToRv(const std::vector<Real3>& positions, RodVector& rod)
{
    auto& pos = rod.local()->positions ();
    auto& vel = rod.local()->velocities();

    for (size_t i = 0; i < positions.size(); ++i)
    {
        Particle p;
        p.r = make_real3(positions[i]);
        p.u = make_real3(0);
        p.setId(i);
        pos[i] = p.r2Real4();
        vel[i] = p.u2Real4();
    }
    pos.uploadToDevice(defaultStream);
    vel.uploadToDevice(defaultStream);
}

static void checkMomentum(const PinnedBuffer<real4>& pos, const HostBuffer<Force>& forces)
{
    double3 totForce  {0., 0., 0.};
    double3 totTorque {0., 0., 0.};

    for (size_t i = 0; i < forces.size(); ++i)
    {
        auto r4 = pos[i];
        auto f4 = forces[i].f;
        double3 r {(double) r4.x, (double) r4.y, (double) r4.z};
        double3 f {(double) f4.x, (double) f4.y, (double) f4.z};

        totForce  += f;
        totTorque += cross(r, f);
    }

    // printf(FMT SEP FMT SEP FMT SEP SEP
    //        FMT SEP FMT SEP FMT "\n",
    //        EXPAND(totForce), EXPAND(totTorque));

    ASSERT_LE(length(totForce),  5e-6);
    ASSERT_LE(length(totTorque), 5e-6);
}

template <class CenterLine>
static double testTwistForces(real kt, real tau0, CenterLine centerLine, int nSegments, Real h)
{
    MirState state(DomainInfo(), 0.f, UnitConversion{});

    RodParameters params;
    params.kBending = {0.f, 0.f, 0.f};
    params.kappaEq  = {{0.f, 0.f}};
    params.kTwist   = kt;
    params.tauEq    = {tau0};
    params.groundE  = {0.f};
    params.a0       = 0.f;
    params.l0       = 0.f;
    params.ksCenter = 0.f;
    params.ksFrame  = 0.f;

    std::vector<Real3> refPositions, refFrames, refForces;
    RodVector rod(&state, "rod", 1.f, nSegments, 1);
    auto interactions = createInteractionRod(&state, "rod_interaction", params, StatesParametersNone{}, false);
    initializeRef(centerLine, nSegments, refPositions, refFrames);
    copyToRv(refPositions, rod);


    refForces.resize(refPositions.size());
    twistForces(h, kt, tau0, refPositions, refForces);

    rod.local()->forces().clear(defaultStream);
    interactions->setPrerequisites(&rod, &rod, nullptr, nullptr);
    interactions->local(&rod, &rod, nullptr, nullptr, defaultStream);

    HostBuffer<Force> forces;
    forces.copy(rod.local()->forces(), defaultStream);
    CUDA_Check( cudaDeviceSynchronize() );

    double Linfty = 0;
    for (size_t i = 0; i < refForces.size(); ++i)
    {
        Real3 a = refForces[i];
        Real3 b = make_Real3(forces[i].f);
        Real3 diff = a - b;
        double err = std::max(std::max(math::abs(diff.x), math::abs(diff.y)), math::abs(diff.z));

        // if ((i % 5) == 0) printf("%03d ---------- \n", i/5);
        // if ((i % 5) == 0)
        //     printf(FMT SEP FMT SEP FMT SEP SEP
        //            FMT SEP FMT SEP FMT SEP SEP
        //            FMT SEP FMT "\n",
        //            a.x, a.y, a.z,
        //            b.x, b.y, b.z,
        //            length(a), length(b));

        Linfty = std::max(Linfty, err);
    }

    checkMomentum(rod.local()->positions(), forces);

    return Linfty;
}

template <class CenterLine>
static double testBendingForces(real3 B, real2 kappa, CenterLine centerLine, int nSegments, Real h)
{
    MirState state(DomainInfo(), 0.f, UnitConversion{});

    RodParameters params;
    params.kBending = B;
    params.kappaEq  = {kappa};
    params.kTwist   = 0.f;
    params.tauEq    = {0.f};
    params.groundE  = {0.f};
    params.a0       = 0.f;
    params.l0       = 0.f;
    params.ksCenter = 0.f;
    params.ksFrame  = 0.f;

    std::vector<Real3> refPositions, refFrames, refForces;
    RodVector rod(&state, "rod", 1.f, nSegments, 1);
    auto interactions = createInteractionRod(&state, "rod_interaction", params, StatesParametersNone{}, false);
    initializeRef(centerLine, nSegments, refPositions, refFrames);
    copyToRv(refPositions, rod);


    refForces.resize(refPositions.size());
    const real2 B_[2] {{B.x, B.y}, {B.y, B.z}};
    bendingForces(h, B_, kappa, refPositions, refForces);

    rod.local()->forces().clear(defaultStream);
    interactions->setPrerequisites(&rod, &rod, nullptr, nullptr);
    interactions->local(&rod, &rod, nullptr, nullptr, defaultStream);

    HostBuffer<Force> forces;
    forces.copy(rod.local()->forces(), defaultStream);
    CUDA_Check( cudaDeviceSynchronize() );

    double Linfty = 0;
    for (size_t i = 0; i < refForces.size(); ++i)
    {
        Real3 a = refForces[i];
        Real3 b = make_Real3(forces[i].f);
        Real3 diff = a - b;
        double err = std::max(std::max(math::abs(diff.x), math::abs(diff.y)), math::abs(diff.z));

        // if ((i % 5) == 0) printf("%03d ---------- \n", i/5);
        // if ((i % 5) == 0)
        //     printf(FMT SEP FMT SEP FMT SEP SEP
        //            FMT SEP FMT SEP FMT SEP SEP
        //            FMT SEP FMT "\n",
        //            EXPAND(a), EXPAND(b),
        //            length(a), length(b));

        Linfty = std::max(Linfty, err);
    }

    checkMomentum(rod.local()->positions(), forces);

    return Linfty;
}

template <class CenterLine>
static double testSmoothingForces(real kSmoothing, CenterLine centerLine, int nSegments, Real h)
{
    MirState state(DomainInfo(), 0.f);

    RodParameters params;
    params.kBending = {0.f, 0.f, 0.f};
    params.kappaEq  = {{0.f, 0.f}, {0.f, 0.f}};
    params.kTwist   = 0.f;
    params.tauEq    = {0.f, 0.f};
    params.groundE  = {0.f, 0.f};
    params.a0       = 0.f;
    params.l0       = 0.f;
    params.ksCenter = 0.f;
    params.ksFrame  = 0.f;

    StatesSmoothingParameters stateParams;
    stateParams.kSmoothing = kSmoothing;

    std::vector<Real3> refPositions, refFrames, refForces;
    RodVector rod(&state, "rod", 1.f, nSegments, 1);
    auto interactions = createInteractionRod(&state, "rod_interaction", params, stateParams, false);
    initializeRef(centerLine, nSegments, refPositions, refFrames);
    copyToRv(refPositions, rod);

    refForces.resize(refPositions.size());
    smoothingForces(h, kSmoothing, refPositions, refForces);

    rod.local()->forces().clear(defaultStream);
    interactions->setPrerequisites(&rod, &rod, nullptr, nullptr);
    interactions->local(&rod, &rod, nullptr, nullptr, defaultStream);

    HostBuffer<Force> forces;
    forces.copy(rod.local()->forces(), defaultStream);
    CUDA_Check( cudaDeviceSynchronize() );

    // FILE * f = fopen("tmp", "w");
    double Linfty = 0;
    for (int i = 0; i < refForces.size(); ++i)
    {
        Real3 a = refForces[i];
        Real3 b = make_Real3(forces[i].f);
        Real3 diff = a - b;
        double err = std::max(std::max(math::abs(diff.x), math::abs(diff.y)), math::abs(diff.z));

        // if ((i % 5) == 0) printf("%03d ---------- \n", i/5);
        // if ((i % 5) == 0)
        //     fprintf(f,
        //             FMT SEP FMT SEP FMT SEP SEP
        //             FMT SEP FMT SEP FMT SEP SEP
        //             FMT SEP FMT "\n",
        //             EXPAND(a), EXPAND(b),
        //             length(a), length(b));

        Linfty = std::max(Linfty, err);
    }
    // fclose(f);
    checkMomentum(rod.local()->positions(), forces);

    return Linfty;
}


TEST (ROD, twistForces_straight)
{
    Real height = 5.0;
    Real h = 1e-6;

    auto centerLine = [&](Real s) -> Real3 {
                          return {0.f, 0.f, s*height};
                      };

    auto err = testTwistForces(1.f, 0.1f, centerLine, 50, h);
    ASSERT_LE(err, 1e-5);
}

TEST (ROD, twistForces_helix)
{
    Real pitch  = 1.0;
    Real radius = 0.5;
    Real height = 1.0;
    Real h = 1e-4;

    auto centerLine = [&](Real s) -> Real3 {
                          Real z = s * height;
                          Real theta = 2 * M_PI * z / pitch;
                          Real x = radius * cos(theta);
                          Real y = radius * sin(theta);
                          return {x, y, z};
                      };

    auto err = testTwistForces(1.f, 0.1f, centerLine, 50, h);
    ASSERT_LE(err, 1e-3);
}


TEST (ROD, bendingForces_straight)
{
    Real height = 5.0;
    Real h = 1e-4;

    auto centerLine = [&](Real s) -> Real3 {
                          return {0.f, 0.f, s*height};
                      };

    int nSegs = 20;
    auto err = testBendingForces({1.0f, 0.0f, 0.5f}, {0.1f, 0.2f}, centerLine, nSegs, h);
    ASSERT_LE(err, 5e-4);
}

TEST (ROD, bendingForces_circle)
{
    Real radius = 4.0;
    Real h = 5e-5;

    auto centerLine = [&](Real s) -> Real3 {
                          Real theta = s * 2 * M_PI;
                          Real x = radius * cos(theta);
                          Real y = radius * sin(theta);
                          return {x, y, 0.f};
                      };


    real3 B {1.0f, 0.0f, 1.0f};
    real2 kappa {0.f, 0.f};

    std::vector<int> nsegs = {8, 16, 32};
    for (auto n : nsegs)
    {
        auto err = testBendingForces(B, kappa, centerLine, n, h);
        // printf("%d %g\n", n, err);
        ASSERT_LE(err, 1e-3);
    }
}

TEST (ROD, bendingForces_helix)
{
    Real pitch  = 1.0;
    Real radius = 0.5;
    Real height = 1.0;
    Real h = 1e-3;

    auto centerLine = [&](Real s) -> Real3 {
                          Real z = s * height;
                          Real theta = 2 * M_PI * z / pitch;
                          Real x = radius * cos(theta);
                          Real y = radius * sin(theta);
                          return {x, y, z};
                      };

    real3 B {1.0f, 0.0f, 1.0f};
    real2 kappa {0.f, 0.f};

    std::vector<int> nsegs = {4, 8, 16};
    for (auto n : nsegs)
    {
        auto err = testBendingForces(B, kappa, centerLine, n, h);
        ASSERT_LE(err, 1e-3);
    }
}

// // expect zero forces everywhere
// TEST (ROD, smoothingForces_circle)
// {
//     Real radius = 4.0;
//     Real h = 5e-5;

//     auto centerLine = [&](Real s) -> Real3
//     {
//         Real theta = s * 2 * M_PI;
//         Real x = radius * cos(theta);
//         Real y = radius * sin(theta);
//         return {x, y, 0.f};
//     };


//     real kSmoothing = 1.0f;

//     std::vector<int> nsegs = {8, 16, 32};
//     for (auto n : nsegs)
//     {
//         auto err = testSmoothingForces(kSmoothing, centerLine, n, h);
//         // printf("%d %g\n", n, err);
//         ASSERT_LE(err, 1e-3);
//     }
// }

// TEST (ROD, smoothingForces_complex)
// {
//     Real h = 5e-3;

//     auto centerLine = [&](Real s) -> Real3
//     {
//         Real xmagn = 1.0;
//         Real ymagn = 2.0;
//         Real zmagn = 0.5;
//         Real x = xmagn * cos(s);
//         Real y = ymagn * s;
//         Real z = zmagn * s*s;
//         return {x, y, z};
//     };

//     real kSmoothing = 1.0f;

//     // std::vector<int> nsegs = {8, 16, 32};
//     std::vector<int> nsegs = {256};
//     for (auto n : nsegs)
//     {
//         auto err = testSmoothingForces(kSmoothing, centerLine, n, h);
//         // printf("%d %g\n", n, err);
//         ASSERT_LE(err, 1e-3);
//     }
// }



int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    logger.init(MPI_COMM_WORLD, "rod_forces.log", 0);

    testing::InitGoogleTest(&argc, argv);
    auto ret = RUN_ALL_TESTS();

    MPI_Finalize();
    return ret;
}
