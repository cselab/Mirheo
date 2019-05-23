#include <core/interactions/rod.h>
#include <core/logger.h>
#include <core/pvs/rod_vector.h>
#include <core/utils/helper_math.h>
#include <core/utils/quaternion.h>
#include <plugins/utils/xyz.h>

#include <vector>
#include <functional>
#include <gtest/gtest.h>

Logger logger;

#define FMT "%+6e"
#define SEP "\t"
#define EXPAND(v) v.x, v.y, v.z

using real = double;
using real2 = double2;
using real3 = double3;
using real4 = double4;

static real2 make_real2(float2 v) { return {(real) v.x, (real) v.y}; }
static real3 make_real3(float3 v) { return {(real) v.x, (real) v.y, (real) v.z}; }

using CenterLineFunc = std::function<real3(real)>;

static void initialFlagellum(int n, std::vector<real3>& positions, CenterLineFunc centerLine)
{
    positions.resize(5 * n + 1);
    real h = 1.0 / n;

    for (int i = 0; i < n; ++i) {
        real3 r = centerLine(i*h);

        positions[i * 5 + 0] = r;
        positions[i * 5 + 1] = r;
        positions[i * 5 + 2] = r;
        positions[i * 5 + 3] = r;
        positions[i * 5 + 4] = r;
    }

    positions[5*n] = centerLine(1.f);
}

static void getTransformation(real3 t0, real3 t1, real4& Q)
{
    Q = getQfrom(t0, t1);
    auto t0t1 = cross(t0, t1);
    if (length(t0t1) > 1e-6)
        t0t1 = normalize(t0t1);

    real err_t0_t1   = length(t1 - rotate(t0, Q));
    real err_t01_t01 = length(t0t1 - rotate(t0t1, Q));

    ASSERT_LE(err_t01_t01, 1e-6f);
    ASSERT_LE(err_t0_t1, 1e-6);
}

static void initialFrame(real3 t0, real3& u, real3& v)
{
    t0 = normalize(t0);
    u = anyOrthogonal(t0);
    u = normalize(u);
    v = normalize(cross(t0, u));
}

static void transportBishopFrame(const std::vector<real3>& positions, std::vector<real3>& frames)
{
    int n = (positions.size() - 1) / 5;
    
    for (int i = 1; i < n; ++i)
    {
        auto r0 = positions[5*(i-1)];
        auto r1 = positions[5*(i)];
        auto r2 = positions[5*(i+1)];
        
        auto t0 = normalize(r1-r0);
        auto t1 = normalize(r2-r1);

        real4 Q;
        getTransformation(t0, t1, Q);
        auto u0 = frames[2*(i-1) + 0];
        auto u1 = rotate(u0, Q);
        auto v1 = cross(t1, u1);
        frames[2*i + 0] = u1;
        frames[2*i + 1] = v1;
    }
}

static real bendingEnergy(const float2 B[2], float2 omega_eq, const std::vector<real3>& positions)
{
    int n = (positions.size() - 1) / 5;

    real Etot = 0;
    
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
        
        auto m10 = normalize(dp0Perp);
        auto m20 = cross(t0, m10);

        auto m11 = normalize(dp1Perp);
        auto m21 = cross(t1, m11);
        
        real denom = length(e0) * length(e1) + dot(e0, e1);
        auto bicur = (2.f / denom) * cross(e0, e1);
        
        real dp0Perpinv = 1.0 / length(dp0Perp);
        real dp1Perpinv = 1.0 / length(dp1Perp);

        real l = 0.5 * (length(e0) + length(e1));
        real linv = 1.0 / l;
        
        real2 om0 = {+ linv * dp0Perpinv * dot(bicur, cross(t0, dp0)),
                     - linv * dp0Perpinv * dot(bicur, dp0)};
        real2 om1 = {+ linv * dp1Perpinv * dot(bicur, cross(t1, dp1)),
                     - linv * dp1Perpinv * dot(bicur, dp1)};
        
        
        om0 -= make_real2(omega_eq);
        om1 -= make_real2(omega_eq);

        real2 Bom0 {dot(om0, make_real2(B[0])),
                    dot(om0, make_real2(B[1]))};
        
        real2 Bom1 {dot(om1, make_real2(B[0])),
                    dot(om1, make_real2(B[1]))};

        real E = 0.25 * l * (dot(Bom0, om0) + dot(Bom1, om1));
        Etot += E;
    }

    return Etot;
}

inline real safeDiffTheta(real t0, real t1)
{
    auto dth = t1 - t0;
    if (dth >  M_PI) dth -= 2.0 * M_PI;
    if (dth < -M_PI) dth += 2.0 * M_PI;
    return dth;
}

static real twistEnergy(real kTwist, real tau0, const std::vector<real3>& positions, const std::vector<real3>& frames)
{
    int n = (positions.size() - 1) / 5;

    real Etot = 0;
    
    for (int i = 1; i < n; ++i)
    {
        auto r0 = positions[5*(i-1)];
        auto r1 = positions[5*(i)];
        auto r2 = positions[5*(i+1)];

        auto u0 = frames[2*(i-1)   ];
        auto v0 = frames[2*(i-1) + 1];

        auto u1 = frames[2*i    ];
        auto v1 = frames[2*i + 1];
        
        auto dp0 = positions[5*(i-1) + 2] - positions[5*(i-1) + 1];
        auto dp1 = positions[5*i     + 2] - positions[5*i     + 1];
        
        auto e0 = r1-r0;
        auto e1 = r2-r1;
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

static void bendingForces(real h, const float2 B[2], float2 omega_eq, const std::vector<real3>& positions, std::vector<real3>& forces)
{
    auto perturbed = positions;
    auto E0 = bendingEnergy(B, omega_eq, positions);

    auto computeEnergy = [&]() {
        return bendingEnergy(B, omega_eq, perturbed);
    };
    
    for (size_t i = 0; i < positions.size(); ++i)
    {
        auto computeForce = [&](real3 dir) {
            const auto r = positions[i];
            perturbed[i] = r + (h/2) * dir;
            auto Ep = computeEnergy();
            perturbed[i] = r - (h/2) * dir;
            auto Em = computeEnergy();
            perturbed[i] = r;
            return - (Ep - Em) / h;
        };

        forces[i].x = computeForce({1.0, 0.0, 0.0});
        forces[i].y = computeForce({0.0, 1.0, 0.0});
        forces[i].z = computeForce({0.0, 0.0, 1.0});
    }
}

static void twistForces(real h, float kt, float tau0, const std::vector<real3>& positions, std::vector<real3>& forces)
{
    auto perturbed = positions;
    int nSegments = (positions.size() - 1) / 5;
    
    std::vector<real3> frames(2*nSegments);

    auto compEnergy = [&]() {
                          initialFrame(perturbed[5]-perturbed[0], frames[0], frames[1]);
                          transportBishopFrame(perturbed, frames);
                          return twistEnergy(kt, tau0, perturbed, frames);
                      };
    
    for (size_t i = 0; i < positions.size(); ++i)
    {
        auto computeForce = [&](real3 dir) {
            const auto r = positions[i];
            perturbed[i] = r + (h/2) * dir;
            auto Ep = compEnergy();
            perturbed[i] = r - (h/2) * dir;
            auto Em = compEnergy();
            perturbed[i] = r;
            return - (Ep - Em) / h;
        };

        forces[i].x = computeForce({1.0, 0.0, 0.0});
        forces[i].y = computeForce({0.0, 1.0, 0.0});
        forces[i].z = computeForce({0.0, 0.0, 1.0});
    }
}

static void setCrosses(const std::vector<real3>& frames, std::vector<real3>& positions)
{
    int n = (positions.size() - 1) / 5;
    for (int i = 0; i < n; ++i)
    {
        auto u = frames[2*i+0];
        auto v = frames[2*i+1];
        auto r0 = positions[5*i+0];
        auto r1 = positions[5*i+5];
        auto dr = 0.5f * (r1 - r0);
        real a = length(dr);
        auto c = 0.5f * (r0 + r1);

        positions[i*5+1] = c - a * u;
        positions[i*5+2] = c + a * u;
        positions[i*5+3] = c - a * v;
        positions[i*5+4] = c + a * v;
    }
}

template <class CenterLine>
static void initializeRef(CenterLine centerLine, int nSegments, std::vector<real3>& positions, std::vector<real3>& frames)
{
    initialFlagellum(nSegments, positions, centerLine);

    frames.resize(2*nSegments);
    initialFrame(positions[5]-positions[0],
                 frames[0], frames[1]);

    transportBishopFrame(positions, frames);
    setCrosses(frames, positions);
}

static void copyToRv(const std::vector<real3>& positions, RodVector& rod)
{
    auto& pos = rod.local()->positions ();
    auto& vel = rod.local()->velocities();

    for (int i = 0; i < positions.size(); ++i)
    {
        Particle p;
        p.r = make_float3(positions[i]);
        p.u = make_float3(0);
        p.setId(i);
        pos[i] = p.r2Float4();
        vel[i] = p.u2Float4();
    }
    pos.uploadToDevice(defaultStream);
    vel.uploadToDevice(defaultStream);    
}

template <class CenterLine>
static double testBishopFrame(CenterLine centerLine)
{
    YmrState state(DomainInfo(), 0.f);
    int nSegments {200};
    
    std::vector<real3> refPositions, refFrames;
    RodVector rod(&state, "rod", 1.f, nSegments, 1);

    initializeRef(centerLine, nSegments, refPositions, refFrames);
    copyToRv(refPositions, rod);
    
    rod.updateBishopFrame(defaultStream);

    HostBuffer<float3> frames;
    frames.copy(rod.local()->bishopFrames, defaultStream);
    CUDA_Check( cudaDeviceSynchronize() );

    double Linfty = 0;
    for (int i = 0; i < refFrames.size() / 2; ++i)
    {
        real3 a = refFrames[2*i];
        real3 b = make_real3(frames[i]);
        auto diff = a - b;
        double err = std::max(std::max(fabs(diff.x), fabs(diff.y)), fabs(diff.z));

        Linfty = std::max(Linfty, err);
    }
    return Linfty;
}

TEST (ROD, BishopFrames_straight)
{
    real height = 1.0;
    
    auto centerLine = [&](real s) -> real3 {
                          return {(real)0.0, (real)0.0, s*height};
                      };

    auto err = testBishopFrame(centerLine);
    ASSERT_LE(err, 1e-5);
}

TEST (ROD, BishopFrames_circle)
{
    real radius = 0.5;

    auto centerLine = [&](real s) -> real3 {
                          real theta = s * 2 * M_PI;
                          real x = radius * cos(theta);
                          real y = radius * sin(theta);
                          return {x, y, 0.f};
                      };

    auto err = testBishopFrame(centerLine);
    ASSERT_LE(err, 3e-5);
}

TEST (ROD, BishopFrames_helix)
{
    real pitch  = 1.0;
    real radius = 0.5;
    real height = 1.0;
    
    auto centerLine = [&](real s) -> real3 {
                          real z = s * height;
                          real theta = 2 * M_PI * z / pitch;
                          real x = radius * cos(theta);
                          real y = radius * sin(theta);
                          return {x, y, z};
                      };

    auto err = testBishopFrame(centerLine);
    ASSERT_LE(err, 3e-5);
}

static void checkMomentum(const PinnedBuffer<float4>& pos, const HostBuffer<Force>& forces)
{
    double3 totForce  {0., 0., 0.};
    double3 totTorque {0., 0., 0.};

    for (int i = 0; i < forces.size(); ++i)
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
static double testTwistForces(float kt, float tau0, CenterLine centerLine, int nSegments, real h)
{
    YmrState state(DomainInfo(), 0.f);

    RodParameters params;
    params.kBending = {0.f, 0.f, 0.f};
    params.kappaEq = {{0.f, 0.f}};
    params.kTwist = kt;
    params.tauEq = {tau0};
    params.groundE = {0.f};
    params.a0 = params.l0 = 0.f;
    params.kBounds = 0.f;
    params.kVisc = 0.f;
    
    std::vector<real3> refPositions, refFrames, refForces;
    RodVector rod(&state, "rod", 1.f, nSegments, 1);
    InteractionRod interactions(&state, "rod_interaction", params);
    initializeRef(centerLine, nSegments, refPositions, refFrames);
    copyToRv(refPositions, rod);


    refForces.resize(refPositions.size());
    twistForces(h, kt, tau0, refPositions, refForces);

    rod.local()->forces().clear(defaultStream);
    interactions.setPrerequisites(&rod, &rod, nullptr, nullptr);
    interactions.local(&rod, &rod, nullptr, nullptr, defaultStream);

    HostBuffer<Force> forces;
    forces.copy(rod.local()->forces(), defaultStream);
    CUDA_Check( cudaDeviceSynchronize() );

    double Linfty = 0;
    for (int i = 0; i < refForces.size(); ++i)
    {
        real3 a = refForces[i];
        real3 b = make_real3(forces[i].f);
        real3 diff = a - b;
        double err = std::max(std::max(fabs(diff.x), fabs(diff.y)), fabs(diff.z));

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
static double testBendingForces(float3 B, float2 kappa, CenterLine centerLine, int nSegments, real h)
{
    YmrState state(DomainInfo(), 0.f);

    RodParameters params;
    params.kBending = B;
    params.kappaEq  = {kappa};
    params.kTwist   = 0.f;
    params.tauEq    = {0.f};
    params.groundE  = {0.f};
    params.a0       = 0.f;
    params.l0       = 0.f;
    params.kBounds  = 0.f;
    params.kVisc    = 0.f;
    
    std::vector<real3> refPositions, refFrames, refForces;
    RodVector rod(&state, "rod", 1.f, nSegments, 1);
    InteractionRod interactions(&state, "rod_interaction", params);
    initializeRef(centerLine, nSegments, refPositions, refFrames);
    copyToRv(refPositions, rod);


    refForces.resize(refPositions.size());
    const float2 B_[2] {{B.x, B.y}, {B.y, B.z}};
    bendingForces(h, B_, kappa, refPositions, refForces);

    rod.local()->forces().clear(defaultStream);
    interactions.setPrerequisites(&rod, &rod, nullptr, nullptr);
    interactions.local(&rod, &rod, nullptr, nullptr, defaultStream);

    HostBuffer<Force> forces;
    forces.copy(rod.local()->forces(), defaultStream);
    CUDA_Check( cudaDeviceSynchronize() );

    double Linfty = 0;
    for (int i = 0; i < refForces.size(); ++i)
    {
        real3 a = refForces[i];
        real3 b = make_real3(forces[i].f);
        real3 diff = a - b;
        double err = std::max(std::max(fabs(diff.x), fabs(diff.y)), fabs(diff.z));
        
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


TEST (ROD, twistForces_straight)
{
    real height = 5.0;
    real h = 1e-6;
    
    auto centerLine = [&](real s) -> real3 {
                          return {0.f, 0.f, s*height};
                      };

    auto err = testTwistForces(1.f, 0.1f, centerLine, 50, h);
    ASSERT_LE(err, 1e-5);
}

TEST (ROD, twistForces_helix)
{
    real pitch  = 1.0;
    real radius = 0.5;
    real height = 1.0;
    real h = 1e-7;
    
    auto centerLine = [&](real s) -> real3 {
                          real z = s * height;
                          real theta = 2 * M_PI * z / pitch;
                          real x = radius * cos(theta);
                          real y = radius * sin(theta);
                          return {x, y, z};
                      };

    auto err = testTwistForces(1.f, 0.1f, centerLine, 50, h);
    ASSERT_LE(err, 1e-3);
}


TEST (ROD, bendingForces_straight)
{
    real height = 5.0;
    real h = 1e-4;
    
    auto centerLine = [&](real s) -> real3 {
                          return {0.f, 0.f, s*height};
                      };

    int nSegs = 20;
    auto err = testBendingForces({1.0f, 0.0f, 0.5f}, {0.1f, 0.2f}, centerLine, nSegs, h);
    ASSERT_LE(err, 5e-4);
}

TEST (ROD, bendingForces_circle)
{
    real radius = 4.0;
    real h = 5e-5;
    
    auto centerLine = [&](real s) -> real3 {
                          real theta = s * 2 * M_PI;
                          real x = radius * cos(theta);
                          real y = radius * sin(theta);
                          return {x, y, 0.f};
                      };


    float3 B {1.0f, 0.0f, 1.0f};
    float2 kappa {0.f, 0.f};

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
    real pitch  = 1.0;
    real radius = 0.5;
    real height = 1.0;
    real h = 1e-3;
    
    auto centerLine = [&](real s) -> real3 {
                          real z = s * height;
                          real theta = 2 * M_PI * z / pitch;
                          real x = radius * cos(theta);
                          real y = radius * sin(theta);
                          return {x, y, z};
                      };

    float3 B {1.0f, 0.0f, 1.0f};
    float2 kappa {0.f, 0.f};

    std::vector<int> nsegs = {4, 8, 16};
    for (auto n : nsegs)
    {
        auto err = testBendingForces(B, kappa, centerLine, n, h);
        ASSERT_LE(err, 1e-3);
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
