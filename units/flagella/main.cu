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

template <typename T3>
inline void print(T3 v)
{
    printf("%g %g %g\n", v.x, v.y, v.z);
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

        auto t0 = normalize(r1-r0);
        auto t1 = normalize(r2-r1);
        
        auto dp0 = normalize(positions[5*(i-1) + 2] - positions[5*(i-1) + 1]);
        auto dp1 = normalize(positions[5*i + 2] - positions[5*i + 1]);

        auto m10 = normalize(dp0 - dot(dp0, t0) * t0);
        auto m20 = cross(t0, m10);

        auto m11 = normalize(dp1 - dot(dp1, t1) * t1);
        auto m21 = cross(t1, m11);
        
        auto e0 = r1-r0;
        auto e1 = r2-r1;

        real denom = dot(e0, e1) + sqrtf(dot(e0,e0) * dot(e1,e1));
        auto kappab = (2.f / denom) * cross(e0, e1);
        
        real2 om0 {dot(kappab, m20), -dot(kappab, m10)};
        real2 om1 {dot(kappab, m21), -dot(kappab, m11)};

        om0 -= make_real2(omega_eq);
        om1 -= make_real2(omega_eq);

        real l = 0.5 * (length(e0) + length(e1));

        real2 Bw {dot(om0 + om1, make_real2(B[0])),
                  dot(om0 + om1, make_real2(B[1]))};

        real E = dot(Bw, om0 + om1) / l;
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
        
        auto E = kTwist * l * dtau * dtau;

        Etot += E;
    }

    return Etot;
}



static void bendingForces(const float2 B[2], float2 omega_eq, const std::vector<real3>& positions, real h, std::vector<real3>& forces)
{
    auto perturbed = positions;
    auto E0 = bendingEnergy(B, omega_eq, positions);
    
    for (size_t i = 0; i < positions.size(); ++i)
    {
#define COMPUTE_FORCE(comp) do {                                \
            auto r = positions[i];                              \
            r.comp += h;                                        \
            perturbed[i] = r;                                   \
            auto E = bendingEnergy(B, omega_eq, perturbed);     \
            forces[i].comp = (E0 - E) / h;                      \
            perturbed[i] = positions[i];                        \
        } while(0)
        
        COMPUTE_FORCE(x);
        COMPUTE_FORCE(y);
        COMPUTE_FORCE(z);

#undef COMPUTE_FORCE
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
            auto r = positions[i];
            perturbed[i] = r + (h/2) * dir;
            auto Ep = compEnergy();
            perturbed[i] = r - (h/2) * dir;
            auto Em = compEnergy();
            perturbed[i] = positions[i];
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
    for (int i = 0; i < positions.size(); ++i)
    {
        Particle p;
        p.r = make_float3(positions[i]);
        p.u = make_float3(0);
        rod.local()->coosvels[i] = p;
    }
    rod.local()->coosvels.uploadToDevice(defaultStream);    
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

TEST (FLAGELLA, BishopFrames_straight)
{
    real height = 1.0;
    
    auto centerLine = [&](real s) -> real3 {
                          return {(real)0.0, (real)0.0, s*height};
                      };

    auto err = testBishopFrame(centerLine);
    ASSERT_LE(err, 1e-5);
}

TEST (FLAGELLA, BishopFrames_circle)
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

TEST (FLAGELLA, BishopFrames_helix)
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
    ASSERT_LE(err, 2e-5);
}


template <class CenterLine>
static double testTwistForces(float kt, float tau0, CenterLine centerLine, real h)
{
    YmrState state(DomainInfo(), 0.f);
    int nSegments {50};

    RodParameters params;
    params.kBending = {0.f, 0.f, 0.f};
    params.omegaEq = {0.f, 0.f};
    params.kTwist = kt;
    params.tauEq = tau0;
    params.a0 = params.l0 = 0.f;
    params.kBounds = 0.f;
    
    std::vector<real3> refPositions, refFrames, refForces;
    RodVector rod(&state, "rod", 1.f, nSegments, 1);
    InteractionRod interactions(&state, "rod_interaction", params);
    initializeRef(centerLine, nSegments, refPositions, refFrames);
    copyToRv(refPositions, rod);


    refForces.resize(refPositions.size());
    twistForces(h, kt, tau0, refPositions, refForces);

    rod.local()->forces.clear(defaultStream);
    interactions.setPrerequisites(&rod, &rod, nullptr, nullptr);
    interactions.local(&rod, &rod, nullptr, nullptr, defaultStream);

    HostBuffer<Force> forces;
    forces.copy(rod.local()->forces, defaultStream);
    CUDA_Check( cudaDeviceSynchronize() );

    double Linfty = 0;
    for (int i = 0; i < refForces.size(); ++i)
    {
        real3 a = refForces[i];
        real3 b = make_real3(forces[i].f);
        real3 diff = a - b;
        double err = std::max(std::max(fabs(diff.x), fabs(diff.y)), fabs(diff.z));

        // if ((i % 5) == 0) printf("%03d ---------- \n", i/5);
        // printf("%g\t%g\t%g\t%g\t%g\t%g\n",
        //        a.x, a.y, a.z, b.x, b.y, b.z);
        
        Linfty = std::max(Linfty, err);
    }
    return Linfty;
}

TEST (FLAGELLA, twistForces_straight)
{
    real height = 5.0;
    real h = 1e-6;
    
    auto centerLine = [&](real s) -> real3 {
                          return {0.f, 0.f, s*height};
                      };

    auto err = testTwistForces(1.f, 0.1f, centerLine, h);
    ASSERT_LE(err, 1e-5);
}

TEST (FLAGELLA, twistForces_helix)
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

    auto err = testTwistForces(1.f, 0.1f, centerLine, h);
    ASSERT_LE(err, 1e-3);
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
