#include <core/logger.h>
#include <core/utils/helper_math.h>
#include <core/utils/quaternion.h>
#include <core/pvs/rod_vector.h>
#include <plugins/utils/xyz.h>

#include <vector>
#include <functional>
#include <gtest/gtest.h>

Logger logger;

using CenterLineFunc = std::function<float3(float)>;

static void initialFlagellum(int n, std::vector<float3>& positions, CenterLineFunc centerLine)
{
    positions.resize(5 * n + 1);
    float h = 1.f / n;

    for (int i = 0; i < n; ++i) {
        float3 r = centerLine(i*h);

        positions[i * 5 + 0] = r;
        positions[i * 5 + 1] = r;
        positions[i * 5 + 2] = r;
        positions[i * 5 + 3] = r;
        positions[i * 5 + 4] = r;
    }

    positions[5*n] = centerLine(1.f);
}

inline void print(float3 v)
{
    printf("%g %g %g\n", v.x, v.y, v.z);
}

static void getTransformation(float3 t0, float3 t1, float4& Q)
{
    Q = getQfrom(t0, t1);
    auto t0t1 = cross(t0, t1);
    if (length(t0t1) > 1e-6)
        t0t1 = normalize(t0t1);

    float err_t0_t1   = length(t1 - rotate(t0, Q));
    float err_t01_t01 = length(t0t1 - rotate(t0t1, Q));

    ASSERT_LE(err_t01_t01, 1e-6f);
    ASSERT_LE(err_t0_t1, 1e-6);
}

static void transportBishopFrame(const std::vector<float3>& positions, std::vector<float3>& frames)
{
    int n = (positions.size() - 1) / 5;
    
    for (int i = 1; i < n; ++i)
    {
        auto r0 = positions[5*(i-1)];
        auto r1 = positions[5*(i)];
        auto r2 = positions[5*(i+1)];
        
        auto t0 = normalize(r1-r0);
        auto t1 = normalize(r2-r1);

        float4 Q;
        getTransformation(t0, t1, Q);
        auto u0 = frames[2*(i-1) + 0];
        auto v0 = frames[2*(i-1) + 1];
        auto u1 = rotate(u0, Q);
        auto v1 = rotate(v0, Q);
        frames[2*i + 0] = u1;
        frames[2*i + 1] = v1;

        // printf("%g %g %g %g %g %g\n",
        //        dot(u0, t0), dot(v0, t0), dot(u0, v0),
        //        dot(u0, cross(t0,t1)),
        //        dot(v0, cross(t0,t1)),
        //        dot(u0, u0));
    }
}

static float bendingEnergy(const float2 B[2], float2 omega_eq, const std::vector<float3>& positions)
{
    int n = (positions.size() - 1) / 5;

    float Etot = 0;
    
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

        float denom = dot(e0, e1) + sqrtf(dot(e0,e0) * dot(e1,e1));
        auto kappab = (2.f / denom) * cross(e0, e1);
        
        float2 om0 {dot(kappab, m20), -dot(kappab, m10)};
        float2 om1 {dot(kappab, m21), -dot(kappab, m11)};

        om0 -= omega_eq;
        om1 -= omega_eq;

        float l = 0.5 * (length(e0) + length(e1));

        float2 Bw {dot(om0 + om1, B[0]),
                   dot(om0 + om1, B[1])};

        float E = dot(Bw, om0 + om1) / l;
        Etot += E;
    }

    return Etot;
}



static void bendingForces(const float2 B[2], float2 omega_eq, const std::vector<float3>& positions, float h, std::vector<float3>& forces)
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
            forces[i].comp = (E - E0) / h;                      \
        } while(0)
        
        COMPUTE_FORCE(x);
        COMPUTE_FORCE(y);
        COMPUTE_FORCE(z);

#undef COMPUTE_FORCE
    }
}

static void setCrosses(const std::vector<float3>& frames, std::vector<float3>& positions)
{
    int n = (positions.size() - 1) / 5;
    for (int i = 0; i < n; ++i)
    {
        auto u = frames[2*i+0];
        auto v = frames[2*i+1];
        auto r0 = positions[5*i+0];
        auto r1 = positions[5*i+5];
        auto dr = 0.5f * (r1 - r0);
        float a = length(dr);
        auto c = 0.5f * (r0 + r1);

        positions[i*5+1] = c - a * u;
        positions[i*5+2] = c + a * u;
        positions[i*5+3] = c - a * v;
        positions[i*5+4] = c + a * v;
    }
}

static void initialFrame(float3 t0, float3& u, float3& v)
{
    t0 = normalize(t0);
    u = anyOrthogonal(t0);
    u = normalize(u);
    v = normalize(cross(t0, u));
}

template <class CenterLine>
static void initializeRef(CenterLine centerLine, int nSegments, std::vector<float3>& positions, std::vector<float3>& frames)
{
    initialFlagellum(nSegments, positions, centerLine);

    frames.resize(2*nSegments);
    initialFrame(positions[5]-positions[0],
                 frames[0], frames[1]);

    transportBishopFrame(positions, frames);
    setCrosses(frames, positions);
}

static void copyToRv(const std::vector<float3>& positions, RodVector& rod)
{
    for (int i = 0; i < positions.size(); ++i)
    {
        Particle p;
        p.r = positions[i];
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
    
    std::vector<float3> refPositions, refFrames;
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
        float3 a = refFrames[2*i];
        float3 b = frames[i];
        float3 diff = a - b;
        double err = std::max(std::max(fabs(diff.x), fabs(diff.y)), fabs(diff.z));

        Linfty = std::max(Linfty, err);
    }
    return Linfty;
}

TEST (FLAGELLA, BishopFrames_straight)
{
    float height = 1.0;
    
    auto centerLine = [&](float s) -> float3 {
                          return {0.f, 0.f, s*height};
                      };

    auto err = testBishopFrame(centerLine);
    ASSERT_LE(err, 1e-5);
}

TEST (FLAGELLA, BishopFrames_circle)
{
    float radius = 0.5;

    auto centerLine = [&](float s) -> float3 {
                          float theta = s * 2 * M_PI;
                          float x = radius * cos(theta);
                          float y = radius * sin(theta);
                          return {x, y, 0.f};
                      };

    auto err = testBishopFrame(centerLine);
    ASSERT_LE(err, 3e-5);
}

TEST (FLAGELLA, BishopFrames_helix)
{
    float pitch  = 1.0;
    float radius = 0.5;
    float height = 1.0;
    
    auto centerLine = [&](float s) -> float3 {
                          float z = s * height;
                          float theta = 2 * M_PI * z / pitch;
                          float x = radius * cos(theta);
                          float y = radius * sin(theta);
                          return {x, y, z};
                      };

    auto err = testBishopFrame(centerLine);
    ASSERT_LE(err, 1e-5);
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
