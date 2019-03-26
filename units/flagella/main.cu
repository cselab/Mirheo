#include <core/logger.h>
#include <core/utils/helper_math.h>
#include <core/utils/quaternion.h>
#include <plugins/utils/xyz.h>

#include <vector>
#include <functional>
#include <gtest/gtest.h>

Logger logger;

static void clear(std::vector<float3>& vect)
{
    for (auto& v : vect) v = make_float3(0, 0, 0);
}

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
    ASSERT_LE(length(t0t1 - rotate(t0t1, Q)), 1e-6f);
    ASSERT_LE(length(t1 - rotate(t0, Q)), 1e-6);
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

        positions[i*5+1] = c + a * u;
        positions[i*5+2] = c - a * u;
        positions[i*5+3] = c + a * v;
        positions[i*5+4] = c - a * v;
    }
}

static void initialFrame(float3 t0, float3& u, float3& v)
{
    t0 = normalize(t0);
    // while (true)
    // {
    //     u = {float(drand48()-0.5), float(drand48()-0.5), float(drand48()-0.5)};        
    //     u -= t0 * dot(u, t0);
    //     if (length(u) > 1e-1)
    //         break;
    // }
    u = anyOrthogonal(t0);
    u = normalize(u);
    v = normalize(cross(t0, u));
}

static void dump(MPI_Comm comm, int id, int np, const float3 *positions)
{
    std::vector<Particle> particles(np);
    for (int i = 0; i < np; ++i) {
        Particle p;
        float3 r = positions[i];
        p.r.x = r.x;
        p.r.y = r.y;
        p.r.z = r.z;
        particles[i] = p;
    }
    std::string tstr = std::to_string(id);
    std::string name = "flagella_" + std::string(5 - tstr.length(), '0') + tstr + ".xyz";

    writeXYZ(comm, name, particles.data(), particles.size());
}

static void run(MPI_Comm comm)
{
    // number of edges
    int n = 1000;

    float pitch = 1.0;
    float radius = 0.5;
    float height = 1.0;
    
    std::vector<float3> positions, frames;

    auto centerLine = [&](float s) -> float3 {
                          float z = s * height;
                          float theta = 2 * M_PI * z / pitch;
                          float x = radius * cos(theta);
                          float y = radius * sin(theta);
                          return {x, y, z};
                      };

    // auto centerLine = [&](float s) -> float3 {
    //                       float theta = s * 2 * M_PI;
    //                       float x = radius * cos(theta);
    //                       float y = radius * sin(theta);
    //                       return {x, y, 0.f};
    //                   };

    // auto centerLine = [&](float s) -> float3 {
    //                       return {0.f, 0.f, s*height};
    //                   };

    initialFlagellum(n, positions, centerLine);
    frames.resize(2*n);
    initialFrame(positions[5]-positions[0],
                 frames[0], frames[1]);

    transportBishopFrame(positions, frames);
    setCrosses(frames, positions);
    
    dump(comm, 0, positions.size(), positions.data());
}


int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    run(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}
