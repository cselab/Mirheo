#include <mirheo/core/logger.h>
#include <mirheo/core/marching_cubes.h>

#include <cstdio>
#include <cmath>
#include <gtest/gtest.h>

using namespace mirheo;

inline void dump_off(const std::vector<marching_cubes::Triangle>& triangles)
{
    FILE *f = fopen("mesh.off", "w");

    fprintf(f, "%d %d\n",
            static_cast<int>(triangles.size() * 3),
            static_cast<int>(triangles.size()));
    
    for (const auto& t : triangles) {
        fprintf(f, "%g %g %g\n", t.a.x, t.a.y, t.a.z);
        fprintf(f, "%g %g %g\n", t.b.x, t.b.y, t.b.z);
        fprintf(f, "%g %g %g\n", t.c.x, t.c.y, t.c.z);
    }

    for (size_t i = 0; i < triangles.size(); ++i)
        fprintf(f, "3 %d %d %d\n",
                static_cast<int>(i * 3 + 0),
                static_cast<int>(i * 3 + 1),
                static_cast<int>(i * 3 + 2));
    
    fclose(f);
}

TEST (MARCHING_CUBES, Sphere)
{
    real R = 1.0;
    real L = 2.5 * R;
    
    DomainInfo domain;    
    domain.globalStart = make_real3(0, 0, 0);
    domain.localSize   = make_real3(L, L, L);
    domain.globalSize  = domain.localSize;

    real3 center = domain.globalStart + 0.5 * domain.globalSize;
    // real3 center = domain.globalStart;

    auto sphereSurface = [&] (real3 r)
    {
        r -= center;
        return math::sqrt(dot(r, r)) - R;
    };

    std::vector<marching_cubes::Triangle> triangles;

    real h = 0.1;
    real3 resolution {h, h, h};
    
    marching_cubes::computeTriangles(domain, resolution, sphereSurface, triangles);

    real maxVal  = 0.0;
    real meanVal = 0.0;

    // dump_off(triangles);
    
    for (auto& t : triangles) {
        real va, vb, vc;
        va = sphereSurface( domain.local2global(t.a) );
        vb = sphereSurface( domain.local2global(t.b) );
        vc = sphereSurface( domain.local2global(t.c) );

        maxVal = std::max(std::abs(va), maxVal);
        maxVal = std::max(std::abs(vb), maxVal);
        maxVal = std::max(std::abs(vc), maxVal);
        meanVal += va + vb + vc;
    }
    meanVal /= 3*triangles.size();

    ASSERT_LE(meanVal, 0.01);
    ASSERT_LE(maxVal, 0.01);
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
