#pragma once

#include "domain.h"

#include <functional>
#include <vector>

namespace MarchingCubes
{
using ImplicitSurfaceFunction = std::function< float(float3) >;

struct Triangle
{
    float3 a, b, c;
};

void computeTriangles(DomainInfo domain, float3 resolution,
                      const ImplicitSurfaceFunction& surface,
                      std::vector<Triangle>& triangles);

}
