#pragma once

#include "domain.h"

#include <functional>
#include <vector>

namespace MarchingCubes
{
using ImplicitSurfaceFunction = std::function< real(real3) >;

struct Triangle
{
    real3 a, b, c;
};

void computeTriangles(DomainInfo domain, real3 resolution,
                      const ImplicitSurfaceFunction& surface,
                      std::vector<Triangle>& triangles);

} // namespace MarchingCubes
