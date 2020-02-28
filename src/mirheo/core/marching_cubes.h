#pragma once

#include "domain.h"

#include <functional>
#include <vector>

namespace mirheo
{

namespace MarchingCubes
{
/// Represents a surface implicitly with a scalar field
/// The zero level set represents the surface
using ImplicitSurfaceFunction = std::function< real(real3) >;

/// simple tructure that represents a triangle in 3D
struct Triangle
{
    real3 a; ///< vertex 0
    real3 b; ///< vertex 1
    real3 c; ///< vertex 2
};

/** \brief Create an explicit surface (triangles) from implicit surface (scalar field)
    using marching cubes
    \param [in] domain Domain information
    \param [in] resolution the number of grid points in each direction
    \param [in] surface The scalar field that represents implicitly the surface (0 levelset)
    \param [out] triangles The explicit surface representation
 */
void computeTriangles(DomainInfo domain, real3 resolution,
                      const ImplicitSurfaceFunction& surface,
                      std::vector<Triangle>& triangles);

} // namespace MarchingCubes

} // namespace mirheo
