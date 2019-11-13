#pragma once

#include <mirheo/core/utils/helper_math.h>
#include <mirheo/core/utils/cpu_gpu_defines.h>

#include <mpi.h>
#include <vector_types.h>

namespace mirheo
{

struct DomainInfo
{
    real3 globalSize, globalStart, localSize;

    inline __HD__ real3 local2global(real3 x) const
    {
        return x + globalStart + 0.5_r * localSize;
    }
    inline __HD__ real3 global2local(real3 x) const
    {
        return x - globalStart - 0.5_r * localSize;
    }

    inline __HD__ real4 global2localPlane(real4 plane) const noexcept {
        // v * x_global + d == v * x_local + (v * delta_{local -> global} + d)
        real3 v = make_real3(plane);
        return make_real4(v, dot(v, local2global(real3{0, 0, 0})) + plane.w);
    }

    template <typename RealType3>
    inline __HD__ bool inSubDomain(RealType3 xg) const
    {
        return (globalStart.x <= xg.x) && (xg.x < (globalStart.x + localSize.x))
            && (globalStart.y <= xg.y) && (xg.y < (globalStart.y + localSize.y))
            && (globalStart.z <= xg.z) && (xg.z < (globalStart.z + localSize.z));
    }    
};

DomainInfo createDomainInfo(MPI_Comm cartComm, real3 globalSize);

} // namespace mirheo
