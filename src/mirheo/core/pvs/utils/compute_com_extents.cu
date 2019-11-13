#include "compute_com_extents.h"

#include <mirheo/core/pvs/object_vector.h>
#include <mirheo/core/pvs/views/ov.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/utils/kernel_launch.h>

#include <cuda_runtime.h>

namespace mirheo
{
namespace ComputeComExtentsKernels
{

__global__ void minMaxCom(OVview ovView)
{
    const int gid    = threadIdx.x + blockDim.x * blockIdx.x;
    const int objId  = gid / warpSize;
    const int laneId = gid % warpSize;
    if (objId >= ovView.nObjects) return;

    real3 mymin = make_real3(+1e10_r);
    real3 mymax = make_real3(-1e10_r);
    real3 mycom = make_real3(0.0_r);

#pragma unroll 3
    for (int i = laneId; i < ovView.objSize; i += warpSize)
    {
        const int offset = objId * ovView.objSize + i;

        const real3 coo = make_real3(ovView.readPosition(offset));

        mymin = math::min(mymin, coo);
        mymax = math::max(mymax, coo);
        mycom += coo;
    }

    mycom = warpReduce( mycom, [] (real a, real b) { return a+b; } );
    mymin = warpReduce( mymin, [] (real a, real b) { return math::min(a, b); } );
    mymax = warpReduce( mymax, [] (real a, real b) { return math::max(a, b); } );

    if (laneId == 0)
        ovView.comAndExtents[objId] = {mycom / ovView.objSize, mymin, mymax};
}

} // namespace ComputeComExtentsKernels

void computeComExtents(ObjectVector *ov, LocalObjectVector *lov, cudaStream_t stream)
{
    OVview view(ov, lov);
    
    constexpr int warpSize = 32;
    const int nthreads = 128;
    const int nblocks = getNblocks(view.nObjects * warpSize, nthreads);
    
    SAFE_KERNEL_LAUNCH(
        ComputeComExtentsKernels::minMaxCom,
        nblocks, nthreads, 0, stream,
        view );
}

} // namespace mirheo
