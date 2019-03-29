#include "rod_vector.h"
#include "views/rv.h"

#include <core/utils/quaternion.h>
#include <core/utils/kernel_launch.h>

#include <extern/cub/cub/device/device_scan.cuh>

LocalRodVector::LocalRodVector(ParticleVector *pv, int objSize, int nObjects) :
    LocalObjectVector(pv, objSize, nObjects)
{}

LocalRodVector::~LocalRodVector() = default;


__HD__ inline constexpr int getNParts(int nSegments)
{
    return 5 * nSegments + 1;
}

__HD__ inline constexpr int getNSegments(int np)
{
    return (np - 1) / 5;
}


namespace RodVectorKernels
{

__device__ float3 fetchPosition(RVview view, int i)
{
    Particle p;
    p.readCoordinate(view.particles, i);
    return p.r;
}

__global__ void computeBishopQuaternion(RVview view)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int nSegments = getNSegments(view.objSize);
    
    int objId     = i / nSegments;
    int segmentId = i % nSegments;
    int objStart  = objId * view.objSize;

    if (objId > view.nObjects) return;
    if (segmentId > nSegments) return;

    float4 Q {1.f, 0.f, 0.f, 0.f};

    if (segmentId > 0)
    {
        auto r0 = fetchPosition(view, objStart + 5 * (segmentId - 1));
        auto r1 = fetchPosition(view, objStart + 5 * (segmentId    ));
        auto r2 = fetchPosition(view, objStart + 5 * (segmentId + 1));
        
        auto t0 = normalize(r1-r0);
        auto t1 = normalize(r2-r1);
        
        Q = getQfrom(t0, t1);
    }

    int dstId = objId * nSegments + segmentId;
    view.bishopQuaternions[dstId] = Q;
}

} // namespace RodVectorKernels


RodVector::RodVector(const YmrState *state, std::string name, float mass, int nSegments, int nObjects) :
    ObjectVector( state, name, mass, getNParts(nSegments),
                  std::make_unique<LocalRodVector>(this, getNParts(nSegments), nObjects),
                  std::make_unique<LocalRodVector>(this, getNParts(nSegments), 0) )
{}

RodVector::~RodVector() = default;


struct ComposeRotations
{
    __device__ __forceinline__
    float4 operator()(const float4 &a, const float4 &b) const
    {
        return multiplyQ(a, b);
    }
};

void RodVector::updateBishopFrame(cudaStream_t stream)
{
    RVview view(this, local(), stream);

    const int nthreads = 128;

    int nSegments = getNSegments(view.size);
    int nSegmentsTot = nSegments * view.nObjects;
    
    SAFE_KERNEL_LAUNCH(
        RodVectorKernels::computeBishopQuaternion,
        getNblocks(nSegmentsTot, nthreads), nthreads, 0, stream,
        view );

    size_t workSize = scanWork.size();

    if (workSize == 0)
    {        
        CUDA_Check( cub::DeviceScan::InclusiveScan
                    (nullptr, workSize,
                     view.bishopQuaternions,
                     view.bishopQuaternions,
                     ComposeRotations(), nSegments, stream)) ;
        
        scanWork.resize_anew(workSize);
    }

    for (int i = 0; i < view.nObjects; ++i)
    {
        CUDA_Check( cub::DeviceScan::InclusiveScan
                    (scanWork.devPtr(), workSize,
                     view.bishopQuaternions + nSegments * i,
                     view.bishopQuaternions + nSegments * i,
                     ComposeRotations(), nSegments, stream)) ;
    }
}
