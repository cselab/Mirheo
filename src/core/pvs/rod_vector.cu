#include "rod_vector.h"
#include "views/rv.h"

#include <core/utils/quaternion.h>
#include <core/utils/kernel_launch.h>

#include <extern/cub/cub/device/device_scan.cuh>
#include <extern/cub/cub/block/block_scan.cuh>

__HD__ inline constexpr int getNumParts(int nSegments)
{
    return 5 * nSegments + 1;
}

__HD__ inline constexpr int getNumSegments(int np)
{
    return (np - 1) / 5;
}

LocalRodVector::LocalRodVector(ParticleVector *pv, int objSize, int nObjects) :
    LocalObjectVector(pv, objSize, nObjects)
{
    resize_anew(objSize * nObjects);
}

LocalRodVector::~LocalRodVector() = default;

void LocalRodVector::resize(int np, cudaStream_t stream)
{
    LocalObjectVector::resize(np, stream);

    int numTotSegments = getNumSegmentsPerRod() * nObjects;

    bishopQuaternions.resize(numTotSegments, stream);
    bishopFrames     .resize(numTotSegments, stream);
}

void LocalRodVector::resize_anew(int np)
{
    LocalObjectVector::resize_anew(np);
    
    int numTotSegments = getNumSegmentsPerRod() * nObjects;
    bishopQuaternions.resize_anew(numTotSegments);
    bishopFrames     .resize_anew(numTotSegments);
}

int LocalRodVector::getNumSegmentsPerRod() const
{
    return getNumSegments(objSize);
}

namespace RodVectorKernels
{

__device__ inline float3 fetchPosition(const RVview& view, int i)
{
    Particle p;
    p.readCoordinate(view.particles, i);
    return p.r;
}

__global__ void computeBishopQuaternion(RVview view)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int nSegments = getNumSegments(view.objSize);
    
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

__device__ inline float3 getInitialFrame(const RVview& view, int objId)
{
    int start = view.objSize * objId;
    auto r0 = fetchPosition(view, start + 0);
    auto pu = fetchPosition(view, start + 1);
    auto mu = fetchPosition(view, start + 2);
    auto r1 = fetchPosition(view, start + 5);

    auto t0 = normalize(r1 - r0);
    auto u = pu - mu;
    u -= t0 * dot(t0, u);
    return normalize(u);
}

struct ComposeRotations
{
    __device__ __forceinline__
    float4 operator()(const float4 &a, const float4 &b) const
    {
        return multiplyQ(b, a);
    }
};

template <int BlockSize>
__global__ void computeBishopFrames(RVview view)
{
    using BlockScan = cub::BlockScan<float4, BlockSize>;
    
    __shared__ typename BlockScan::TempStorage blockScanStorage;
    
    assert(blockDim.x == BlockSize);
    int tid = threadIdx.x;
    int nSegments = getNumSegments(view.objSize);
    
    int objId     = blockIdx.x;
    int objStart  = objId * nSegments;

    int nsteps = (view.objSize + BlockSize - 1) / BlockSize;

    float3 initialFrame = getInitialFrame(view, objId);

    float4 Qstart {1.f, 0.f, 0.f, 0.f};
    ComposeRotations composeRotations;
    
    for (int step = 0; step < nsteps; ++step)
    {
        int segmentId = step * BlockSize + tid;

        float4 Q {1.f, 0.f, 0.f, 0.f}, Qaggregate;

        if (segmentId < nSegments)
            Q = view.bishopQuaternions[objStart + segmentId];
            
        BlockScan(blockScanStorage).InclusiveScan(Q, Q, composeRotations, Qaggregate);

        Q      = composeRotations(Qstart, Q);
        Qstart = composeRotations(Qstart, Qaggregate);

        if (segmentId < nSegments)
            view.bishopFrames[objStart + segmentId] = normalize(rotate(initialFrame, Q));
    }    
}    

} // namespace RodVectorKernels


RodVector::RodVector(const YmrState *state, std::string name, float mass, int nSegments, int nObjects) :
    ObjectVector( state, name, mass, getNumParts(nSegments),
                  std::make_unique<LocalRodVector>(this, getNumParts(nSegments), nObjects),
                  std::make_unique<LocalRodVector>(this, getNumParts(nSegments), 0) )
{}

RodVector::~RodVector() = default;

void RodVector::updateBishopFrame(cudaStream_t stream)
{
    RVview view(this, local(), stream);

    {
        const int nthreads = 128;
        
        int nSegments = getNumSegments(view.size);
        int nSegmentsTot = nSegments * view.nObjects;
        
        SAFE_KERNEL_LAUNCH(
             RodVectorKernels::computeBishopQuaternion,
             getNblocks(nSegmentsTot, nthreads), nthreads, 0, stream,
             view );
    }

    {
        constexpr int nthreads = 256;
        SAFE_KERNEL_LAUNCH(
            RodVectorKernels::computeBishopFrames<nthreads>,
            view.nObjects, nthreads, 0, stream,
            view );
    }
}
