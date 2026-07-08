// Copyright 2020 ETH Zurich. All Rights Reserved.

#include "obj_binding.h"

#include <mirheo/core/pvs/particle_vector.h>
#include <mirheo/core/pvs/views/pv.h>
#include <mirheo/core/utils/common.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/utils/kernel_launch.h>
#include <mirheo/core/utils/macros.h>

namespace mirheo
{

static constexpr int NoPartner = -1;

namespace obj_binding_kernels
{

__global__ void fill(int n, int *array, int value)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n)
        return;

    array[i] = value;
}

__global__ void createGidMap(int maxGid, int *map, PVview view)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= view.size)
        return;

    const auto gid = view.readParticle(i).getId();

    if (gid < maxGid)
        map[gid] = i;
}

__global__ void createPartnersMap(int npairs, const int2 *pairs, const int *gidToPv1Ids, const int *gidToPv2Ids, int *partnerIds)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= npairs)
        return;

    const auto pair = pairs[i];

    const int from = gidToPv1Ids[pair.x];
    const int to = gidToPv2Ids[pair.y];

    if (from != NoPartner)
        partnerIds[from] = to;
}

__global__ void applyBindingForces(real kBound, const int *partnerIds, PVview view1, PVview view2, DomainInfo domain)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= view1.size)
        return;

    const int j = partnerIds[i];

    if (j == NoPartner)
        return;

    const auto ri = Real3_int(view1.readPosition(i)).v;
    const auto rj = Real3_int(view2.readPosition(j)).v;

    const auto dr = rj - ri;

    if (math::abs(dr.x) >= 0.5_r * domain.localSize.x ||
        math::abs(dr.y) >= 0.5_r * domain.localSize.y ||
        math::abs(dr.z) >= 0.5_r * domain.localSize.z)
        return;

    const real3 f = kBound * dr;

    atomicAdd(view1.forces + i,  f);
    atomicAdd(view2.forces + j, -f);
}

} // namespace obj_binding_kernels


ObjectBindingInteraction::ObjectBindingInteraction(const MirState *state, std::string name,
                                                   real kBound, std::vector<int2> pairs) :
    Interaction(state, name),
    kBound_(kBound),
    pairs_(pairs.size())
{
    int maxId1 {-1}, maxId2 {-1};
    for (const auto& pair : pairs)
    {
        maxId1 = std::max(maxId1, pair.x);
        maxId2 = std::max(maxId2, pair.y);
    }

    gidToPv1Ids_.resize_anew(maxId1+1);
    gidToPv2Ids_.resize_anew(maxId2+1);

    CUDA_Check( cudaMemcpy(pairs_.devPtr(), pairs.data(), sizeof(pairs[0]) * pairs.size(), cudaMemcpyHostToDevice) );
}

ObjectBindingInteraction::~ObjectBindingInteraction() = default;

void ObjectBindingInteraction::local(ParticleVector *pv1, ParticleVector *pv2,
                                     __UNUSED CellList *cl1, __UNUSED CellList *cl2, cudaStream_t stream)
{
    _buildInteractionMap(pv1, pv2, pv1->local(), pv2->local(), stream);
    _computeForces(pv1, pv2, pv1->local(), pv2->local(), stream);
}

void ObjectBindingInteraction::halo(ParticleVector *pv1, ParticleVector *pv2,
                                    __UNUSED CellList *cl1, __UNUSED CellList *cl2, cudaStream_t stream)
{
    _buildInteractionMap(pv1, pv2, pv1->local(), pv2->halo(), stream);
    _computeForces(pv1, pv2, pv1->local(), pv2->halo(), stream);

    _buildInteractionMap(pv1, pv2, pv1->halo(), pv2->local(), stream);
    _computeForces(pv1, pv2, pv1->halo(), pv2->local(), stream);
}

void ObjectBindingInteraction::_buildInteractionMap(ParticleVector *pv1, ParticleVector *pv2,
                                                    LocalParticleVector *lpv1, LocalParticleVector *lpv2,
                                                    cudaStream_t stream)
{
    auto createGidMap = [stream](DeviceBuffer<int>& gidToPvIds, const PVview view)
    {
        constexpr int nthreads = 128;
        const int nblocks = getNblocks(gidToPvIds.size(), nthreads);

        SAFE_KERNEL_LAUNCH(
            obj_binding_kernels::fill,
            nblocks, nthreads, 0, stream,
            gidToPvIds.size(), gidToPvIds.devPtr(), NoPartner);

        SAFE_KERNEL_LAUNCH(
            obj_binding_kernels::createGidMap,
            nblocks, nthreads, 0, stream,
            gidToPvIds.size(), gidToPvIds.devPtr(), view);
    };

    PVview view1(pv1, lpv1);
    PVview view2(pv2, lpv2);

    createGidMap(gidToPv1Ids_, view1);
    createGidMap(gidToPv2Ids_, view2);

    constexpr int nthreads = 128;

    partnersMaps_.resize_anew(view1.size);

    SAFE_KERNEL_LAUNCH(
        obj_binding_kernels::fill,
        getNblocks(partnersMaps_.size(), nthreads), nthreads, 0, stream,
        partnersMaps_.size(), partnersMaps_.devPtr(), NoPartner);

    SAFE_KERNEL_LAUNCH(
        obj_binding_kernels::createPartnersMap,
        getNblocks(pairs_.size(), nthreads), nthreads, 0, stream,
        pairs_.size(), pairs_.devPtr(),
        gidToPv1Ids_.devPtr(), gidToPv2Ids_.devPtr(),
        partnersMaps_.devPtr());
}

void ObjectBindingInteraction::_computeForces(ParticleVector *pv1, ParticleVector *pv2,
                                              LocalParticleVector *lpv1, LocalParticleVector *lpv2,
                                              cudaStream_t stream) const
{
    PVview view1(pv1, lpv1);
    PVview view2(pv2, lpv2);

    constexpr int nthreads = 128;
    const int nblocks = getNblocks(view1.size, nthreads);

    SAFE_KERNEL_LAUNCH(
         obj_binding_kernels::applyBindingForces,
         nblocks, nthreads, 0, stream,
         kBound_, partnersMaps_.devPtr(), view1, view2, getState()->domain);
}


} // namespace mirheo
