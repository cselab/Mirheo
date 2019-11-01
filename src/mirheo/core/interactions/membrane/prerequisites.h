#pragma once

#include "kernels/parameters.h"

#include <core/pvs/membrane_vector.h>
#include <core/celllist.h>
#include <core/utils/kernel_launch.h>

template <class EnergyParams>
inline void setPrerequisitesPerEnergy(__UNUSED const EnergyParams& params,
                                      __UNUSED ParticleVector *pv1,
                                      __UNUSED ParticleVector *pv2,
                                      __UNUSED CellList *cl1,
                                      __UNUSED CellList *cl2)
{}

inline void setPrerequisitesPerEnergy(const JuelicherBendingParameters&, ParticleVector *pv1,
                                      __UNUSED ParticleVector *pv2,
                                      __UNUSED CellList *cl1,
                                      __UNUSED CellList *cl2)
{
    auto ov = dynamic_cast<MembraneVector*>(pv1);
    
    ov->requireDataPerObject<real>(ChannelNames::lenThetaTot, DataManager::PersistenceMode::None);

    ov->requireDataPerParticle<real>(ChannelNames::areas, DataManager::PersistenceMode::None);
    ov->requireDataPerParticle<real>(ChannelNames::meanCurvatures, DataManager::PersistenceMode::None);
}




namespace InteractionMembraneJuelicherKernels
{
__device__ inline mReal compute_lenTheta(mReal3 v0, mReal3 v1, mReal3 v2, mReal3 v3)
{
    const mReal len = length(v2 - v0);
    const mReal theta = supplementaryDihedralAngle(v0, v1, v2, v3);
    return len * theta;
}

__global__ void computeAreasAndCurvatures(OVviewWithJuelicherQuants view, MembraneMeshView mesh)
{
    const int rbcId = blockIdx.y;
    const int idv0  = blockIdx.x * blockDim.x + threadIdx.x;
    const int offset = rbcId * mesh.nvertices;

    mReal lenTheta = 0;
    
    if (idv0 < mesh.nvertices)
    {        
        const int startId = mesh.maxDegree * idv0;
        const int degree = mesh.degrees[idv0];
        
        const int idv1 = mesh.adjacent[startId];
        const int idv2 = mesh.adjacent[startId+1];
        
        const mReal3 v0 = fetchPosition(view, offset + idv0);
        mReal3 v1 = fetchPosition(view, offset + idv1);
        mReal3 v2 = fetchPosition(view, offset + idv2);
        
        mReal area = 0;
        
#pragma unroll 2
        for (int i = 0; i < degree; i++) {
            
            const int idv3 = mesh.adjacent[startId + (i+2) % degree];
            const mReal3 v3 = fetchPosition(view, offset + idv3);
            
            area     += 0.3333333_mr * triangleArea(v0, v1, v2);
            lenTheta += compute_lenTheta(v0, v1, v2, v3);
            
            v1 = v2;
            v2 = v3;
        }
        
        view.vertexAreas          [offset + idv0] = area;
        view.vertexMeanCurvatures [offset + idv0] = lenTheta / (4 * area);
    }
    
    lenTheta = warpReduce( lenTheta, [] (mReal a, mReal b) { return a+b; } );

    if (laneId() == 0)
        atomicAdd(&view.lenThetaTot[rbcId], (real) lenTheta);
}
} // namespace InteractionMembraneJuelicherKernels


template <class EnergyParams>
inline void precomputeQuantitiesPerEnergy(__UNUSED const EnergyParams&,
                                          __UNUSED ParticleVector *pv1,
                                          __UNUSED cudaStream_t stream)
{}

inline void precomputeQuantitiesPerEnergy(const JuelicherBendingParameters&, ParticleVector *pv1, cudaStream_t stream)
{
    auto ov = dynamic_cast<MembraneVector *>(pv1);

    debug("Computing vertex areas and curvatures for %d cells of '%s'",
          ov->local()->nObjects, ov->name.c_str());

    ov->local()->dataPerObject.getData<real>(ChannelNames::lenThetaTot)->clear(stream);

    OVviewWithJuelicherQuants view(ov, ov->local());

    MembraneMeshView mesh(static_cast<MembraneMesh*>(ov->mesh.get()));

    const int nthreads = 128;    

    const dim3 threads(nthreads, 1);
    const dim3 blocks(getNblocks(mesh.nvertices, nthreads), view.nObjects);
        
    SAFE_KERNEL_LAUNCH(
        InteractionMembraneJuelicherKernels::computeAreasAndCurvatures,
        blocks, threads, 0, stream,
        view, mesh );
}
