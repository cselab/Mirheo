#include "prerequisites.h"
#include "force_kernels/real.h"
#include "force_kernels/common.h"

#include <mirheo/core/pvs/membrane_vector.h>
#include <mirheo/core/logger.h>
#include <mirheo/core/utils/kernel_launch.h>

namespace mirheo
{

void setPrerequisitesPerEnergy(const JuelicherBendingParameters&, MembraneVector *mv)
{
    mv->requireDataPerObject<real>(channel_names::lenThetaTot, DataManager::PersistenceMode::None);

    mv->requireDataPerParticle<real>(channel_names::areas, DataManager::PersistenceMode::None);
    mv->requireDataPerParticle<real>(channel_names::meanCurvatures, DataManager::PersistenceMode::None);
}


namespace interaction_membrane_juelicher_kernels
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
} // namespace interaction_membrane_juelicher_kernels


void precomputeQuantitiesPerEnergy(const JuelicherBendingParameters&, MembraneVector *mv, cudaStream_t stream)
{
    debug("Computing vertex areas and curvatures for %d cells of '%s'",
          mv->local()->getNumObjects(), mv->getCName());

    mv->local()->dataPerObject.getData<real>(channel_names::lenThetaTot)->clear(stream);

    OVviewWithJuelicherQuants view(mv, mv->local());

    MembraneMeshView mesh(static_cast<MembraneMesh*>(mv->mesh.get()));

    const int nthreads = 128;

    const dim3 threads(nthreads, 1);
    const dim3 blocks(getNblocks(mesh.nvertices, nthreads), view.nObjects);

    SAFE_KERNEL_LAUNCH(
        interaction_membrane_juelicher_kernels::computeAreasAndCurvatures,
        blocks, threads, 0, stream,
        view, mesh );
}

} // namespace mirheo
