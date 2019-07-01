#include "wall_with_velocity.h"

#include "common_kernels.h"
#include "stationary_walls/box.h"
#include "stationary_walls/cylinder.h"
#include "stationary_walls/plane.h"
#include "stationary_walls/sdf.h"
#include "stationary_walls/sphere.h"
#include "velocity_field/oscillate.h"
#include "velocity_field/rotate.h"
#include "velocity_field/translate.h"

#include <core/bounce_solver.h>
#include <core/celllist.h>
#include <core/logger.h>
#include <core/pvs/object_vector.h>
#include <core/pvs/particle_vector.h>
#include <core/pvs/views/pv.h>
#include <core/utils/cuda_common.h>
#include <core/utils/kernel_launch.h>

#include <cassert>
#include <cmath>
#include <fstream>
#include <texture_types.h>


template<typename VelocityField>
__global__ void imposeVelField(PVview view, const VelocityField velField)
{
    const int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= view.size) return;

    Particle p(view.readParticle(pid));

    p.u = velField(p.r);

    view.writeParticle(pid, p);
}

//===============================================================================================
// Member functions
//===============================================================================================

template<class InsideWallChecker, class VelocityField>
WallWithVelocity<InsideWallChecker, VelocityField>::WallWithVelocity
(std::string name, const MirState *state, InsideWallChecker&& insideWallChecker, VelocityField&& velField) :
    SimpleStationaryWall<InsideWallChecker>(name, state, std::move(insideWallChecker)),
    velField(std::move(velField))
{}


template<class InsideWallChecker, class VelocityField>
void WallWithVelocity<InsideWallChecker, VelocityField>::setup(MPI_Comm& comm)
{
    info("Setting up wall %s", this->name.c_str());

    CUDA_Check( cudaDeviceSynchronize() );

    this->insideWallChecker.setup(comm, this->state->domain);
    velField.setup(this->state->currentTime, this->state->domain);

    CUDA_Check( cudaDeviceSynchronize() );
}

template<class InsideWallChecker, class VelocityField>
void WallWithVelocity<InsideWallChecker, VelocityField>::attachFrozen(ParticleVector* pv)
{
    SimpleStationaryWall<InsideWallChecker>::attachFrozen(pv);

    const int nthreads = 128;
    PVview view(pv, pv->local());
    SAFE_KERNEL_LAUNCH(
            imposeVelField,
            getNblocks(view.size, nthreads), nthreads, 0, 0,
            view, velField.handler() );

    CUDA_Check( cudaDeviceSynchronize() );
}

template<class InsideWallChecker, class VelocityField>
void WallWithVelocity<InsideWallChecker, VelocityField>::bounce(cudaStream_t stream)
{
    float t  = this->state->currentTime;
    float dt = this->state->dt;
    
    velField.setup(t, this->state->domain);
    this->bounceForce.clear(stream);

    for (int i=0; i < this->particleVectors.size(); i++)
    {
        auto  pv = this->particleVectors[i];
        auto  cl = this->cellLists[i];
        auto& bc = this->boundaryCells[i];
        auto view = cl->CellList::getView<PVviewWithOldParticles>();

        debug2("Bouncing %d %s particles with wall velocity, %d boundary cells",
               pv->local()->size(), pv->name.c_str(), bc.size());

        const int nthreads = 64;
        SAFE_KERNEL_LAUNCH(
                BounceKernels::sdfBounce,
                getNblocks(bc.size(), nthreads), nthreads, 0, stream,
                view, cl->cellInfo(), bc.devPtr(), bc.size(), dt,
                this->insideWallChecker.handler(),
                velField.handler(),
                this->bounceForce.devPtr());

        CUDA_Check( cudaPeekAtLastError() );
    }
}


template class WallWithVelocity<StationaryWall_Sphere,   VelocityField_Rotate>;
template class WallWithVelocity<StationaryWall_Cylinder, VelocityField_Rotate>;
template class WallWithVelocity<StationaryWall_Plane,    VelocityField_Translate>;
template class WallWithVelocity<StationaryWall_Plane,    VelocityField_Oscillate>;




