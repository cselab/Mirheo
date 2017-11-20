#include "wall_with_velocity.h"

#include <fstream>
#include <cmath>
#include <texture_types.h>
#include <cassert>

#include <core/logger.h>
#include <core/utils/kernel_launch.h>
#include <core/utils/cuda_common.h>
#include <core/celllist.h>
#include <core/pvs/particle_vector.h>
#include <core/pvs/object_vector.h>
#include <core/bounce_solver.h>

#include <core/utils/cuda_rng.h>

#include "stationary_walls/cylinder.h"
#include "stationary_walls/sdf.h"
#include "stationary_walls/sphere.h"
#include "stationary_walls/plane.h"
#include "stationary_walls/box.h"

#include "velocity_field/rotate.h"
#include "velocity_field/translate.h"
#include "velocity_field/oscillate.h"

//===============================================================================================
// SDF bouncing kernel
//===============================================================================================

template<typename InsideWallChecker>
__device__ __forceinline__ float3 rescue(float3 candidate, float dt, float tol, int id, const InsideWallChecker& checker)
{
	const int maxIters = 20;
	const float factor = 5.0f*dt;

	for (int i=0; i<maxIters; i++)
	{
		float v = checker(candidate);
		if (v < -tol) break;

		float3 rndShift;
		rndShift.x = Saru::mean0var1(candidate.x - floorf(candidate.x), id+i, id*id);
		rndShift.y = Saru::mean0var1(rndShift.x,                        id+i, id*id);
		rndShift.z = Saru::mean0var1(rndShift.y,                        id+i, id*id);

		if (checker(candidate + factor*rndShift) < v)
			candidate += factor*rndShift;
	}

	return candidate;
}

template<typename InsideWallChecker, typename VelocityField>
__global__ void bounceWithVelocity(
		PVview_withOldParticles view, CellListInfo cinfo,
		const int* wallCells, const int nWallCells, const float dt,
		const InsideWallChecker checker,
		const VelocityField velField)
{
	const float tol = 2e-6f;

	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= nWallCells) return;
	const int cid = wallCells[tid];
	const int pstart = cinfo.cellStarts[cid];
	const int pend   = cinfo.cellStarts[cid+1];

	for (int pid = pstart; pid < pend; pid++)
	{
		Particle p(view.particles, pid);
		if (checker(p.r) <= -tol) continue;

		Particle pOld(view.old_particles, pid);
		float3 dr = p.r - pOld.r;

		const float alpha = solveLinSearch([=] (float lambda) {
			return checker(pOld.r + dr*lambda) + tol;
		});
		float3 candidate = (alpha >= 0.0f) ? pOld.r + alpha * dr : pOld.r;
		candidate = rescue(candidate, dt, tol, p.i1, checker);

		p.r = candidate;
		float3 uWall = velField(p.r);
		p.u = uWall - (p.u - uWall);

		p.write2Float4(cinfo.particles, pid);
	}
}

template<typename VelocityField>
__global__ void imposeVelField(PVview view, const VelocityField velField)
{
	const int pid = blockIdx.x * blockDim.x + threadIdx.x;
	if (pid >= view.size) return;

	Particle p(view.particles, pid);

	p.u = velField(p.r);

	p.write2Float4(view.particles, pid);
}

//===============================================================================================
// Member functions
//===============================================================================================

template<class InsideWallChecker, class VelocityField>
void WallWithVelocity<InsideWallChecker, VelocityField>::setup(MPI_Comm& comm, DomainInfo domain, ParticleVector* jointPV)
{
	info("Setting up wall %s", this->name.c_str());
	this->domain = domain;

	CUDA_Check( cudaDeviceSynchronize() );
	MPI_Check( MPI_Comm_dup(comm, &this->wallComm) );

	this->insideWallChecker.setup(this->wallComm, domain);
	velField.setup(this->wallComm, domain);

	if (jointPV == nullptr)
		error("Moving wall requires that corresponding frozen particles are named the same as the wall '%s' itself", this->name.c_str());
	else
	{
		const int nthreads = 128;
		PVview view(jointPV, jointPV->local());
		SAFE_KERNEL_LAUNCH(
				imposeVelField,
				getNblocks(view.size, nthreads), nthreads, 0, 0,
				view, velField.handler() );
	}

	CUDA_Check( cudaDeviceSynchronize() );
}

template<class InsideWallChecker, class VelocityField>
void WallWithVelocity<InsideWallChecker, VelocityField>::bounce(float dt, cudaStream_t stream)
{
	velField.setup(this->wallComm, domain);

	for (int i=0; i < this->particleVectors.size(); i++)
	{
		auto pv = this->particleVectors[i];
		auto cl = this->cellLists[i];
		auto bc = this->boundaryCells[i];
		PVview_withOldParticles view(pv, pv->local());

		debug2("Bouncing %d %s particles with wall velocity, %d boundary cells",
				pv->local()->size(), pv->name.c_str(), bc->size());

		const int nthreads = 64;
		SAFE_KERNEL_LAUNCH(
				bounceWithVelocity,
				getNblocks(bc->size(), nthreads), nthreads, 0, stream,
				view, cl->cellInfo(), bc->devPtr(), bc->size(), dt,
				this->insideWallChecker.handler(),
				velField.handler() );

		CUDA_Check( cudaPeekAtLastError() );
		this->nBounceCalls[i]++;
	}
}


template class WallWithVelocity<StationaryWall_Sphere,   VelocityField_Rotate>;
template class WallWithVelocity<StationaryWall_Cylinder, VelocityField_Rotate>;
template class WallWithVelocity<StationaryWall_Plane,    VelocityField_Translate>;
template class WallWithVelocity<StationaryWall_Plane,    VelocityField_Oscillate>;




