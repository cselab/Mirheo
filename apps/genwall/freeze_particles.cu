#include "freeze_particles.h"

#include <core/logger.h>
#include <core/pvs/particle_vector.h>
#include <core/utils/cuda_common.h>
#include <core/utils/kernel_launch.h>

#include <core/walls/simple_stationary_wall.h>

#include <core/walls/stationary_walls/cylinder.h>
#include <core/walls/stationary_walls/sphere.h>
#include <core/walls/stationary_walls/sdf.h>


template<bool QUERY, typename InsideWallChecker>
__global__ void collectFrozen(PVview view, float minVal, float maxVal, float4* frozen, int* nFrozen, InsideWallChecker checker)
{
	const int pid = blockIdx.x * blockDim.x + threadIdx.x;
	if (pid >= view.size) return;

	Particle p(view.particles, pid);
	p.u = make_float3(0);

	const float val = checker(view, p.r);

	if (val > minVal && val < maxVal)
	{
		const int ind = atomicAggInc(nFrozen);

		if (!QUERY)
			p.write2Float4(frozen, ind);
	}
}

template<typename InsideWallChecker>
void freezeParticlesInWall(const InsideWallChecker& checker, ParticleVector* pv, ParticleVector* frozen, float minVal, float maxVal)
{
	CUDA_Check( cudaDeviceSynchronize() );

	PinnedBuffer<int> nFrozen(1);

	PVview view(pv, pv->local());
	const int nthreads = 128;
	const int nblocks = getNblocks(view.size, nthreads);

	nFrozen.clear(0);
	SAFE_KERNEL_LAUNCH(collectFrozen<true>,
				nblocks, nthreads, 0, 0,
				view, minVal, maxVal,
				(float4*)frozen->local()->coosvels.devPtr(), nFrozen.devPtr(), checker.handler());

	nFrozen.downloadFromDevice(0);

	frozen->local()->resize(nFrozen[0], 0);
	frozen->mass = pv->mass;
	frozen->globalDomainStart = pv->globalDomainStart;
	frozen->localDomainSize = pv->localDomainSize;

	debug("Freezing %d particles", nFrozen[0]);

	nFrozen.clear(0);
	SAFE_KERNEL_LAUNCH(collectFrozen<false>,
			nblocks, nthreads, 0, 0,
			view, minVal, maxVal,
			(float4*)frozen->local()->coosvels.devPtr(), nFrozen.devPtr(), checker.handler());
	nFrozen.downloadFromDevice(0);

	CUDA_Check( cudaDeviceSynchronize() );
}

void freezeParticlesWrapper(Wall* wall, ParticleVector* pv, ParticleVector* frozen, float minVal, float maxVal)
{
	{
		SimpleStationaryWall<StationaryWall_Cylinder>* w;
		if ( (w = dynamic_cast< SimpleStationaryWall<StationaryWall_Cylinder>* >(wall)) != nullptr)
			freezeParticlesInWall<StationaryWall_Cylinder> (w->getChecker(), pv, frozen, minVal, maxVal);
	}

	{
		SimpleStationaryWall<StationaryWall_Sphere>* w;
		if ( (w = dynamic_cast< SimpleStationaryWall<StationaryWall_Sphere>* >(wall)) != nullptr)
			freezeParticlesInWall<StationaryWall_Sphere> (w->getChecker(), pv, frozen, minVal, maxVal);
	}

	{
		SimpleStationaryWall<StationaryWall_SDF>* w;
		if ( (w = dynamic_cast< SimpleStationaryWall<StationaryWall_SDF>* >(wall)) != nullptr)
			freezeParticlesInWall<StationaryWall_SDF> (w->getChecker(), pv, frozen, minVal, maxVal);
	}
}

