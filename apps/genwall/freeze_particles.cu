#include "freeze_particles.h"

#include <core/logger.h>
#include <core/pvs/particle_vector.h>
#include <core/utils/cuda_common.h>
#include <core/utils/kernel_launch.h>

#include <core/walls/simple_stationary_wall.h>

#include <core/walls/stationary_walls/cylinder.h>
#include <core/walls/stationary_walls/sphere.h>
#include <core/walls/stationary_walls/plane.h>
#include <core/walls/stationary_walls/sdf.h>
#include <core/walls/stationary_walls/box.h>


template<bool QUERY, typename InsideWallChecker>
__global__ void collectFrozen(PVview view, float* sdfs, float minVal, float maxVal, float4* frozen, int* nFrozen)
{
	const int pid = blockIdx.x * blockDim.x + threadIdx.x;
	if (pid >= view.size) return;

	Particle p(view.particles, pid);
	p.u = make_float3(0);

	const float val = sdfs[pid];

	if (val > minVal && val < maxVal)
	{
		const int ind = atomicAggInc(nFrozen);

		if (!QUERY)
			p.write2Float4(frozen, ind);
	}
}

void freezeParticlesInWall(SDF_basedWall* wall, ParticleVector* pv, ParticleVector* frozen, float minVal, float maxVal)
{
	CUDA_Check( cudaDeviceSynchronize() );

	DeviceBuffer<float> sdfs;

	wall->sdfPerParticle(pv, &sdfs, nullptr, 0);

	PinnedBuffer<int> nFrozen(1);

	PVview view(pv, pv->local());
	const int nthreads = 128;
	const int nblocks = getNblocks(view.size, nthreads);

	nFrozen.clear(0);
	SAFE_KERNEL_LAUNCH(collectFrozen<true>,
				nblocks, nthreads, 0, 0,
				view, sdfs.devPtr(), minVal, maxVal,
				(float4*)frozen->local()->coosvels.devPtr(), nFrozen.devPtr());

	nFrozen.downloadFromDevice(0);

	frozen->local()->resize(nFrozen[0], 0);
	frozen->mass = pv->mass;
	frozen->domain = pv->domain;

	debug("Freezing %d particles", nFrozen[0]);

	nFrozen.clear(0);
	SAFE_KERNEL_LAUNCH(collectFrozen<false>,
			nblocks, nthreads, 0, 0,
			view, minVal, maxVal,
			(float4*)frozen->local()->coosvels.devPtr(), nFrozen.devPtr(), checker.handler());
	nFrozen.downloadFromDevice(0);

	CUDA_Check( cudaDeviceSynchronize() );
}


