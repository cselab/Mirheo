#include "freeze_particles.h"

#include <core/logger.h>
#include <core/pvs/particle_vector.h>
#include <core/walls/sdf_wall.h>
#include <core/utils/cuda_common.h>
#include <core/utils/kernel_launch.h>
#include <core/walls/sdf_kernels.h>


__global__ void countFrozen(PVview view, SDFWall::SdfInfo sdfInfo, float minSdf, float maxSdf, int* nFrozen)
{
	const int pid = blockIdx.x * blockDim.x + threadIdx.x;
	if (pid >= view.size) return;

	const float4 coo = view.particles[2*pid];
	const float sdf = evalSdf(coo, sdfInfo);

	if (sdf > minSdf && sdf < maxSdf)
		atomicAggInc(nFrozen);
}

__global__ void collectFrozen(PVview view, SDFWall::SdfInfo sdfInfo, float minSdf, float maxSdf,
		float4* frozen, int* nFrozen)
{
	const int pid = blockIdx.x * blockDim.x + threadIdx.x;
	if (pid >= view.size) return;

	const float4 coo = view.particles[2*pid];
	const float4 vel = view.particles[2*pid+1];

	const float sdf = evalSdf(coo, sdfInfo);

	if (sdf > minSdf && sdf < maxSdf)
	{
		const int ind = atomicAggInc(nFrozen);
		frozen[2*ind] = coo;
		frozen[2*ind + 1] = make_float4(0.0f, 0.0f, 0.0f, vel.w);
	}
}

void freezeParticlesInWall(SDFWall* wall, ParticleVector* pv, ParticleVector* frozen, float minSdf, float maxSdf)
{
	CUDA_Check( cudaDeviceSynchronize() );

	PinnedBuffer<int> nFrozen(1);

	auto view = create_PVview(pv, pv->local());
	const int nthreads = 128;
	const int nblocks = getNblocks(view.size, nthreads);

	nFrozen.clear(0);
	SAFE_KERNEL_LAUNCH( countFrozen,
			nblocks, nthreads, 0, 0,
			view, wall->sdfInfo, minSdf, maxSdf, nFrozen.devPtr());

	nFrozen.downloadFromDevice(0);

	frozen->local()->resize(nFrozen[0], 0);
	frozen->mass = pv->mass;
	frozen->globalDomainStart = pv->globalDomainStart;
	frozen->localDomainSize = pv->localDomainSize;

	debug("Freezing %d particles", nFrozen[0]);

	nFrozen.clear(0);
	SAFE_KERNEL_LAUNCH(collectFrozen,
			nblocks, nthreads, 0, 0,
			view, wall->sdfInfo, minSdf, maxSdf,
			(float4*)frozen->local()->coosvels.devPtr(), nFrozen.devPtr());
	nFrozen.downloadFromDevice(0);

	CUDA_Check( cudaDeviceSynchronize() );
}
