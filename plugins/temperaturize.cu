#include "temperaturize.h"

#include <core/utils/kernel_launch.h>
#include <core/pvs/particle_vector.h>
#include <core/simulation.h>

#include <core/utils/cuda_common.h>
#include <core/utils/cuda_rng.h>


__device__ __forceinline__ float2 normal_BoxMuller(float seed)
{
	float u1 = Saru::uniform01(seed, threadIdx.x, blockIdx.x);
	float u2 = Saru::uniform01(u1,   blockIdx.x, threadIdx.x);

	float r = sqrtf(-2.0f * logf(u1));
	float theta = 2.0f * M_PI * u2;

	float2 res;
	sincosf(theta, &res.x, &res.y);
	res *= r;

	return res;
}

__global__ void applyTemperature(PVview view, float kbT, float seed1, float seed2, bool keepVelocity)
{
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	if (gid >= view.size) return;

	float2 rand1 = normal_BoxMuller(seed1);
	float2 rand2 = normal_BoxMuller(seed2);

	float3 vel = sqrtf(kbT * view.invMass) * make_float3(rand1.x, rand1.y, rand2.x);

	Float3_int u(view.particles[2*gid+1]);
	if (keepVelocity) u.v += vel;
	else              u.v  = vel;

	view.particles[2*gid+1] = u.toFloat4();
}


void TemperaturizePlugin::setup(Simulation* sim, const MPI_Comm& comm, const MPI_Comm& interComm)
{
	SimulationPlugin::setup(sim, comm, interComm);

	pv =sim->getPVbyNameOrDie(pvName);
}

void TemperaturizePlugin::beforeForces(cudaStream_t stream)
{
	PVview view(pv, pv->local());
	const int nthreads = 128;

	SAFE_KERNEL_LAUNCH(
			applyTemperature,
			getNblocks(view.size, nthreads), nthreads, 0, stream,
			view, kbT, drand48(), drand48(), keepVelocity );
}

