#include "temperaturize.h"

#include <core/pvs/particle_vector.h>
#include <core/simulation.h>

#include <core/cuda_common.h>
#include <core/cuda-rng.h>


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

__global__ void applyTemperature(float4* coosvels, int np, float invm, float kbT, float seed1, float seed2)
{
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	if (gid >= np) return;

	float2 rand1 = normal_BoxMuller(seed1);
	float2 rand2 = normal_BoxMuller(seed2);

	float3 vel = sqrtf(kbT * invm) * make_float3(rand1.x, rand1.y, rand2.x);

	coosvels[2*gid+1] = make_float4(vel, 0.0f);
}


void TemperaturizePlugin::setup(Simulation* sim, const MPI_Comm& comm, const MPI_Comm& interComm)
{
	SimulationPlugin::setup(sim, comm, interComm);

	for (auto& name : pvNames)
	{
		auto pv = sim->getPVbyName(name);
		if (pv == nullptr)
			die("Cannot apply temperature to particle vector %s, not found", name.c_str());

		pvs.push_back(pv);
	}
}

void TemperaturizePlugin::beforeForces(cudaStream_t stream)
{
	for (auto pv : pvs)
	{
		if (pv->local()->size() > 0)
			applyTemperature<<<getNblocks(pv->local()->size(), 128), 128, 0, stream>>>(
					(float4*)pv->local()->coosvels.devPtr(), pv->local()->size(), 1.0/pv->mass, kbT, drand48(), drand48());
	}
}

