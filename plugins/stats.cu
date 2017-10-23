#include "stats.h"
#include <plugins/simple_serializer.h>
#include <core/datatypes.h>
#include <core/pvs/particle_vector.h>
#include <core/simulation.h>
#include <core/utils/cuda_common.h>
#include <core/utils/kernel_launch.h>


__global__ void totalMomentumEnergy(PVview view, ReductionType* momentum, ReductionType* energy)
{
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	const int wid = tid % warpSize;
	if (tid >= view.size) return;

	const float3 vel = make_float3(view.particles[2*tid+1]);

	float3 myMomentum = vel * view.mass;
	float myEnergy = dot(vel, vel) * view.mass*0.5f;

	myMomentum = warpReduce(myMomentum, [](float a, float b) { return a+b; });
	myEnergy   = warpReduce(myEnergy,   [](float a, float b) { return a+b; });

	if (wid == 0)
	{
		atomicAdd(momentum+0, (ReductionType)myMomentum.x);
		atomicAdd(momentum+1, (ReductionType)myMomentum.y);
		atomicAdd(momentum+2, (ReductionType)myMomentum.z);
		atomicAdd(energy,     (ReductionType)myEnergy);
	}
}

SimulationStats::SimulationStats(std::string name, int fetchEvery) :
		SimulationPlugin(name), fetchEvery(fetchEvery)
{
	timer.start();
}

void SimulationStats::afterIntegration(cudaStream_t stream)
{
	if (currentTimeStep % fetchEvery != 0) return;

	auto& pvs = sim->getParticleVectors();

	momentum.clear(stream);
	energy  .clear(stream);

	nparticles = 0;
	for (auto& pv : pvs)
	{
		PVview view(pv, pv->local());

		SAFE_KERNEL_LAUNCH(
				totalMomentumEnergy,
				getNblocks(view.size, 128), 128, 0, stream,
				view, momentum.devPtr(), energy.devPtr() );

		nparticles += view.size;
	}

	momentum.downloadFromDevice(stream, false);
	energy  .downloadFromDevice(stream);

	needToDump = true;
}

void SimulationStats::serializeAndSend(cudaStream_t stream)
{
	if (needToDump)
	{
		float tm = timer.elapsedAndReset() / (currentTimeStep < fetchEvery ? 1.0f : fetchEvery);
		SimpleSerializer::serialize(sendBuffer, tm, currentTime, currentTimeStep, nparticles, momentum, energy);
		send(sendBuffer.data(), sendBuffer.size());
		needToDump = false;
	}
}

PostprocessStats::PostprocessStats(std::string name) :
		PostprocessPlugin(name)
{
	if (std::is_same<ReductionType, float>::value)
		mpiReductionType = MPI_FLOAT;
	else if (std::is_same<ReductionType, double>::value)
		mpiReductionType = MPI_DOUBLE;
	else
		die("Incompatible type");
}

void PostprocessStats::deserialize(MPI_Status& stat)
{
	float currentTime, realTime;
	int nparticles, currentTimeStep;
	std::vector<ReductionType> momentum, energy;

	SimpleSerializer::deserialize(data, realTime, currentTime, currentTimeStep, nparticles, momentum, energy);

    MPI_Check( MPI_Reduce(rank == 0 ? MPI_IN_PLACE : &nparticles,     &nparticles,     1, MPI_INT,          MPI_SUM, 0, comm) );
    MPI_Check( MPI_Reduce(rank == 0 ? MPI_IN_PLACE : energy.data(),   energy.data(),   1, mpiReductionType, MPI_SUM, 0, comm) );
    MPI_Check( MPI_Reduce(rank == 0 ? MPI_IN_PLACE : momentum.data(), momentum.data(), 3, mpiReductionType, MPI_SUM, 0, comm) );

    MPI_Check( MPI_Reduce(rank == 0 ? MPI_IN_PLACE : &realTime,       &realTime,       1, MPI_FLOAT,        MPI_MAX, 0, comm) );

    if (rank == 0)
    {
    	momentum[0] /= (double)nparticles;
    	momentum[1] /= (double)nparticles;
    	momentum[2] /= (double)nparticles;
    	const ReductionType temperature = energy[0] / ( (3/2.0)*nparticles );

    	printf("Stats at timestep %d (simulation time %f):\n", currentTimeStep, currentTime);
    	printf("\tOne timespep takes %.2f ms", realTime);
    	printf("\tTotal number of particles: %d\n", nparticles);
    	printf("\tAverage momentum: [%e %e %e]\n", momentum[0], momentum[1], momentum[2]);
    	printf("\tTemperature: %.4f\n\n", temperature);
    }
}


