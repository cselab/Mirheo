#include <plugins/stats.h>
#include <core/datatypes.h>
#include <core/containers.h>
#include <core/simulation.h>

void SimulationStats::afterIntegration(bool& reordered)
{
	reordered = false;
	if (currentTimeStep % fetchEvery != 0) return;

	auto& pvs = sim->getParticleVectors();

	int total = 0;
	for (auto& pv : pvs)
		total += pv->np;

	allParticles.clear();
	allParticles.reserve(total*1.1);

	for (auto& pv : pvs)
	{
		pv->coosvels.downloadFromDevice(true);
		auto partPtr = pv->coosvels.hostPtr();

		for (int i=0; i < pv->np; i++)
			allParticles.push_back(partPtr[i]);
	}

	needToDump = true;
}

void SimulationStats::serializeAndSend()
{
	if (needToDump)
	{
		send(allParticles.data(), sizeof(Particle) * allParticles.size());
		needToDump = false;
	}
}

void SimulationStats::handshake()
{
	auto& pvs = sim->getParticleVectors();

	int total = 0;
	for (auto& pv : pvs)
		total += pv->np;

	total *= sizeof(Particle);

	// TODO: variable size maybe?
	MPI_Check( MPI_Send(&total, 1, MPI_INT, rank, id, interComm) );
}





void PostprocessStats::deserialize(MPI_Status& stat)
{
	int recvBytes;
	MPI_Check( MPI_Get_count(&stat, MPI_BYTE, &recvBytes) );
	Particle* parts = (Particle*)data.hostPtr();
	int np = recvBytes / sizeof(Particle);

	double3 momentum{0,0,0};
	double  energy = 0;

	for (int i=0; i<np; i++)
	{
		Particle& p = parts[i];
		momentum.x += p.u[0];
		momentum.y += p.u[1];
		momentum.z += p.u[2];

		//printf("%f %f %f\n", p.u[0], p.u[1], p.u[2]);

		energy += p.u[0]*p.u[0] + p.u[1]*p.u[1] + p.u[2]*p.u[2];
	}

    MPI_Check( MPI_Reduce(rank == 0 ? MPI_IN_PLACE : &np,       &np,       1, MPI_INT,    MPI_SUM, 0, comm) );
    MPI_Check( MPI_Reduce(rank == 0 ? MPI_IN_PLACE : &energy,   &energy,   1, MPI_DOUBLE, MPI_SUM, 0, comm) );
    MPI_Check( MPI_Reduce(rank == 0 ? MPI_IN_PLACE : &momentum, &momentum, 3, MPI_DOUBLE, MPI_SUM, 0, comm) );

    if (rank == 0)
    {
    	momentum.x /= np;
    	momentum.y /= np;
    	momentum.z /= np;
    	const double temperature = energy / (3*np);

    	int currentTimeStep = 0;
    	float currentTime = 0;
    	printf("Stats at timestep %d (simulation time %f):\n", currentTimeStep, currentTime);
    	printf("\tTotal number of particles: %d\n", np);
    	printf("\tAverage momentum: [%e %e %e]\n", momentum.x, momentum.y, momentum.z);
    	printf("\tTemperature: %.4f\n\n", temperature);
    }
}

void PostprocessStats::handshake()
{
	MPI_Check( MPI_Recv(&size, 1, MPI_INT, rank, id, interComm, MPI_STATUS_IGNORE) );
	size = size*2 + 32000;
	data.resize(size);
}



