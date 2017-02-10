#include <plugins/stats.h>
#include <core/datatypes.h>
#include <core/containers.h>
#include <core/simulation.h>

void SimulationStats::afterIntegration(float t)
{
	if (invoked % fetchEvery != 0) return;

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
}

void SimulationStats::serializeAndSend()
{
	invoked++;
	if (invoked % fetchEvery != 0) return;

	send(allParticles.data(), sizeof(Particle) * allParticles.size());
}

void SimulationStats::handshake()
{
	auto& pvs = sim->getParticleVectors();

	int total = 0;
	for (auto& pv : pvs)
		total += pv->np;

	total *= sizeof(Particle);

	MPI_Check( MPI_Send(&total, 1, MPI_INT, rank, id, interComm) );
}





void PostprocessStats::deserialize(MPI_Status& stat)
{
	int recvBytes;
	MPI_Check( MPI_Get_count(&stat, MPI_BYTE, &recvBytes) );
	Particle* parts = (Particle*)data.hostPtr();
	int np = recvBytes / sizeof(Particle);

	float3 momentum{0,0,0};
	float  temp = 0;

	for (int i=0; i<np; i++)
	{
		Particle& p = parts[i];
		momentum.x += p.u[0];
		momentum.y += p.u[1];
		momentum.z += p.u[2];

		//printf("%f %f %f\n", p.u[0], p.u[1], p.u[2]);

		temp += p.u[0]*p.u[0] + p.u[1]*p.u[1] + p.u[2]*p.u[2];
	}

	momentum;
	temp /= 3*np;

	printf("[%f %f %f]  %f\n", momentum.x, momentum.y, momentum.z, temp);
}

void PostprocessStats::handshake()
{
	MPI_Check( MPI_Recv(&size, 1, MPI_INT, rank, id, interComm, MPI_STATUS_IGNORE) );
	size = size*2 + 32000;
	data.resize(size);
}



