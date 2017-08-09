#include <core/celllist.h>
#include <core/particle_vector.h>
#include <core/initial_conditions.h>
#include <core/helper_math.h>

#include <random>

UniformIC::UniformIC(pugi::xml_node node)
{
	mass    = node.attribute("mass")   .as_float(1.0);
	density = node.attribute("density").as_float(1.0);
}

void UniformIC::exec(const MPI_Comm& comm, ParticleVector* pv, float3 globalDomainStart, float3 subDomainSize)
{
	int3 ncells = make_int3( ceilf(subDomainSize) );
	float3 h = subDomainSize / make_float3(ncells);

	float volume = h.x*h.y*h.z;
	float avg = volume * density;
	int predicted = round(avg * ncells.x*ncells.y*ncells.z * 1.05);
	pv->local()->resize(predicted);

	int rank;
	MPI_Check( MPI_Comm_rank(comm, &rank) );

	const int seed = rank + 0;
	std::mt19937 gen(seed);
	std::poisson_distribution<> particleDistribution(avg);
	std::uniform_real_distribution<float> coordinateDistribution(0, 1);

	int mycount = 0;
	auto cooPtr = pv->local()->coosvels.hostPtr();
	for (int i=0; i<ncells.x; i++)
		for (int j=0; j<ncells.y; j++)
			for (int k=0; k<ncells.z; k++)
			{
				int nparts = particleDistribution(gen);
				for (int p=0; p<nparts; p++)
				{
					pv->local()->resize(mycount+1, resizePreserve);
					cooPtr[mycount].r.x = i*h.x - 0.5*subDomainSize.x + coordinateDistribution(gen);
					cooPtr[mycount].r.y = j*h.y - 0.5*subDomainSize.y + coordinateDistribution(gen);
					cooPtr[mycount].r.z = k*h.z - 0.5*subDomainSize.z + coordinateDistribution(gen);
					cooPtr[mycount].i1 = mycount;

					cooPtr[mycount].u.x = 0*coordinateDistribution(gen);
					cooPtr[mycount].u.y = 0*coordinateDistribution(gen);
					cooPtr[mycount].u.z = 0*coordinateDistribution(gen);

					cooPtr[mycount].i1 = mycount;
					mycount++;
				}
			}

	pv->domainSize = subDomainSize;
	pv->mass = mass;

	int totalCount=0; // TODO: int64!
	MPI_Check( MPI_Exscan(&mycount, &totalCount, 1, MPI_INT, MPI_SUM, comm) );
	for (int i=0; i < pv->local()->size(); i++)
		cooPtr[i].i1 += totalCount;

	pv->local()->coosvels.uploadToDevice();

	debug2("Generated %d %s particles", pv->local()->size(), pv->name.c_str());
}






