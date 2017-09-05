#include <core/celllist.h>
#include <core/pvs/particle_vector.h>
#include <core/pvs/rigid_object_vector.h>
#include <core/initial_conditions.h>
#include <core/helper_math.h>

#include <random>

UniformIC::UniformIC(pugi::xml_node node)
{
	mass    = node.attribute("mass")   .as_float(1.0);
	density = node.attribute("density").as_float(1.0);
}

void UniformIC::exec(const MPI_Comm& comm, ParticleVector* pv, float3 globalDomainStart, float3 localDomainSize, cudaStream_t stream)
{
	int3 ncells = make_int3( ceilf(localDomainSize) );
	float3 h = localDomainSize / make_float3(ncells);

	float volume = h.x*h.y*h.z;
	float avg = volume * density;
	int predicted = round(avg * ncells.x*ncells.y*ncells.z * 1.05);
	pv->local()->resize(predicted, stream, ResizeKind::resizeAnew);

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
					pv->local()->resize(mycount+1, stream, ResizeKind::resizePreserve);
					cooPtr[mycount].r.x = i*h.x - 0.5*localDomainSize.x + coordinateDistribution(gen);
					cooPtr[mycount].r.y = j*h.y - 0.5*localDomainSize.y + coordinateDistribution(gen);
					cooPtr[mycount].r.z = k*h.z - 0.5*localDomainSize.z + coordinateDistribution(gen);
					cooPtr[mycount].i1 = mycount;

					cooPtr[mycount].u.x = 0*coordinateDistribution(gen);
					cooPtr[mycount].u.y = 0*coordinateDistribution(gen);
					cooPtr[mycount].u.z = 0*coordinateDistribution(gen);

					cooPtr[mycount].i1 = mycount;
					mycount++;
				}
			}

	pv->globalDomainStart = globalDomainStart;
	pv->localDomainSize = localDomainSize;
	pv->mass = mass;

	int totalCount=0; // TODO: int64!
	MPI_Check( MPI_Exscan(&mycount, &totalCount, 1, MPI_INT, MPI_SUM, comm) );
	for (int i=0; i < pv->local()->size(); i++)
		cooPtr[i].i1 += totalCount;

	pv->local()->coosvels.uploadToDevice(stream);

	debug2("Generated %d %s particles", pv->local()->size(), pv->name.c_str());
}

EllipsoidIC::EllipsoidIC(pugi::xml_node node)
{
	mass    = node.attribute("objmass").as_float(10.0);

	axes    = node.attribute("axes")   .as_float3({1, 1, 1});
	nObjs   = node.attribute("nobjs")  .as_int(0);
}

void EllipsoidIC::exec(const MPI_Comm& comm, ParticleVector* pv, float3 globalDomainStart, float3 localDomainSize, cudaStream_t stream)
{
	auto ov = dynamic_cast<RigidObjectVector*>(pv);
	if (ov == nullptr)
		die("Ellipsoids can only be generated out of rigid object vectors");

	auto& motions = ov->local()->motions;

	auto overlap = [&motions](float3 r, int n, float dist2) {
		for (int i=0; i<n; i++)
			if (dot(r-motions[i].r, r-motions[i].r) < dist2)
				return true;
		return false;
	};

	float maxAxis = max(axes.x, max(axes.y, axes.z));
	for (int i=0; i<nObjs; i++)
	{
		memset(&motions[i], 0, sizeof(LocalRigidObjectVector::RigidMotion));

		do {
			motions[i].r.x = localDomainSize.x*(drand48() - 0.5);
			motions[i].r.y = localDomainSize.y*(drand48() - 0.5);
			motions[i].r.z = localDomainSize.z*(drand48() - 0.5);
		} while (overlap(motions[i].r, i, (2*maxAxis+0.2)*(2*maxAxis+0.2)));

		const float phi = M_PI*drand48();
		const float sphi = sin(phi);
		const float cphi = cos(phi);

		float3 v = make_float3(drand48(), drand48(), drand48());
		v = normalize(v);

		motions[i].q = make_float4(cphi, sphi*v.x, sphi*v.y, sphi*v.z);
	}

	float3 invAxes = 1.0f / axes;

	for (int i=0; i<objSize; i++)
	{
		float4 pos;
		auto sqr = [] (float x) { return x*x; };

		do
		{
			pos.x = 2*axes.x*(drand48() - 0.5);
			pos.y = 2*axes.y*(drand48() - 0.5);
			pos.z = 2*axes.z*(drand48() - 0.5);

		} while ( sqr(pos.x * invAxes.x) + sqr(pos.y * invAxes.y) + sqr(pos.z * invAxes.z) - 1.0f > 0.0f );

		ov->initialPositions[i] = pos;
	}

	ov->local()->motions.uploadToDevice(stream);
	ov->initialPositions.uploadToDevice(stream);


}





