#include "ellipsoid.h"

#include <random>
#include <fstream>

#include <core/pvs/particle_vector.h>
#include <core/pvs/rigid_ellipsoid_object_vector.h>
#include <core/integrators/rigid_vv.h>

void EllipsoidIC::readXYZ(std::string fname, PinnedBuffer<float4>& positions, cudaStream_t stream)
{
	int n;
	float dummy;

	std::ifstream fin(fname);
	fin >> n;

	positions.resize(n, stream, ResizeKind::resizeAnew);
	for (int i=0; i<n; i++)
		fin >> dummy >> positions[i].x >>positions[i].y >>positions[i].z;

	positions.uploadToDevice(stream);
}

void EllipsoidIC::exec(const MPI_Comm& comm, ParticleVector* pv, float3 globalDomainStart, float3 localDomainSize, cudaStream_t stream)
{
	auto ov = dynamic_cast<RigidEllipsoidObjectVector*>(pv);
	if (ov == nullptr)
		die("Ellipsoids can only be generated out of rigid object vectors");

	int rank;
	MPI_Check( MPI_Comm_rank(comm, &rank) );

	const int seed = rank + 0;
	std::mt19937 gen(seed);
	std::uniform_real_distribution<float> udistr(0, 1);

	auto& motions = ov->local()->motions;

	auto overlap = [&motions](float3 r, int n, float dist2) {
		for (int i=0; i<n; i++)
			if (dot(r-motions[i].r, r-motions[i].r) < dist2)
				return true;
		return false;
	};

	int generated = 0;
	float maxAxis = max(ov->axes.x, max(ov->axes.y, ov->axes.z));
	while (true)
	{
		int tries = 0;
		int maxTries = 1000;
		LocalRigidObjectVector::RigidMotion motion = {};

		for (tries=0; tries<maxTries; tries++)
		{
			motion.r.x = (localDomainSize.x - 2*maxAxis - 2*separation) * (udistr(gen) - 0.5);
			motion.r.y = (localDomainSize.x - 2*maxAxis - 2*separation) * (udistr(gen) - 0.5);
			motion.r.z = (localDomainSize.x - 2*maxAxis - 2*separation) * (udistr(gen) - 0.5);

			if ( !overlap(motion.r, generated, (2*maxAxis+separation)*(2*maxAxis+separation)) )
			{
				generated++;
				break;
			}
		}
		if (tries >= maxTries)
			break;

		const float phi = M_PI*udistr(gen);
		const float sphi = sin(phi);
		const float cphi = cos(phi);

		float3 v = make_float3(udistr(gen), udistr(gen), udistr(gen));
		v = normalize(v);

		motions.resize(generated, stream, ResizeKind::resizePreserve);
		motions[generated-1].q = make_float4(cphi, sphi*v.x, sphi*v.y, sphi*v.z);
		motions[generated-1] = motion;

	}
	ov->local()->motions.uploadToDevice(stream);

	ov->local()->resize(generated * ov->objSize, stream, ResizeKind::resizePreserve);

	int totalCount=0; // TODO: int64!
	MPI_Check( MPI_Exscan(&generated, &totalCount, 1, MPI_INT, MPI_SUM, comm) );
	for (int i=0; i < ov->local()->size(); i++)
		ov->local()->coosvels[i].i1 = totalCount + i;

	readXYZ(xyzfname, ov->initialPositions, stream);
	if (ov->objSize != ov->initialPositions.size())
		die("Object size and XYZ initial conditions don't match in size for %s", ov->name.c_str());

	// Do the initial rotation
	IntegratorVVRigid integrator("dummy", 0.0f);
	integrator.stage2(pv, stream);
}

