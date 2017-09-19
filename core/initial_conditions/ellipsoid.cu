#include "ellipsoid.h"

#include <random>

#include <core/pvs/particle_vector.h>
#include <core/pvs/rigid_object_vector.h>
#include <core/integrators/rigid_vv.h>

EllipsoidIC::readXYZ(std::string fname, PinnedBuffer<float4>& positions)
{
	int n;
	float dummy;

	std::ifstream fin(fname);
	fin >> n;

	particles.resize(n, stream, ResizeKind::resizeAnew);
	for (int i=0; i<n; i++)
		fin >> dummy >> positions[i].x >>positions[i].y >>positions[i].z;
}

void EllipsoidIC::exec(const MPI_Comm& comm, ParticleVector* pv, float3 globalDomainStart, float3 localDomainSize, cudaStream_t stream)
{
	auto ov = dynamic_cast<RigidObjectVector*>(pv);
	if (ov == nullptr)
		die("Ellipsoids can only be generated out of rigid object vectors");

	const int seed = rank + 0;
	std::mt19937 gen(seed);
	std::poisson_distribution<> particleDistribution(avg);
	std::uniform_real_distribution<float> udistr(0, 1);

	auto& motions = ov->local()->motions;

	auto overlap = [&motions](float3 r, int n, float dist2) {
		for (int i=0; i<n; i++)
			if (dot(r-motions[i].r, r-motions[i].r) < dist2)
				return true;
		return false;
	};

	int generated = 0;
	float maxAxis = max(axes.x, max(axes.y, axes.z));
	for (int i=0; i<nObjs; i++)
	{
		motions.resize(generated+1, stream, ResizeKind::resizePreserve);
		memset(&motions[i], 0, sizeof(LocalRigidObjectVector::RigidMotion));

		for (int tries=0; tries<10000; tries++)
		{
			motions[i].r.x = (localDomainSize.x - 2*maxAxis - 2*distance) * (udistr() - 0.5);
			motions[i].r.y = (localDomainSize.x - 2*maxAxis - 2*distance) * (udistr() - 0.5);
			motions[i].r.z = (localDomainSize.x - 2*maxAxis - 2*distance) * (udistr() - 0.5);

			if ( !overlap(motions[i].r, i, (2*maxAxis+0.2)*(2*maxAxis+0.2)) )
			{
				generated++;
				break;
			}
		}

		const float phi = M_PI*udistr();
		const float sphi = sin(phi);
		const float cphi = cos(phi);

		float3 v = make_float3(udistr(), udistr(), udistr());
		v = normalize(v);

		motions[i].q = make_float4(cphi, sphi*v.x, sphi*v.y, sphi*v.z);
	}

	readXYZ(xyzfname, ov->initialPositions);

	if (objSize != ov->initialPositions.size())
		die("Object size and XYZ initial conditions don't match in size for %s", ov->name.c_str());

	ov->local()->resize(generated * objSize, stream, ResizeKind::resizePreserve);
	ov->local()->motions.uploadToDevice(stream);
	ov->initialPositions.uploadToDevice(stream);

	// Do the initial rotation
	IntegratorVVRigid integrator(pugi::node_null);
	integrator.dt = 0.0f;
	integrator.stage2(pv, stream);
}

