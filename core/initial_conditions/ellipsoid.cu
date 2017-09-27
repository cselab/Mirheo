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
	std::string line;

	std::ifstream fin(fname);
	if (!fin.good())
		die("XYZ ellipsoid file %s not found", fname.c_str());
	fin >> n;

	// skip the comment line
	std::getline(fin, line);
	std::getline(fin, line);

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

	pv->globalDomainStart = globalDomainStart;
	pv->localDomainSize = localDomainSize;

	readXYZ(xyzfname, ov->initialPositions, stream);
	if (ov->objSize != ov->initialPositions.size())
		die("Object size and XYZ initial conditions don't match in size for %s", ov->name.c_str());

	std::ifstream fic(icfname);
	int nObj=0;
	auto pvView = PVview(ov, ov->local());

	while (fic.good())
	{
		LocalRigidObjectVector::RigidMotion motion{};

		fic >> motion.r.x >> motion.r.y >> motion.r.z;
		fic >> motion.q.x >> motion.q.y >> motion.q.z >> motion.q.w;

		if (!fic.good())
			break;

//		if (ov->globalDomainStart.x <= motion.r.x && motion.r.x < ov->globalDomainStart.x + ov->localDomainSize.x &&
//		    ov->globalDomainStart.y <= motion.r.y && motion.r.y < ov->globalDomainStart.y + ov->localDomainSize.y &&
//		    ov->globalDomainStart.z <= motion.r.z && motion.r.z < ov->globalDomainStart.z + ov->localDomainSize.z)
		{
			motion.r = pvView.global2local(motion.r);
			ov->local()->resize(ov->local()->size() + ov->objSize, stream);
			ov->local()->motions[nObj] = motion;
			nObj++;
		}
	}

	ov->local()->motions.uploadToDevice(stream);

	int totalCount=0; // TODO: int64!
	MPI_Check( MPI_Exscan(&nObj, &totalCount, 1, MPI_INT, MPI_SUM, comm) );
	totalCount *= ov->objSize;

	for (int i=0; i < ov->local()->size(); i++)
	{
		Particle p(make_float4(0), make_float4(0));
		p.i1 = totalCount + i;
		ov->local()->coosvels[i] = p;
	}
	ov->local()->coosvels.uploadToDevice(stream);

	info("Read %d %s objects", nObj, ov->name.c_str());

	// Do the initial rotation
	ov->local()->forces.clear(stream);
	IntegratorVVRigid integrator("dummy", 0.0f);
	integrator.stage2(pv, stream);
}

