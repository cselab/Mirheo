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

	positions.resize_anew(n);
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
	int nObjs=0;
	PVview pvView(ov, ov->local());

	HostBuffer<LocalRigidObjectVector::RigidMotion> motions;

	while (fic.good())
	{
		LocalRigidObjectVector::RigidMotion motion{};

		fic >> motion.r.x >> motion.r.y >> motion.r.z;
		fic >> motion.q.x >> motion.q.y >> motion.q.z >> motion.q.w;

		if (!fic.good())
			break;

		if (ov->globalDomainStart.x <= motion.r.x && motion.r.x < ov->globalDomainStart.x + ov->localDomainSize.x &&
		    ov->globalDomainStart.y <= motion.r.y && motion.r.y < ov->globalDomainStart.y + ov->localDomainSize.y &&
		    ov->globalDomainStart.z <= motion.r.z && motion.r.z < ov->globalDomainStart.z + ov->localDomainSize.z)
		{
			motion.r = pvView.global2local(motion.r);
			motions.resize(nObjs + 1);
			motions[nObjs] = motion;
			nObjs++;
		}
	}

	ov->local()->resize_anew(nObjs * ov->objSize);

	auto ovMotions = ov->local()->getDataPerObject<LocalRigidObjectVector::RigidMotion>("motions");
	ovMotions->copy(motions);
	ovMotions->uploadToDevice(stream);

	// Set ids
	int totalCount=0; // TODO: int64!
	MPI_Check( MPI_Exscan(&nObjs, &totalCount, 1, MPI_INT, MPI_SUM, comm) );

	auto ids = ov->local()->getDataPerObject<int>("ids");
	for (int i=0; i<nObjs; i++)
		(*ids)[i] = totalCount + i;


	for (int i=0; i < ov->local()->size(); i++)
	{
		Particle p(make_float4(0), make_float4(0));
		p.i1 = totalCount*ov->objSize + i;
		ov->local()->coosvels[i] = p;
	}

	ids->uploadToDevice(stream);
	ov->local()->coosvels.uploadToDevice(stream);

	info("Read %d %s objects", nObjs, ov->name.c_str());

	// Do the initial rotation
	ov->local()->forces.clear(stream);
	IntegratorVVRigid integrator("dummy", 0.0f);
	integrator.stage2(pv, 0, stream);

//	//for (auto& ov: objectVectors)
//	{
//		ov->local()->coosvels.downloadFromDevice(stream);
//		for (int i=0; i<5; i++)
//			printf("%d  %f %f %f\n", i, ov->local()->coosvels[i].r.x, ov->local()->coosvels[i].r.y, ov->local()->coosvels[i].r.z);
//	}
}

