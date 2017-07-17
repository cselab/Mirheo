// Yo ho ho ho
#define private public
#define protected public

#include <core/particle_vector.h>
#include <core/celllist.h>
#include <core/mpi/api.h>
#include <core/logger.h>
#include <core/components.h>

#include <core/xml/pugixml.hpp>
#include <core/rigid_object_vector.h>
#include <core/rigid_kernels.h>

Logger logger;

Particle addShift(Particle p, float a, float b, float c)
{
	Particle res = p;
	res.r.x += a;
	res.r.y += b;
	res.r.z += c;

	return res;
}

float ellipsoid(LocalRigidObjectVector::RigidMotion motion, float3 invAxes, Particle p)
{
	const float3 v = p.r - motion.r;

	//https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
	const float phi = -2*acos(motion.q.x);
	const float3 k = make_float3(motion.q.y, motion.q.z, motion.q.w) / sin(phi*0.5f);

	const float3 vRot = v*cos(phi) + cross(k, v) * sin(phi) + k * dot(k, v) * (1-cos(phi));

	return sqr(vRot.x * invAxes.x) + sqr(vRot.y * invAxes.y) + sqr(vRot.z * invAxes.z) - 1.0f;
}

int main(int argc, char ** argv)
{
	// Init

	int nranks, rank;
	int ranks[] = {1, 1, 1};
	int periods[] = {1, 1, 1};
	MPI_Comm cartComm;

	MPI_Init(&argc, &argv);
	logger.init(MPI_COMM_WORLD, "bounce.log", 9);

	MPI_Check( MPI_Comm_size(MPI_COMM_WORLD, &nranks) );
	MPI_Check( MPI_Comm_rank(MPI_COMM_WORLD, &rank) );
	MPI_Check( MPI_Cart_create(MPI_COMM_WORLD, 3, ranks, periods, 0, &cartComm) );

	std::string xml = R"(<node mass="1.0" density="20.0">)";
	pugi::xml_document config;
	config.load_string(xml.c_str());

	float3 length{4,4,4};
	float3 domainStart = -length / 2.0f;
	const float rc = 1.0f;
	ParticleVector dpds("dpd");
	CellList cells(&dpds, rc, length);
	cells.setStream(0);
	cells.makePrimary();

	InitialConditions ic = createIC(config.child("node"));
	ic.exec(MPI_COMM_WORLD, &dpds, {0,0,0}, length);

	const int initialNP = dpds.local()->size();
	HostBuffer<Particle> initial(dpds.local()->size()), final(dpds.local()->size());
	const float dt = 0.1;
	for (int i=0; i<dpds.local()->size(); i++)
	{
		dpds.local()->coosvels[i].u.z = 2*(drand48() - 0.5);
		dpds.local()->coosvels[i].u.y = 2*(drand48() - 0.5);
		dpds.local()->coosvels[i].u.z = 2*(drand48() - 0.5);

		//dpds.local()->coosvels[i].r += dt * dpds.local()->coosvels[i].u;
	}

	dpds.local()->coosvels.uploadToDevice();
	cells.build();
	dpds.local()->coosvels.downloadFromDevice();
	initial.copy(dpds.local()->coosvels);


	const int nobj = 1;
	const float3 axes{0.4, 0.8, 1.0};
	const float3 invAxes = 1.0f / axes;

	const float maxAxis = std::max({axes.x, axes.y, axes.z});

	PinnedBuffer<LocalRigidObjectVector::RigidMotion> motions(nobj);
	PinnedBuffer<LocalRigidObjectVector::COMandExtent> com_ext(nobj);

	for (int i=0; i<nobj; i++)
	{
		motions[i].r.x = length.x*(drand48() - 0.5);
		motions[i].r.y = length.y*(drand48() - 0.5);
		motions[i].r.z = length.z*(drand48() - 0.5);

		motions[i].omega.x = 2*(drand48() - 0.5);
		motions[i].omega.y = 2*(drand48() - 0.5);
		motions[i].omega.z = 2*(drand48() - 0.5);

		motions[i].vel.x = 2*(drand48() - 0.5);
		motions[i].vel.y = 2*(drand48() - 0.5);
		motions[i].vel.z = 2*(drand48() - 0.5);

		motions[i].force  = make_float3(0);
		motions[i].torque = make_float3(0);

		const float phi = M_PI*drand48();
		const float sphi = sin(0.5f*phi);
		const float cphi = cos(0.5f*phi);

		float3 v = make_float3(drand48(), drand48(), drand48());
		//float3 v = make_float3(1, 0, 0);
		v = normalize(v);

		motions[i].q = make_float4(cphi, sphi*v.x, sphi*v.y, sphi*v.z);

		printf("Obj %d:\n"
				"   r [%f %f %f]\n\n", i, motions[i].r.x, motions[i].r.y, motions[i].r.z);

		com_ext[i].com  = motions[i].r;
		com_ext[i].high = com_ext[i].com + make_float3(maxAxis);
		com_ext[i].low  = com_ext[i].com - make_float3(maxAxis);
	}

	motions.uploadToDevice();
	com_ext.uploadToDevice();



	bounceEllipsoid<<< nobj, 32 >>> ((float4*)dpds.local()->coosvels.devPtr(), dpds.mass, com_ext.devPtr(), motions.devPtr(),
			nobj, invAxes,
			cells.cellsStartSize.devPtr(), cells.cellInfo(), dt);

//			(float4* coosvels, float mass, const LocalObjectVector::COMandExtent* props, LocalRigidObjectVector::RigidMotion* motions,
//			const int nObj, const float3 invAxes,
//			const uint* __restrict__ cellsStartSize, CellListInfo cinfo, const float dt)

	dpds.local()->coosvels.downloadFromDevice(true);


	for (int objId = 0; objId < nobj; objId++)
	{
		auto motion = motions[objId];

		for (int pid = 0; pid < final.size(); pid++)
		{
			auto pInit  = initial[pid];
			auto pFinal = dpds.local()->coosvels[pid];

			Particle pOld = pInit;
			pOld.r = pInit.r - dt*(pInit.u-motion.vel);

			// Inside
			//if (ellipsoid(motion, invAxes, pOld) * ellipsoid(motion, invAxes, pInit) < 0)
			if (pInit.i1 == 663)
			{
				printf("Particle  %d,  obj  %d:\n"
					   "   [%f %f %f] (%f)  -->  [%f %f %f] (%f)\n"
					   "   Moved to [%f %f %f] (%f)\n\n",
						pInit.i1, objId,
						pOld.r.x,   pOld.r.y,   pOld.r.z,   ellipsoid(motion, invAxes, pOld),
						pInit.r.x,  pInit.r.y,  pInit.r.z,  ellipsoid(motion, invAxes, pInit),
						pFinal.r.x, pFinal.r.y, pFinal.r.z, ellipsoid(motion, invAxes, pFinal) );
			}
		}
	}


	return 0;
}
