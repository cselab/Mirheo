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
#include <core/rigid_kernels/bounce.h>

Logger logger;

Particle addShift(Particle p, float a, float b, float c)
{
	Particle res = p;
	res.r.x += a;
	res.r.y += b;
	res.r.z += c;

	return res;
}

float4 inv_q(float4 q)
{
	return make_float4(q.x, -q.y, -q.z, -q.w);
}

float3 rot(float3 v, float4 q)
{
	//https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula

	double phi = 2.0*atan2( sqrt( (double)q.y*q.y + (double)q.z*q.z + (double)q.w*q.w),  (double)q.x );
	double sphi_1 = 1.0 / sin(0.5*phi);
	const float3 k = make_float3(q.y * sphi_1, q.z * sphi_1, q.w * sphi_1);

	return v*cos(phi) + cross(k, v) * sin(phi) + k * dot(k, v) * (1-cos(phi));
}

float ellipsoid(LocalRigidObjectVector::RigidMotion motion, float3 invAxes, Particle p)
{
	const float3 v = p.r - motion.r;
	const float3 vRot = rot(v, inv_q(motion.q));

//	if (p.i1 == 352474)
//		printf(";lasjjjjj     [%f %f %f] --> [%f %f %f] -rot-> [%f %f %f]          %f\n",
//				p.r.x, p.r.y, p.r.z, v.x, v.y, v.z, vRot.x, vRot.y, vRot.z, dot(motion.q, motion.q));

	return sqr(vRot.x * invAxes.x) + sqr(vRot.y * invAxes.y) + sqr(vRot.z * invAxes.z) - 1.0f;
}

bool overlap(float3 r, LocalRigidObjectVector::RigidMotion* motions, int n, float dist2)
{
	for (int i=0; i<n; i++)
		if (dot(r-motions[i].r, r-motions[i].r) < dist2)
			return true;

	return false;
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

	std::string xml = R"(<node mass="1.0" density="10.0">)";
	pugi::xml_document config;
	config.load_string(xml.c_str());

	float3 length{40,40,40};
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
		dpds.local()->coosvels[i].u.x = 2*(drand48() - 0.5);
		dpds.local()->coosvels[i].u.y = 2*(drand48() - 0.5);
		dpds.local()->coosvels[i].u.z = 2*(drand48() - 0.5);

		//dpds.local()->coosvels[i].r += dt * dpds.local()->coosvels[i].u;
	}

	dpds.local()->coosvels.uploadToDevice();
	cells.build();
	dpds.local()->coosvels.downloadFromDevice();
	initial.copy(dpds.local()->coosvels);


	const int nobj = 100;
	const float3 axes{3, 2, 2.5};
	const float3 invAxes = 1.0f / axes;

	const float maxAxis = std::max({axes.x, axes.y, axes.z});

	PinnedBuffer<LocalRigidObjectVector::RigidMotion> motions(nobj);
	PinnedBuffer<LocalRigidObjectVector::COMandExtent> com_ext(nobj);

	for (int i=0; i<nobj; i++)
	{
		do {
			motions[i].r.x = length.x*(drand48() - 0.5);
			motions[i].r.y = length.y*(drand48() - 0.5);
			motions[i].r.z = length.z*(drand48() - 0.5);
		} while (overlap(motions[i].r, motions.hostPtr(), i, (2*maxAxis+0.2)*(2*maxAxis+0.2)));


		motions[i].omega.x = 2*(drand48() - 0.5);
		motions[i].omega.y = 2*(drand48() - 0.5);
		motions[i].omega.z = 2*(drand48() - 0.5);

		motions[i].vel.x = 2*(drand48() - 0.5);
		motions[i].vel.y = 2*(drand48() - 0.5);
		motions[i].vel.z = 2*(drand48() - 0.5);

		motions[i].force  = make_float3(0);
		motions[i].torque = make_float3(0);

		const float phi = 0.4*M_PI*drand48()+0.1;
		const float sphi = sin(0.5f*phi);
		const float cphi = cos(0.5f*phi);

		float3 v = make_float3(drand48(), drand48(), drand48());
		//float3 v = make_float3(1, 0, 0);
		v = normalize(v);

		motions[i].q = make_float4(cphi, sphi*v.x, sphi*v.y, sphi*v.z);

		com_ext[i].com  = motions[i].r;
		com_ext[i].high = com_ext[i].com + make_float3(maxAxis);
		com_ext[i].low  = com_ext[i].com - make_float3(maxAxis);

//		printf("Obj %d:\n"
//				"   r [%f %f %f]\n"
//				"   phi %f, v [%f %f %f]\n"
//				"   ext : [%f %f %f] -- [%f %f %f]\n\n",
//				i, motions[i].r.x, motions[i].r.y, motions[i].r.z,
//				phi, v.x, v.y, v.z,
//				com_ext[i].low.x,  com_ext[i].low.y,  com_ext[i].low.z,
//				com_ext[i].high.x, com_ext[i].high.y, com_ext[i].high.z);
	}

	printf(" =================================================\n\n");

	motions.uploadToDevice();
	com_ext.uploadToDevice();



	for (int iter=0; iter<100; iter++)
	{
		bounceEllipsoid<<< nobj, 128 >>> ((float4*)dpds.local()->coosvels.devPtr(), dpds.mass, com_ext.devPtr(), motions.devPtr(),
				nobj, invAxes,
				cells.cellsStartSize.devPtr(), cells.cellInfo(), dt);
	}

	return 0;

//			(float4* coosvels, float mass, const LocalObjectVector::COMandExtent* props, LocalRigidObjectVector::RigidMotion* motions,
//			const int nObj, const float3 invAxes,
//			const uint* __restrict__ cellsStartSize, CellListInfo cinfo, const float dt)

	dpds.local()->coosvels.downloadFromDevice(true);


	for (int objId = 0; objId < nobj; objId++)
	{
		auto motion = motions[objId];
		auto oldMot = motion;

		float4 dq_dt = compute_dq_dt(motion.q, motion.omega);
		oldMot.q = motion.q - dq_dt * dt;
		oldMot.r = motion.r - motion.vel * dt;
		oldMot.q = normalize(oldMot.q);

//		printf("Obj %d: old  q [%f %f %f %f],  r [%f %f %f]\n",
//				objId, oldMot.q.x, oldMot.q.y, oldMot.q.z, oldMot.q.w,  oldMot.r.x, oldMot.r.y, oldMot.r.z);

#pragma omp parallel for
		for (int pid = 0; pid < final.size(); pid++)
		{
			auto pInit  = initial[pid];
			auto pFinal = dpds.local()->coosvels[pid];

			Particle pOld = pInit;
			pOld.r = pInit.r - dt*pInit.u;

			float vold  = ellipsoid(oldMot, invAxes, pOld);
			float vinit = ellipsoid(motion, invAxes, pInit);

			// Inside
			if ( vold * vinit < 0 && vinit < 0 )
			{
				float vfin  = ellipsoid(motion, invAxes, pFinal);

				bool wrong = vfin < 0.0f;

				float3 r = pFinal.r;
				float3 v = pInit.u;

				r = rot(r - motion.r,   inv_q(motion.q));
				v = rot(v - motion.vel, inv_q(motion.q));

				v = v - cross(motion.omega, r);

				v = -v;
				v = v + cross(motion.omega, r);
				v = rot(v, motion.q) + motion.vel;

				wrong = wrong || dot(v - pFinal.u, v - pFinal.u) > 1e-4;

				if (wrong)
				{
					int3 cid = cells.getCellIdAlongAxis(pInit.r);

#pragma omp critical
					printf("Particle  %d (cell %d %d %d),  obj  %d:\n"
						   "   [%f %f %f] (%f)  -->  [%f %f %f] (%f)\n"
						   "   Moved to [%f %f %f] (%f)\n"
						   "   Vel from [%f %f %f] to [%f %f %f], reference vel [%f %f %f]\n\n",
							pInit.i1, cid.x, cid.y, cid.z, objId,
							pOld.r.x,   pOld.r.y,   pOld.r.z,   vold,
							pInit.r.x,  pInit.r.y,  pInit.r.z,  vinit,
							pFinal.r.x, pFinal.r.y, pFinal.r.z, vfin,
							pInit.u.x,  pInit.u.y,  pInit.u.z, pFinal.u.x,  pFinal.u.y,  pFinal.u.z,  v.x, v.y, v.z);

					//return 0;
				}
			}
		}
	}


	return 0;
}
