// Yo ho ho ho
#define private public
#define protected public

#include <core/pvs/particle_vector.h>
#include <core/pvs/rigid_object_vector.h>
#include <core/celllist.h>
#include <core/mpi/api.h>
#include <core/logger.h>
#include <core/integrate.h>
#include <core/interactions.h>
#include <core/bounce.h>

#include "timer.h"
#include <unistd.h>

Logger logger;

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

float ellipsoid(LocalRigidObjectVector::RigidMotion motion, float3 invAxes, float3 r)
{
	const float3 v = r - motion.r;
	const float3 vRot = rot(v, inv_q(motion.q));

	return sqr(vRot.x * invAxes.x) + sqr(vRot.y * invAxes.y) + sqr(vRot.z * invAxes.z) - 1.0f;
}

__inline__ __device__ float warpReduceSum(float val)
{
#pragma unroll
	for (int offset = warpSize/2; offset > 0; offset /= 2)
	{
		val += __shfl_down(val, offset);
	}
	return val;
}

__inline__ __device__ float3 warpReduceSum(float3 val)
{
#pragma unroll
	for (int offset = warpSize/2; offset > 0; offset /= 2)
	{
		val.x += __shfl_down(val.x, offset);
		val.y += __shfl_down(val.y, offset);
		val.z += __shfl_down(val.z, offset);
	}
	return val;
}

__global__ void totalMomentumEnergy(const float4* coosvels, const float mass, int n, double* momentum, double* energy)
{
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	const int wid = tid % warpSize;
	if (tid >= n) return;

	const float3 vel = make_float3(coosvels[2*tid+1]);

	float3 myMomentum = vel*mass;
	float myEnergy = dot(vel, vel) * mass*0.5f;

	myMomentum = warpReduceSum(myMomentum);
	myEnergy   = warpReduceSum(myEnergy);

	if (wid == 0)
	{
		atomicAdd(momentum+0, (double)myMomentum.x);
		atomicAdd(momentum+1, (double)myMomentum.y);
		atomicAdd(momentum+2, (double)myMomentum.z);
		atomicAdd(energy,     (double)myEnergy);
	}
}

int main(int argc, char ** argv)
{
	// Init

	int nranks, rank;
	int ranks[] = {1, 1, 1};
	int periods[] = {1, 1, 1};
	MPI_Comm cartComm;

	int provided;
	MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
	if (provided < MPI_THREAD_MULTIPLE)
	{
	    printf("ERROR: The MPI library does not have full thread support\n");
	    MPI_Abort(MPI_COMM_WORLD, 1);
	}

	logger.init(MPI_COMM_WORLD, "rigid.log", 9);
	srand48(2);

	MPI_Check( MPI_Comm_size(MPI_COMM_WORLD, &nranks) );
	MPI_Check( MPI_Comm_rank(MPI_COMM_WORLD, &rank) );
	MPI_Check( MPI_Cart_create(MPI_COMM_WORLD, 3, ranks, periods, 0, &cartComm) );

	cudaStream_t defStream;
	CUDA_Check( cudaStreamCreateWithPriority(&defStream, cudaStreamNonBlocking, 10) );

	float3 length{32, 32, 32};
	float3 domainStart = -length / 2.0f;
	const float rc = 1.0f;
	const int ndens = 8;

	float3 axes{3, 4, 5};
	float3 invAxes = 1.0 / axes;
	const float maxAxis = std::max({axes.x, axes.y, axes.z});

	int objSize = 4/3.0 * M_PI * axes.x*axes.y*axes.z * ndens + 1;
	RigidObjectVector obj("obj", objSize, 1);

	// Init object
	obj.initialPositions.resize(objSize, defStream);
	memset(&obj.local()->motions[0], 0, sizeof(LocalRigidObjectVector::RigidMotion));
	obj.local()->motions[0].r = make_float3(0);

	obj.local()->motions[0].omega.x = 0;//0*(drand48() - 0.5);
	obj.local()->motions[0].omega.y = 0;//0*(drand48() - 0.5);
	obj.local()->motions[0].omega.z = 10;//0*(drand48() - 0.5);

	obj.local()->motions[0].vel.x = 0*(drand48() - 0.5);
	obj.local()->motions[0].vel.y = 0*(drand48() - 0.5);
	obj.local()->motions[0].vel.z = 0*(drand48() - 0.5);

	obj.local()->motions[0].force  = make_float3(0);
	obj.local()->motions[0].torque = make_float3(0);

	const float phi = M_PI*drand48()*0.001;
	const float sphi = sin(0.5f*phi);
	const float cphi = cos(0.5f*phi);

	float3 v = make_float3(drand48(), drand48(), drand48());
	v = normalize(v);

	float4 q = obj.local()->motions[0].q = make_float4(cphi, sphi*v.x, sphi*v.y, sphi*v.z);

	LocalRigidObjectVector::RigidMotion m;
	m.r = make_float3(0);
	m.q = make_float4(0.999, 0, 0, 0.0447101778);
	m.vel = make_float3(0);

	for (int i=0; i<obj.objSize; i++)
	{
		float4 pos;

		do
		{
			pos.x = 2*maxAxis*(drand48() - 0.5);
			pos.y = 2*maxAxis*(drand48() - 0.5);
			pos.z = 2*maxAxis*(drand48() - 0.5);

		} while ( ellipsoid(m, invAxes, f4tof3(pos)) > 0 );

		obj.initialPositions[i] = pos;
	}

	for (int i=0; i<obj.local()->size(); i++)
	{
		Particle p;
		p.i1 = 0;
		p.s21 = (short)i;

		p.r = rot( f4tof3(obj.initialPositions[i % obj.objSize]), obj.local()->motions[i / objSize].q);
		p.u = make_float3(0);

		obj.local()->coosvels[i] = p;
	}

	obj.initialPositions. uploadToDevice(defStream);
	obj.local()->motions. uploadToDevice(defStream);
	obj.local()->coosvels.uploadToDevice(defStream);
	obj.local()->findExtentAndCOM(0);

	obj.objMass = objSize * 1.0f;
	obj.mass = 1.0f;
	obj.axes = axes;

	HostBuffer<LocalRigidObjectVector::COMandExtent> com_ext;
	com_ext.copy(obj.local()->comAndExtents, 0);

	printf("Object: \n"
			"   phi %f, v [%f %f %f]\n"
			"   ext : [%f %f %f] -- [%f %f %f]\n\n",
			phi, v.x, v.y, v.z,
			com_ext[0].low.x,  com_ext[0].low.y,  com_ext[0].low.z,
			com_ext[0].high.x, com_ext[0].high.y, com_ext[0].high.z);


	ParticleVector dpds("dpd");
	CellList *cells = new PrimaryCellList(&dpds, rc, length);
	CellList *objCells = new CellList(&obj, rc, length);

	dpds.local()->resize(cells->ncells.x*cells->ncells.y*cells->ncells.z * ndens, defStream);

	printf("initializing...\n");

	auto motion = obj.local()->motions[0];
	int c = 0;
	float3 totU = make_float3(0);
	for (int i=0; i<cells->ncells.x; i++)
		for (int j=0; j<cells->ncells.y; j++)
			for (int k=0; k<cells->ncells.z; k++)
				for (int p=0; p<ndens; p++)
				{
					dpds.local()->coosvels[c].r.x = i + drand48() + domainStart.x;
					dpds.local()->coosvels[c].r.y = j + drand48() + domainStart.y;
					dpds.local()->coosvels[c].r.z = k + drand48() + domainStart.z;

					if (ellipsoid(motion, invAxes, dpds.local()->coosvels[c].r) < 0.02)
						continue;

					dpds.local()->coosvels[c].i1 = c;

					dpds.local()->coosvels[c].u.x = 1*(drand48() - 0.5);
					dpds.local()->coosvels[c].u.y = 1*(drand48() - 0.5);
					dpds.local()->coosvels[c].u.z = 1*(drand48() - 0.5);

					totU += dpds.local()->coosvels[c].u;

					c++;
				}

	totU /= c;
	for (int i=0; i<c; i++)
		dpds.local()->coosvels[i].u -= totU;

	printf("generated %d particles\n", c);
	dpds.local()->resize(c, defStream);
	dpds.domainSize = length;
	dpds.mass = 1.0f;
	dpds.local()->coosvels.uploadToDevice(defStream);

	ParticleHaloExchanger halo(cartComm);
	halo.attach(&dpds, cells);
	ParticleRedistributor redist(cartComm);
	redist.attach(&dpds, cells);

	CUDA_Check( cudaStreamSynchronize(defStream) );

	const int niters = 4200000;
	const float dt = 0.002;

	std::string xml = R"(<interaction name="dpd" kbt="1.0" gamma="20" a="50" dt="0.002"/>
    <integrate dt="0.002"/>)";
	pugi::xml_document config;
	config.load_string(xml.c_str());

	Interaction *inter = new InteractionDPD(config.child("interaction"));
	Integrator  *noflow = new IntegratorVVNoFlow(config.child("integrate"));
	Integrator  *rigInt = new IntegratorVVRigid(config.child("integrate"));

	printf("GPU execution\n");

	Timer tm;
	tm.start();

	HostBuffer<Force> frcs;

	PinnedBuffer<double> energy(1), momentum(3);

	cudaDeviceSynchronize();

	auto prnCoosvels = [defStream] (ParticleVector* pv) {
		pv->local()->coosvels.downloadFromDevice(defStream, true);
		auto ptr = pv->local()->coosvels.hostPtr();
		for (int j=0; j<pv->local()->size(); j++)
		{
			if (ptr[j].s21 == 42)
				printf("??? %4d :  [%f %f %f] [%f %f %f]\n", ptr[j].s21, ptr[j].r.x, ptr[j].r.y, ptr[j].r.z, ptr[j].u.x, ptr[j].u.y, ptr[j].u.z);
		}
	};

	const int nparticles = dpds.local()->size() + obj.local()->size();
	for (int i=0; i<niters; i++)
	{
		energy.clear(defStream);
		momentum.clear(defStream);

		totalMomentumEnergy<<< getNblocks(dpds.local()->size(), 128), 128, 0, defStream >>> (
				(float4*)dpds.local()->coosvels.devPtr(), dpds.mass, dpds.local()->size(), momentum.devPtr(), energy.devPtr());

		totalMomentumEnergy<<< getNblocks(obj.local()->size(), 128), 128, 0, defStream >>> (
				(float4*)obj.local()->coosvels.devPtr(), 1.0f, obj.local()->size(), momentum.devPtr(), energy.devPtr());

		momentum.downloadFromDevice(defStream, false);
		energy.  downloadFromDevice(defStream, true);

		if (i % 100 == 0)
		{
			printf("Iteration %d, temp %f, momentum  %.2e %.2e %.2e\n",
					i, energy[0]/ ( (3/2.0)*nparticles ), momentum[0] / nparticles, momentum[1] / nparticles, momentum[2] / nparticles);

			obj.local()->motions.downloadFromDevice(defStream, true);
			auto motion = obj.local()->motions[0];

			printf("obj  %d  f [%f %f %f],  t [%f %f %f],  r [%f %f %f]   v [%f %f %f] \n"
					"    q [%f %f %f %f]   w [%f %f %f] \n", 0,
					motion.force.x,  motion.force.y,  motion.force.z,
					motion.torque.x, motion.torque.y, motion.torque.z,
					motion.r.x,  motion.r.y,  motion.r.z,
					motion.vel.x,  motion.vel.y,  motion.vel.z,
					motion.q.x,  motion.q.y,  motion.q.z, motion.q.w,
					motion.omega.x,  motion.omega.y,  motion.omega.z);
		}

		cells->build(defStream);
		objCells->build(defStream);

		dpds.local()->forces.clear(defStream);
		obj.local()->forces.clear(defStream);

		cells->forces->clear(defStream);
		objCells->forces->clear(defStream);

		halo.init(defStream);
		inter->regular(&dpds, &dpds, cells, cells,    dt*i, defStream);
		inter->regular(&dpds, &obj,  cells, objCells, dt*i, defStream);
		halo.finalize();

//		CUDA_Check( cudaStreamSynchronize(defStream) );

		inter->halo(&dpds, &dpds, cells, cells, dt*i, defStream);

		objCells->addForces(defStream);

		noflow->stage1(&dpds, defStream);
		rigInt->stage1(&obj,  defStream);
		noflow->stage2(&dpds, defStream);
		rigInt->stage2(&obj,  defStream);

		obj.local()->findExtentAndCOM(defStream);
		bounceFromRigidEllipsoid(&dpds, cells, &obj, dt, true, defStream);

		CUDA_Check( cudaStreamSynchronize(defStream) );

		redist.redistribute(defStream);
	}

	double elapsed = tm.elapsed() * 1e-9;

	printf("Finished in %f s, 1 step took %f ms\n", elapsed, elapsed / niters * 1000.0);

	return 0;
}
