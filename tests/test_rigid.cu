// Yo ho ho ho
#define private public

#include <core/particle_vector.h>
#include <core/rigid_object_vector.h>
#include <core/celllist.h>
#include <core/mpi/api.h>
#include <core/logger.h>
#include <core/integrate.h>
#include <core/bounce.h>
#include <core/components.h>

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
	srand48(0);

	MPI_Check( MPI_Comm_size(MPI_COMM_WORLD, &nranks) );
	MPI_Check( MPI_Comm_rank(MPI_COMM_WORLD, &rank) );
	MPI_Check( MPI_Cart_create(MPI_COMM_WORLD, 3, ranks, periods, 0, &cartComm) );

	cudaStream_t defStream;
	CUDA_Check( cudaStreamCreateWithPriority(&defStream, cudaStreamNonBlocking, 10) );

	float3 length{40, 40, 40};
	float3 domainStart = -length / 2.0f;
	const float rc = 1.0f;
	const int ndens = 8;

	float3 axes{3, 4, 5};
	float3 invAxes = 1.0 / axes;
	const float maxAxis = std::max({axes.x, axes.y, axes.z});

	int objSize = 4/3.0 * M_PI * axes.x*axes.y*axes.z * ndens;
	RigidObjectVector obj("obj", objSize, 1);

	// Init object
	memset(&obj.local()->motions[0], 0, sizeof(LocalRigidObjectVector::RigidMotion));
	obj.local()->motions[0].r = make_float3(0);

	obj.local()->motions[0].omega.x = 0*(drand48() - 0.5);
	obj.local()->motions[0].omega.y = 0*(drand48() - 0.5);
	obj.local()->motions[0].omega.z = 0*(drand48() - 0.5);

	obj.local()->motions[0].vel.x = 0*(drand48() - 0.5);
	obj.local()->motions[0].vel.y = 0*(drand48() - 0.5);
	obj.local()->motions[0].vel.z = 0*(drand48() - 0.5);

	obj.local()->motions[0].force  = make_float3(0);
	obj.local()->motions[0].torque = make_float3(0);

	const float phi = M_PI*drand48();
	const float sphi = sin(0.5f*phi);
	const float cphi = cos(0.5f*phi);

	float3 v = make_float3(drand48(), drand48(), drand48());
	v = normalize(v);

	float4 q = obj.local()->motions[0].q = make_float4(cphi, sphi*v.x, sphi*v.y, sphi*v.z);


	for (int i=0; i<obj.objSize; i++)
	{
		Particle p;
		p.u = make_float3(0);
		p.i1 = 0;
		p.s21 = (short)i;

		do
		{
			p.r.x = 2*maxAxis*(drand48() - 0.5);
			p.r.y = 2*maxAxis*(drand48() - 0.5);
			p.r.z = 2*maxAxis*(drand48() - 0.5);

		} while ( ellipsoid(obj.local()->motions[0], invAxes, p.r) > 0 );

		obj.local()->coosvels[i] = p;
	}

	obj.local()->motions. uploadToDevice();
	obj.local()->coosvels.uploadToDevice();
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
	dpds.local()->pushStream(defStream);
	CellList cells(&dpds, rc, length);
	cells.setStream(defStream);
	cells.makePrimary();

	dpds.local()->resize(cells.ncells.x*cells.ncells.y*cells.ncells.z * ndens);

	printf("initializing...\n");

	auto motion = obj.local()->motions[0];
	int c = 0;
	float3 totU = make_float3(0);
	for (int i=0; i<cells.ncells.x; i++)
		for (int j=0; j<cells.ncells.y; j++)
			for (int k=0; k<cells.ncells.z; k++)
				for (int p=0; p<ndens; p++)
				{
					dpds.local()->coosvels[c].r.x = i + drand48() + domainStart.x;
					dpds.local()->coosvels[c].r.y = j + drand48() + domainStart.y;
					dpds.local()->coosvels[c].r.z = k + drand48() + domainStart.z;

					if (ellipsoid(motion, invAxes, dpds.local()->coosvels[c].r) < 0.5)
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
	dpds.local()->resize(c);
	dpds.domainSize = length;
	dpds.mass = 1.0f;
	dpds.local()->coosvels.uploadToDevice();

	ParticleHaloExchanger halo(cartComm, defStream);
	halo.attach(&dpds, &cells);
	ParticleRedistributor redist(cartComm, defStream);
	redist.attach(&dpds, &cells);

	CUDA_Check( cudaStreamSynchronize(defStream) );

	const float dt = 0.002;
	const int niters = 300;

	const float kBT = 1.0;
	const float gammadpd = 20;
	const float sigmadpd = sqrt(2 * gammadpd * kBT);
	const float sigma_dt = sigmadpd / sqrt(dt);
	const float adpd = 50;

	auto inter = [=] (InteractionType type, ParticleVector* pv1, ParticleVector* pv2, CellList* cl, const float t, cudaStream_t stream) {
		interactionDPD(type, pv1, pv2, cl, t, stream, adpd, gammadpd, sigma_dt, 1, rc);
	};

	printf("GPU execution\n");

	Timer tm;
	tm.start();

	HostBuffer<Force> frcs;

	PinnedBuffer<double> energy(1), momentum(3);
	energy.pushStream(defStream);
	momentum.pushStream(defStream);


	for (int i=0; i<niters; i++)
	{
		energy.clear();
		momentum.clear();

		totalMomentumEnergy<<< getNblocks(dpds.local()->size(), 128), 128, 0, defStream >>> (
				(float4*)dpds.local()->coosvels.devPtr(), dpds.mass, dpds.local()->size(), momentum.devPtr(), energy.devPtr());

		totalMomentumEnergy<<< getNblocks(obj.local()->size(), 128), 128, 0, defStream >>> (
				(float4*)obj.local()->coosvels.devPtr(), 1.0f, obj.local()->size(), momentum.devPtr(), energy.devPtr());

//		momentum.downloadFromDevice();
//		energy.downloadFromDevice(true);
//
//		printf("Iteration %d, energy %f, momentum  %f %f %f\n", i, energy[0], momentum[0], momentum[1], momentum[2]);
		cells.build();
		CUDA_Check( cudaStreamSynchronize(defStream) );

		dpds.local()->forces.clear();
		obj.local()->forces.clear();

		halo.init();
		inter(InteractionType::Regular, &dpds, &dpds, &cells, dt*i, defStream);
		inter(InteractionType::Regular, &dpds, &obj,  &cells, dt*i, defStream);
		halo.finalize();

		inter(InteractionType::Halo, &dpds, &dpds, &cells, dt*i, defStream);

//		dpds.local()->coosvels.downloadFromDevice();
//		CUDA_Check( cudaStreamSynchronize(defStream) );
//		frcs.copy(dpds.local()->forces, defStream);
//		for (int j=0; j<dpds.local()->size(); j++)
//			if (dpds.local()->coosvels[j].i1 == 42)
//			{
//				printf("%d :  %f %f %f\n", dpds.local()->coosvels[j].i1, frcs[j].f.x, frcs[j].f.y, frcs[j].f.z);
//				break;
//			}

		integrateNoFlow(&dpds, dt, defStream);
		integrateRigid(&obj, dt, defStream, make_float3(0));

		obj.local()->findExtentAndCOM(defStream);
		bounceFromRigidEllipsoid(&dpds, &cells, &obj, dt, true, defStream);

		CUDA_Check( cudaStreamSynchronize(defStream) );

		redist.redistribute();
	}

	double elapsed = tm.elapsed() * 1e-9;

	printf("Finished in %f s, 1 step took %f ms\n", elapsed, elapsed / niters * 1000.0);

	return 0;
}
