// Yo ho ho ho
#define private public
#define protected public

#include <core/utils/cuda_common.h>
#include <core/containers.h>
#include <core/pvs/particle_vector.h>
#include <core/pvs/rbc_vector.h>
#include <core/celllist.h>
#include <core/mpi/api.h>
#include <core/logger.h>
#include <core/integrate.h>
#include <core/interactions.h>
#include <core/bounce.h>

#include "timer.h"
#include <unistd.h>

#include <fstream>
#include <string>

Logger logger;


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

const float3 center{0, 0, 0};

__global__ void checkIn(const Particle* ps, float3 center, int n)
{
	const int pid = blockIdx.x * blockDim.x + threadIdx.x;
	if (pid >= n) return;

	Particle p = ps[pid];

	float v = sqrtf(dot(p.r-center, p.r-center));
	if ( v < 4.9 )
	//if (p.i1 == 28218)
		printf("bad particle %d  value  %f pos %f %f %f\n", p.i1, v, p.r.x, p.r.y, p.r.z);
}


void readIC(std::string fname, ObjectMesh& mesh, PinnedBuffer<Particle>* coosvels)
{
	std::ifstream fin(fname);

	if (!fin.good())
		die("file not found %s", fname.c_str());

	std::string dummy;
	int dummy2;
	fin >> dummy;

	fin >> mesh.nvertices >> mesh.ntriangles >> dummy2;
	coosvels->resize(mesh.nvertices, 0);
	mesh.adjacent.       resize(mesh.nvertices * (mesh.maxDegree+1), 0);
	mesh.adjacent_second.resize(mesh.nvertices * (mesh.maxDegree+1), 0);
	mesh.triangles.resize(mesh.ntriangles, 0);

	for (int i=0; i<mesh.nvertices; i++)
	{
		Particle p;
		p.i1 = i;
		p.u = make_float3(0);

		fin >> p.r.x >> p.r.y >> p.r.z;
		p.r += center;
		(*coosvels)[i] = p;
	}

	for (int i=0; i<mesh.ntriangles; i++)
	{
		fin >> dummy2 >> mesh.triangles[i].x >> mesh.triangles[i].y >> mesh.triangles[i].z;
		if (i == 185 || i == 184)
			printf("triangle %d: %f %f %f,  %f %f %f,  %f %f %f\n", i,
					(*coosvels)[mesh.triangles[i].x].r.x,
					(*coosvels)[mesh.triangles[i].x].r.y,
					(*coosvels)[mesh.triangles[i].x].r.z,

					(*coosvels)[mesh.triangles[i].y].r.x,
					(*coosvels)[mesh.triangles[i].y].r.y,
					(*coosvels)[mesh.triangles[i].y].r.z,

					(*coosvels)[mesh.triangles[i].z].r.x,
					(*coosvels)[mesh.triangles[i].z].r.y,
					(*coosvels)[mesh.triangles[i].z].r.z);
	}

	mesh.triangles.uploadToDevice(0);
	coosvels->uploadToDevice(0);
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

	cudaStream_t defStream = 0;

	float3 length{20, 20, 20};
	float3 domainStart = -length / 2.0f;
	const float rc = 1.0f;
	const int ndens = 8;

	ObjectMesh mesh;
	PinnedBuffer<Particle> meshPV;
	readIC("../walls/sphere.off", mesh, &meshPV);

	ParticleVector dpds("dpd");
	CellList *cells = new PrimaryCellList(&dpds, rc, length);

	dpds.local()->resize(cells->ncells.x*cells->ncells.y*cells->ncells.z * ndens, defStream);

	printf("initializing...\n");

	int c = 0;
	float3 totU = make_float3(0);
	for (int i=0; i<cells->ncells.x; i++)
		for (int j=0; j<cells->ncells.y; j++)
			for (int k=0; k<cells->ncells.z; k++)
				for (int p=0; p<ndens; p++)
				{
					float3 r;

					r.x = i + drand48() + domainStart.x;
					r.y = j + drand48() + domainStart.y;
					r.z = k + drand48() + domainStart.z;

					if ( sqrtf(dot(r-center, r-center)) < 5.1f )
						continue;

					dpds.local()->coosvels[c].r = r;
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
	dpds.localDomainSize = length;
	dpds.mass = 1.0f;
	dpds.local()->coosvels.uploadToDevice(defStream);

	ObjectVector ov("mesh", mesh.nvertices, 1);

	ov.mesh.ntriangles = mesh.ntriangles;
	ov.mesh.nvertices  = mesh.nvertices;
	ov.mesh.triangles.copy(mesh.triangles, 0);

	ov.local()->coosvels.copy(meshPV, 0);

	ParticleHaloExchanger halo(cartComm);
	halo.attach(&dpds, cells);
	ParticleRedistributor redist(cartComm);
	redist.attach(&dpds, cells);

	CUDA_Check( cudaStreamSynchronize(defStream) );

	const int niters = 50;
	const float dt = 0.01;

	std::string xml = R"(<interaction name="dpd" kbt="1.0" gamma="20" a="50" dt="0.01"/>
    <integrate dt="0.01"/>)";
	pugi::xml_document config;
	config.load_string(xml.c_str());

	Interaction *inter = new InteractionDPD(config.child("interaction"));
	Integrator  *noflow = new IntegratorVVNoFlow(config.child("integrate"));
	Integrator  *rigInt = new IntegratorVVRigid(config.child("integrate"));

	Bounce* mbounce = new BounceFromMesh(dt);
	mbounce->setup(&ov, &dpds, cells);

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
				printf("??? %4d :  [%f %f %f] [%f %f %f]\n",
						ptr[j].s21, ptr[j].r.x, ptr[j].r.y, ptr[j].r.z, ptr[j].u.x, ptr[j].u.y, ptr[j].u.z);
		}
	};

	const int nparticles = dpds.local()->size() + mesh.nvertices;
	for (int i=0; i<niters; i++)
	{
//		energy.clear(defStream);
//		momentum.clear(defStream);
//
//		SAFE_KERNEL_LAUNCH(
//				totalMomentumEnergy,
//				getNblocks(dpds.local()->size(), 128), 128, 0, defStream,
				//				(float4*)dpds.local()->coosvels.devPtr(), dpds.mass, dpds.local()->size(), momentum.devPtr(), energy.devPtr() );
//
//		SAFE_KERNEL_LAUNCH(
//				totalMomentumEnergy,
//				getNblocks(obj.local()->size(), 128), 128, 0, defStream,
				//				(float4*)obj.local()->coosvels.devPtr(), 1.0f, obj.local()->size(), momentum.devPtr(), energy.devPtr() );
//
//		momentum.downloadFromDevice(defStream, false);
//		energy.  downloadFromDevice(defStream, true);
//
//		if (i % 100 == 0)
//		{
//			printf("Iteration %d, temp %f, momentum  %.2e %.2e %.2e\n",
//					i, energy[0]/ ( (3/2.0)*nparticles )
//					momentum[0] / nparticles, momentum[1] / nparticles, momentum[2] / nparticles);
//		}

		cells->build(defStream);
		dpds.local()->forces.clear(defStream);
		cells->forces->clear(defStream);

		halo.init(defStream);
		inter->regular(&dpds, &dpds, cells, cells,    dt*i, defStream);
		halo.finalize();

//		CUDA_Check( cudaStreamSynchronize(defStream) );

		inter->halo(&dpds, &dpds, cells, cells, dt*i, defStream);

		noflow->stage1(&dpds, defStream);
		noflow->stage2(&dpds, defStream);

		ov.local()->forces.clear(0);
		mbounce->bounceLocal(defStream);

		int nthreads = 128;
		SAFE_KERNEL_LAUNCH(
				checkIn,
				getNblocks(dpds.local()->size(), nthreads), nthreads,
				dpds.local()->coosvels.devPtr(), center, dpds.local()->size() );


		CUDA_Check( cudaStreamSynchronize(defStream) );

		redist.init(defStream);
		redist.finalize();

		//if (i%100==0)
			printf("It %d\n\n", i);
	}

	double elapsed = tm.elapsed() * 1e-9;

	printf("Finished in %f s, 1 step took %f ms\n", elapsed, elapsed / niters * 1000.0);

	return 0;
}
