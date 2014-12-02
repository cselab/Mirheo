#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cassert>

#include <mpi.h>

#include <vector>

//in common.h i define Particle and Acceleration data structures
//as well as the global parameters for the simulation
#include "common.h"

#include "dpd-interactions.h"
#include "wall-interactions.h"
#include "redistribute-particles.h"
#include "rbc-interactions.h"

#include <rbc-cuda.h>

#include "redistribute-rbcs.h"

using namespace std;

__constant__ float gradp[3];

//velocity verlet stages - first stage
__global__ void update_stage1(Particle * p, Acceleration * a, int n, float dt)
{
    assert(blockDim.x * gridDim.x >= n);
    
    const int pid = threadIdx.x + blockDim.x * blockIdx.x;

    if (pid >= n)
	return;
    
    for(int c = 0; c < 3; ++c)
    {
	assert(!isnan(p[pid].x[c]));
	assert(!isnan(p[pid].u[c]));
	assert(!isnan(a[pid].a[c]));
    }

       for(int c = 0; c < 3; ++c)
	p[pid].u[c] += (a[pid].a[c] - gradp[c]) * dt * 0.5;
    
       for(int c = 0; c < 3; ++c)
	   p[pid].x[c] += p[pid].u[c] * dt;

    for(int c = 0; c < 3; ++c)
    {
	assert(p[pid].x[c] >= -L -L/2);
	assert(p[pid].x[c] <= +L +L/2);
    }
}

//fused velocity verlet stage 2 and 1 (for the next iteration)
__global__ void update_stage2_and_1(Particle * p, Acceleration * a, int n, float dt)
{
    assert(blockDim.x * gridDim.x >= n);
    
    const int pid = threadIdx.x + blockDim.x * blockIdx.x;

    if (pid >= n)
	return;

    for(int c = 0; c < 3; ++c)
	assert(!isnan(p[pid].u[c]));

    for(int c = 0; c < 3; ++c)
	assert(!isnan(a[pid].a[c]));

    for(int c = 0; c < 3; ++c)
    {
	const float mya = a[pid].a[c] - gradp[c];
	float myu = p[pid].u[c];
	float myx = p[pid].x[c];

	myu += mya * dt;
	myx += myu * dt;
	
	p[pid].u[c] = myu; 
	p[pid].x[c] = myx; 
    }

    for(int c = 0; c < 3; ++c)
    {
	if (!(p[pid].x[c] >= -L -L/2) || !(p[pid].x[c] <= +L +L/2))
	    printf("Uau: %f %f %f %f %f %f and acc %f %f %f\n", 
		   p[pid].x[0], p[pid].x[1], p[pid].x[2], 
		   p[pid].u[0], p[pid].u[1], p[pid].u[2],
		   a[pid].a[0], a[pid].a[1],a[pid].a[2]);

	assert(p[pid].x[c] >= -L -L/2);
	assert(p[pid].x[c] <= +L +L/2);
    }
}

__global__ void fake_vel(Particle * p, const int n)
{
    const int gid = threadIdx.x + blockDim.x * blockIdx.x;

    if (gid < n)
	for(int c = 0; c < 3; ++c)
	    p[gid].u[c] = 3 * (c == 0);
}

//container for the gpu particles during the simulation
struct ParticleArray
{
    int size;

    SimpleDeviceBuffer<Particle> xyzuvw;
    SimpleDeviceBuffer<Acceleration> axayaz;

    ParticleArray() {}
    
    ParticleArray(vector<Particle> ic)
	{
	    resize(ic.size());

	    CUDA_CHECK(cudaMemcpy(xyzuvw.data, (float*) &ic.front(), sizeof(Particle) * ic.size(), cudaMemcpyHostToDevice));
	    CUDA_CHECK(cudaMemset(axayaz.data, 0, sizeof(Acceleration) * ic.size()));
	}

    void resize(int n)
	{
	    size = n;

	    xyzuvw.resize(n);
	    axayaz.resize(n);

	    CUDA_CHECK(cudaMemset(axayaz.data, 0, sizeof(Acceleration) * size));
	}
};

class CollectionRBC : ParticleArray
{
    int nrbcs, nvertices, L;
    
    CudaRBC::Extent extent;

public:
    CollectionRBC(const int L): L(L), nrbcs(0)
	{
	    printf("hellouus\n");
	    
	    CudaRBC::setup(nvertices, extent);
	    
	    printf("extent: %f %f %f %f %f %f\n",
		   extent.xmax , extent.xmin,
		   extent.ymax , extent.ymin,
		   extent.zmax , extent.zmin);
		   
	}

    void createone()
	{
	    nrbcs = 1;
	    
	    assert(extent.xmax - extent.xmin < L);
	    assert(extent.ymax - extent.ymin < L);
	    assert(extent.zmax - extent.zmin < L);
	    
	    resize(nrbcs);

	    float transform[4][4] = { {1, 0, 0, -0.5 * (extent.xmin + extent.xmax) }, 
				      {0, 1, 0, -0.5 * (extent.ymin + extent.ymax)}, 
				      {0, 0, 1, -0.5 * (extent.zmin + extent.zmax)}, 
				      {0, 0, 0, 1} };

	    CUDA_CHECK(cudaMemset(xyzuvw.data, 0, sizeof(Particle) * nvertices));
	    CudaRBC::initialize((float *)xyzuvw.data, transform);
	}

    Particle * data() { return xyzuvw.data; }
    Acceleration * acc() { return axayaz.data; }
    int count() { return nrbcs; }
    int pcount() { return nrbcs * nvertices; }
    
    void resize(const int count)
	{
	    nrbcs = count;

	    ParticleArray::resize(count * nvertices);
	}

    void update(const int it)
	{
	    if (nrbcs == 0)
		return;
	    
	    if (it < 0)
		update_stage1<<<(xyzuvw.size + 127) / 128, 128 >>>(
		    xyzuvw.data, axayaz.data, xyzuvw.size, dt);
	    else
		update_stage2_and_1<<<(xyzuvw.size + 127) / 128, 128 >>>
		    (xyzuvw.data, axayaz.data, xyzuvw.size, dt);

	}

    void dump(MPI_Comm comm)
	{
	    static bool firsttime = true;
	    
	    const int n = size;

	    Particle * p = new Particle[n];
	    Acceleration * a = new Acceleration[n];

	    CUDA_CHECK(cudaMemcpy(p, xyzuvw.data, sizeof(Particle) * n, cudaMemcpyDeviceToHost));
	    CUDA_CHECK(cudaMemcpy(a, axayaz.data, sizeof(Acceleration) * n, cudaMemcpyDeviceToHost));
		   
	    //we fused VV stages so we need to recover the state before stage 1
	    for(int i = 0; i < n; ++i)
		for(int c = 0; c < 3; ++c)
		{
		    assert(!isnan(p[i].x[c]));
		    assert(!isnan(p[i].u[c]));
		    assert(!isnan(a[i].a[c]));
	    
		    p[i].x[c] -= dt * p[i].u[c];
		    p[i].u[c] -= 0.5 * dt * a[i].a[c];
		}

	    xyz_dump(comm, "rbcs.xyz", "rbcparticles", p, n,  L, !firsttime);
		    
	    delete [] p;
	    delete [] a;

	    firsttime = false;
	}
};

int main(int argc, char ** argv)
{
    int ranks[3];
    
    if (argc != 4)
    {
	printf("usage: ./mpi-dpd <xranks> <yranks> <zranks>\n");
	exit(-1);
    }
    else
    {
	for(int i = 0; i < 3; ++i)
	    ranks[i] = atoi(argv[1 + i]);
    }

    CUDA_CHECK(cudaSetDevice(0));

    int nranks, rank;   
 
    {
	MPI_CHECK( MPI_Init(&argc, &argv) );
    
	{

	    MPI_CHECK( MPI_Comm_size(MPI_COMM_WORLD, &nranks) );
	    MPI_CHECK( MPI_Comm_rank(MPI_COMM_WORLD, &rank) );
	
	    MPI_Comm cartcomm;
	
	    int periods[] = {1,1,1};
	    MPI_CHECK( MPI_Cart_create(MPI_COMM_WORLD, 3, ranks, periods, 1, &cartcomm) );
	
	    vector<Particle> ic(L * L * L * 3  );
	    srand48(rank);

	    for(int i = 0; i < ic.size(); ++i)
		for(int c = 0; c < 3; ++c)
		{
		    ic[i].x[c] = -L * 0.5 + drand48() * L;
		    ic[i].u[c] = 0;// (drand48()*2  - 1);
		}

	    float dpdx[3] = {-0.01, 0, 0};
	    
	    CUDA_CHECK(cudaMemcpyToSymbol(gradp, dpdx, sizeof(float) * 3));
	    
	    //the methods of these classes are not expected to call cudaThreadSynchronize unless really necessary
	    //(be aware of that)
	    H5PartDump dump("trajectories.h5part", cartcomm, L);
	    ParticleArray particles(ic);
	    CellLists cells(L);		  
	    RedistributeParticles redistribute(cartcomm, L);
	    ComputeInteractionsDPD dpd(cartcomm, L);
	    ComputeInteractionsWall * wall = NULL;
	    CollectionRBC rbcs(L);
	    
	    //if (rank == 0)
		rbcs.createone();
	    //exit(0);
	    cudaStream_t stream;
	    CUDA_CHECK(cudaStreamCreate(&stream));
	    
	    RedistributeRBCs redistribute_rbcs(cartcomm, L);
	    redistribute_rbcs.stream = stream;
	    int saru_tag = rank;

	    ComputeInteractionsRBC rbc_interactions(cartcomm, L);

	    rbcs.update(-1);
	    
	    cells.build(particles.xyzuvw.data, particles.size);
	   
	    dpd.evaluate(saru_tag, particles.xyzuvw.data, particles.size, particles.axayaz.data, cells.start, cells.count);
	    rbc_interactions.evaluate(saru_tag, particles.xyzuvw.data, particles.size, particles.axayaz.data, cells.start, cells.count,
				      rbcs.data(), rbcs.count(), rbcs.acc());

	    const size_t nsteps = (int)(tend / dt);

	    for(int it = 0; it < nsteps; ++it)
	    {
		if (rank == 0)
		    printf("beginning of time step %d\n", it);
	    
		if (it == 0)
		    update_stage1<<<(particles.size + 127) / 128, 128 >>>(
			particles.xyzuvw.data, particles.axayaz.data, particles.size, dt);

		int newnp = redistribute.stage1(particles.xyzuvw.data, particles.size);
		
		particles.resize(newnp);
	    
		redistribute.stage2(particles.xyzuvw.data, particles.size);

		int nrbcs = redistribute_rbcs.stage1(rbcs.data(), rbcs.count());

		rbcs.resize(nrbcs);

		redistribute_rbcs.stage2(rbcs.data(), rbcs.count());

		if (walls && it > 500 && wall == NULL)
		{
		    int nsurvived = 0;
		    wall = new ComputeInteractionsWall(cartcomm, L, particles.xyzuvw.data, particles.size, nsurvived);
		    
		    particles.resize(nsurvived);
		    
		    dpdx[0] = -0.1;
		    CUDA_CHECK(cudaMemcpyToSymbol(gradp, dpdx, sizeof(float) * 3));

		    if (rank == 0)
			if( access( "trajectories.xyz", F_OK ) != -1 )
			{
			    const int retval = rename ("trajectories.xyz", "trajectories-equilibration.xyz");
			    assert(retval != -1);
			}
		}

		cells.build(particles.xyzuvw.data, particles.size);

		dpd.evaluate(saru_tag, particles.xyzuvw.data, particles.size, particles.axayaz.data, cells.start, cells.count);
		rbc_interactions.evaluate(saru_tag, particles.xyzuvw.data, particles.size, particles.axayaz.data, cells.start, cells.count,
					  rbcs.data(), rbcs.count(), rbcs.acc());

		//I NEED A REGISTRATION MECHANISM FOR DPD-INTERACTION AND RBC-INTERACTION
		if (wall != NULL)
		{
		    wall->interactions(particles.xyzuvw.data, particles.size, particles.axayaz.data, 
				       cells.start, cells.count, saru_tag);
		    
		    wall->interactions(rbcs.data(), rbcs.pcount(), rbcs.acc(), 
				       NULL, NULL, saru_tag);
		}

		if (particles.size > 0)
		    update_stage2_and_1<<<(particles.size + 127) / 128, 128 >>>
			(particles.xyzuvw.data, particles.axayaz.data, particles.size, dt);

		rbcs.update(it);

		if (wall != NULL)
		{
		    wall->bounce(particles.xyzuvw.data, particles.size);
		    wall->bounce(rbcs.data(), rbcs.pcount());
		}
	    
		if (it % 50 == 0)
		{
		    const int n = particles.size;

		    Particle * p = new Particle[n];
		    Acceleration * a = new Acceleration[n];

		    CUDA_CHECK(cudaMemcpy(p, particles.xyzuvw.data, sizeof(Particle) * n, cudaMemcpyDeviceToHost));
		    CUDA_CHECK(cudaMemcpy(a, particles.axayaz.data, sizeof(Acceleration) * n, cudaMemcpyDeviceToHost));
		   
		    //we fused VV stages so we need to recover the state before stage 1
		    for(int i = 0; i < n; ++i)
			for(int c = 0; c < 3; ++c)
			{
			    assert(!isnan(p[i].x[c]));
			    assert(!isnan(p[i].u[c]));
			    assert(!isnan(a[i].a[c]));
	    
			    p[i].x[c] -= dt * p[i].u[c];
			    p[i].u[c] -= 0.5 * dt * a[i].a[c];
			}

		    diagnostics(cartcomm, p, n, dt, it, L, a, true);
		    rbcs.dump(cartcomm);
		    
		    if (it > 100)
			dump.dump(p, n);

		    delete [] p;
		    delete [] a;
		}
	    }

	    CUDA_CHECK(cudaStreamDestroy(stream));
	
	    if (wall != NULL)
		delete wall;

	    MPI_CHECK(MPI_Comm_free(&cartcomm));
	}
	
	MPI_CHECK( MPI_Finalize() );
	
	if (rank == 0)
	    printf("simulation is done. Ciao.\n");
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaDeviceReset());

    return 0;
}
	
