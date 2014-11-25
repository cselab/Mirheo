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

using namespace std;

//velocity verlet stages - first stage
__global__ void update_stage1(Particle * p, Acceleration * a, int n, float dt, float dpdx, float dpdy, float dpdz)
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

    const float gradp[3] = {dpdx, dpdy, dpdz};
    
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
__global__ void update_stage2_and_1(Particle * p, Acceleration * a, int n, float dt, float dpdx, float dpdy, float dpdz)
{
    assert(blockDim.x * gridDim.x >= n);
    
    const int pid = threadIdx.x + blockDim.x * blockIdx.x;

    if (pid >= n)
	return;

    for(int c = 0; c < 3; ++c)
	assert(!isnan(p[pid].u[c]));

    for(int c = 0; c < 3; ++c)
	assert(!isnan(a[pid].a[c]));

    const float gradp[3] = {dpdx, dpdy, dpdz};
    
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


//container for the gpu particles during the simulation
struct ParticleArray
{
    int capacity, size;

    Particle * xyzuvw;
    Acceleration * axayaz;
    
    ParticleArray(vector<Particle> ic):
	capacity(0), size(0), xyzuvw(NULL), axayaz(NULL)
	{
	    resize(ic.size());

	    CUDA_CHECK(cudaMemcpy(xyzuvw, (float*) &ic.front(), sizeof(Particle) * ic.size(), cudaMemcpyHostToDevice));
	    CUDA_CHECK(cudaMemset(axayaz, 0, sizeof(Acceleration) * ic.size()));
	}

    void resize(int n)
	{
	    size = n;
	    
	    if (capacity >= n)
		return;

	    if (xyzuvw != NULL)
	    {
		CUDA_CHECK(cudaFree(xyzuvw));
		CUDA_CHECK(cudaFree(axayaz));
	    }
	    
	    CUDA_CHECK(cudaMalloc(&xyzuvw, sizeof(Particle) * n));
	    CUDA_CHECK(cudaMalloc(&axayaz, sizeof(Acceleration) * n));
	    CUDA_CHECK(cudaMemset(axayaz, 0, sizeof(Acceleration) * n));
	    capacity = n;
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
	
	    //the methods of these classes are not expected to call cudaThreadSynchronize unless really necessary
	    //(be aware of that)
	    H5PartDump dump("trajectories.h5part", cartcomm, L);
	    ParticleArray particles(ic);
	    CellLists cells(L);		  
	    RedistributeParticles redistribute(cartcomm, L);
	    ComputeInteractionsDPD dpd(cartcomm, L);
	    ComputeInteractionsWall * wall = NULL;
	
	    int saru_tag = rank;
	
	    cells.build(particles.xyzuvw, particles.size);
	
	    dpd.evaluate(saru_tag, particles.xyzuvw, particles.size, particles.axayaz, cells.start, cells.count);
	
	    const size_t nsteps = (int)(tend / dt);
	
	    float grad_p[3] = {0, 0, 0};
	
	    for(int it = 0; it < nsteps; ++it)
	    {
		if (rank == 0)
		    printf("beginning of time step %d\n", it);
	    
		if (it == 0)
		    update_stage1<<<(particles.size + 127) / 128, 128 >>>(
			particles.xyzuvw, particles.axayaz, particles.size, dt, grad_p[0], grad_p[1], grad_p[2]);

		int newnp = redistribute.stage1(particles.xyzuvw, particles.size);
		
		particles.resize(newnp);
	    
		redistribute.stage2(particles.xyzuvw, particles.size);

		if (it > 100 && wall == NULL)
		{
		    int nsurvived = 0;
		    wall = new ComputeInteractionsWall(cartcomm, L, particles.xyzuvw, particles.size, nsurvived);
		    
		    particles.resize(nsurvived);
		    
		    grad_p[2] = -0.1;

		    if (rank == 0)
			if( access( "trajectories.xyz", F_OK ) != -1 )
			{
			    const int retval = rename ("trajectories.xyz", "trajectories-equilibration.xyz");
			    assert(retval != -1);
			}
		}

		cells.build(particles.xyzuvw, particles.size);

		dpd.evaluate(saru_tag, particles.xyzuvw, particles.size, particles.axayaz, cells.start, cells.count);

		if (wall != NULL)
		    wall->interactions(particles.xyzuvw, particles.size, particles.axayaz, 
				       cells.start, cells.count, saru_tag);

		if (particles.size > 0)
		    update_stage2_and_1<<<(particles.size + 127) / 128, 128 >>>
			(particles.xyzuvw, particles.axayaz, particles.size, dt, grad_p[0], grad_p[1], grad_p[2]);

		if (wall != NULL)
		    wall->bounce(particles.xyzuvw, particles.size);
	    
		if (it % 10 == 0)
		{
		    const int n = particles.size;

		    Particle * p = new Particle[n];
		    Acceleration * a = new Acceleration[n];

		    CUDA_CHECK(cudaMemcpy(p, particles.xyzuvw, sizeof(Particle) * n, cudaMemcpyDeviceToHost));
		    CUDA_CHECK(cudaMemcpy(a, particles.axayaz, sizeof(Acceleration) * n, cudaMemcpyDeviceToHost));
		   
		    diagnostics(cartcomm, p, n, dt, it, L, a, true);

		    if (it > 100)
			dump.dump(p, n);

		    delete [] p;
		    delete [] a;
		}
	    }
	
	    if (wall != NULL)
		delete wall;
	}

	MPI_CHECK( MPI_Finalize() );
	
	if (rank == 0)
	    printf("simulation is done. Ciao.\n");
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaDeviceReset());

    return 0;
}
	
