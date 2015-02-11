#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cassert>

#include <mpi.h>

#include <string>
#include <sstream>
#include <vector>

//in common.h i define Particle and Acceleration data structures
//as well as the global parameters for the simulation
#include "common.h"

#include "dpd-interactions.h"
#include "wall-interactions.h"
#include "redistribute-particles.h"
#include "logistic.h"

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

void diagnostics(MPI_Comm comm, Particle * _particles, int n, float dt, int idstep, int L, Acceleration * _acc, bool particledump)
{
    Particle * particles = new Particle[n];
    Acceleration * acc = new Acceleration[n];

    CUDA_CHECK(cudaMemcpy(particles, _particles, sizeof(Particle) * n, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(acc, _acc, sizeof(Acceleration) * n, cudaMemcpyDeviceToHost));
    
    int nlocal = n;

    //we fused VV stages so we need to recover the state before stage 1
    for(int i = 0; i < n; ++i)
	for(int c = 0; c < 3; ++c)
	{
	    assert(!isnan(particles[i].x[c]));
	    assert(!isnan(particles[i].u[c]));
	    assert(!isnan(acc[i].a[c]));
	    
	    particles[i].x[c] -= dt * particles[i].u[c];
	    particles[i].u[c] -= 0.5 * dt * acc[i].a[c];
	}
    
    double p[] = {0, 0, 0};
    for(int i = 0; i < n; ++i)
	for(int c = 0; c < 3; ++c)
	    p[c] += particles[i].u[c];

    int rank;
    MPI_CHECK( MPI_Comm_rank(comm, &rank) );

    int dims[3], periods[3], coords[3];
    MPI_CHECK( MPI_Cart_get(comm, 3, dims, periods, coords) );
    
    MPI_CHECK( MPI_Reduce(rank == 0 ? MPI_IN_PLACE : &p, rank == 0 ? &p : NULL, 3, MPI_DOUBLE, MPI_SUM, 0, comm) );
    
    if (rank == 0)
	printf("momentum: %f %f %f\n", p[0], p[1], p[2]);

    double ke = 0;
    for(int i = 0; i < n; ++i)
	ke += pow(particles[i].u[0], 2) + pow(particles[i].u[1], 2) + pow(particles[i].u[2], 2);

    MPI_CHECK( MPI_Reduce(rank == 0 ? MPI_IN_PLACE : &ke, &ke, 1, MPI_DOUBLE, MPI_SUM, 0, comm) );
    MPI_CHECK( MPI_Reduce(rank == 0 ? MPI_IN_PLACE : &n, &n, 1, MPI_INT, MPI_SUM, 0, comm) );
    
    double kbt = 0.5 * ke / (n * 3. / 2);

    if (rank == 0)
    {
	static bool firsttime = true;
	FILE * f = fopen("diag.txt", firsttime ? "w" : "a");
	firsttime = false;
	
	if (idstep == 0)
	    fprintf(f, "TSTEP\tKBT\tPX\tPY\tPZ\n");
	
	fprintf(f, "%e\t%.10e\t%.10e\t%.10e\t%.10e\n", idstep * dt, kbt, p[0], p[1], p[2]);
	
	fclose(f);
    }

    if (particledump)
    {
	std::stringstream ss;

	if (rank == 0)
	{
	    ss <<  n << "\n";
	    ss << "dpdparticle\n";

	    printf("total number of particles: %d\n", n);
	}
    
	for(int i = 0; i < nlocal; ++i)
	    ss << rank << " " 
	       << (particles[i].x[0] + L / 2 + coords[0] * L) << " "
	       << (particles[i].x[1] + L / 2 + coords[1] * L) << " "
	       << (particles[i].x[2] + L / 2 + coords[2] * L) << "\n";

	string content = ss.str();
	
	int len = content.size();
	int offset = 0;
	MPI_CHECK( MPI_Exscan(&len, &offset, 1, MPI_INTEGER, MPI_SUM, comm)); 
	
	MPI_File f;
	char fn[] = "trajectories.xyz";

	bool filenotthere;
	
	if (rank == 0)
	    filenotthere = access( "trajectories.xyz", F_OK ) == -1;

	MPI_CHECK( MPI_Bcast(&filenotthere, 1, MPI_INT, 0, comm) );
	
	static bool firsttime = true;

	firsttime |= filenotthere;

	MPI_CHECK( MPI_File_open(comm, fn, MPI_MODE_WRONLY | (firsttime ? MPI_MODE_CREATE : MPI_MODE_APPEND), MPI_INFO_NULL, &f) );

	if (firsttime)
	    MPI_CHECK( MPI_File_set_size (f, 0));

	firsttime = false;
	
	MPI_Offset base;
	MPI_CHECK( MPI_File_get_position(f, &base));
	
	MPI_Status status;
	
	MPI_CHECK( MPI_File_write_at_all(f, base + offset, const_cast<char *>(content.c_str()), len, MPI_CHAR, &status));
	
	MPI_CHECK( MPI_File_close(&f));
    }

    delete [] particles;
    delete [] acc;
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
	    ParticleArray particles(ic);
	    CellLists cells(L);		  
	    RedistributeParticles redistribute(cartcomm, L);
	    ComputeInteractionsDPD dpd(cartcomm, L);
	    ComputeInteractionsWall * wall = NULL;
            // in production runs replace the numbers with 4 unique ones that are same across ranks
            KISS rng_trunk( 0x26F94D92, 0x383E7D1E, 0x46144B48, 0x6DDD73CB );
	
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
		    diagnostics(cartcomm, particles.xyzuvw, particles.size, dt, it, L, particles.axayaz, true);
	    }
	
	    if (wall != NULL)
		delete wall;
	}

	MPI_CHECK( MPI_Finalize() );
	
	if (rank == 0)
	    printf("simulation is done\n");
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaDeviceReset());

    return 0;
}
	
