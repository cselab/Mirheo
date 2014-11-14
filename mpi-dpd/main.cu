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
#include "redistribute-particles.h"

#include <cuda-dpd.h>

using namespace std;

//velocity verlet stages
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
	p[pid].u[c] += a[pid].a[c] * dt * 0.5;
    
    for(int c = 0; c < 3; ++c)
	p[pid].x[c] += p[pid].u[c] * dt;
}

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
	const float mya = a[pid].a[c];
	float myu = p[pid].u[c];
	float myx = p[pid].x[c];

	myu += mya * dt;
	myx += myu * dt;
	
	p[pid].u[c] = myu; 
	p[pid].x[c] = myx; 
    }
}

void diagnostics(MPI_Comm comm, Particle * _particles, int n, float dt, int idstep, int L, Acceleration * _acc )
{
    Particle * particles = new Particle[n];
    Acceleration * acc = new Acceleration[n];

    CUDA_CHECK(cudaMemcpy(particles, _particles, sizeof(Particle) * n, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(acc, _acc, sizeof(Acceleration) * n, cudaMemcpyDeviceToHost));
    
    int nlocal = n;

    for(int i = 0; i < n; ++i)
	for(int c = 0; c < 3; ++c)
	    particles[i].u[c] -= 0.5 * dt * acc[i].a[c];
    
    double p[] = {0, 0, 0};
    for(int i = 0; i < n; ++i)
	for(int c = 0; c < 3; ++c)
	    p[c] += particles[i].u[c];

    int rank;
    MPI_CHECK( MPI_Comm_rank(comm, &rank) );

    int dims[3], periods[3], coords[3];
    MPI_CHECK( MPI_Cart_get(comm, 3, dims, periods, coords) );
    
    MPI_CHECK( MPI_Reduce(rank == 0 ? MPI_IN_PLACE : &p, &p, 3, MPI_DOUBLE, MPI_SUM, 0, comm) );
    
    double ke = 0;
    for(int i = 0; i < n; ++i)
	ke += pow(particles[i].u[0], 2) + pow(particles[i].u[1], 2) + pow(particles[i].u[2], 2);

    MPI_CHECK( MPI_Reduce(rank == 0 ? MPI_IN_PLACE : &ke, &ke, 1, MPI_DOUBLE, MPI_SUM, 0, comm) );
    MPI_CHECK( MPI_Reduce(rank == 0 ? MPI_IN_PLACE : &n, &n, 1, MPI_INT, MPI_SUM, 0, comm) );
    
    double kbt = 0.5 * ke / (n * 3. / 2);

    //output temperature and total momentum
    if (rank == 0)
    {
	FILE * f = fopen("diag.txt", idstep ? "a" : "w");

	if (idstep == 0)
	    fprintf(f, "TSTEP\tKBT\tPX\tPY\tPZ\n");
	
	fprintf(f, "%e\t%.10e\t%.10e\t%.10e\t%.10e\n", idstep * dt, kbt, p[0], p[1], p[2]);
	
	fclose(f);
    }
    
    //output VMD file
    {
	std::stringstream ss;

	if (rank == 0)
	{
	    ss << n << "\n";
	    ss << "dpdparticle\n";
	}
    
	for(int i = 0; i < nlocal; ++i)
	    ss << "1 "
	       << (particles[i].x[0] + L / 2 + coords[0] * L) << " "
	       << (particles[i].x[1] + L / 2 + coords[1] * L) << " "
	       << (particles[i].x[2] + L / 2 + coords[2] * L) << "\n";

	string content = ss.str();

	int len = content.size();
	int offset;
	MPI_CHECK( MPI_Exscan(&len, &offset, 1, MPI_INTEGER, MPI_SUM, comm)); 
	
	MPI_File f;
	char fn[] = "trajectories.xyz";
	MPI_CHECK( MPI_File_open(comm, fn, MPI_MODE_WRONLY | (idstep == 0 ? MPI_MODE_CREATE : MPI_MODE_APPEND), MPI_INFO_NULL, &f) );

	if (idstep == 0)
	    MPI_CHECK( MPI_File_set_size (f, 0));

	MPI_Offset base;
	MPI_CHECK( MPI_File_get_position(f, &base));
	
	MPI_Status status;
	MPI_CHECK( MPI_File_write_at_all(f, base + offset, const_cast<char *>(content.data()), len, MPI_CHAR, &status));
	
	MPI_CHECK( MPI_File_close(&f));
    }

    delete [] particles;
    delete [] acc;
}

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
	    capacity = n;
	}
};

struct CellLists
{
    const int ncells, L;

    int * start, * count;
    
    CellLists(const int L): ncells(L * L * L), L(L)
	{
	    CUDA_CHECK(cudaMalloc(&start, sizeof(int) * ncells));
	    CUDA_CHECK(cudaMalloc(&count, sizeof(int) * ncells));
	}

    void build(Particle * const p, const int n)
	{
	    build_clists((float * )p, n, 1, L, L, L, -L/2, -L/2, -L/2, NULL, start, count,  NULL);
	}

    ~CellLists()
	{
	    CUDA_CHECK(cudaFree(start));
	    CUDA_CHECK(cudaFree(count));
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
    
    MPI_CHECK( MPI_Init(&argc, &argv) );
    
    int nranks, rank;
    MPI_CHECK( MPI_Comm_size(MPI_COMM_WORLD, &nranks) );
    MPI_CHECK( MPI_Comm_rank(MPI_COMM_WORLD, &rank) );
    
    MPI_Comm cartcomm;
    
    int periods[] = {1,1,1};
    MPI_CHECK( MPI_Cart_create(MPI_COMM_WORLD, 3, ranks, periods, 1, &cartcomm) );

    const int L = 8;
    
    vector<Particle> ic(L * L * L * 3);

    for(int i = 0; i < ic.size(); ++i)
	for(int c = 0; c < 3; ++c)
	    ic[i].x[c] = -L * 0.5 + drand48() * L;

    ParticleArray particles(ic);
    CellLists cells(L);		  
    RedistributeParticles redistribute(cartcomm, L);
    ComputeInteractionsDPD dpd(cartcomm, L);

    int saru_tag = rank;

    cells.build(particles.xyzuvw, particles.size);
    
    dpd.evaluate(saru_tag, particles.xyzuvw, particles.size, particles.axayaz, cells.start, cells.count);

    const size_t nsteps = (int)(tend / dt);
    for(int it = 0; it < nsteps; ++it)
    {
	if (rank == 0)
	    printf("beginning of time step %d\n", it);
	 
	if (it == 0)
	    update_stage1<<<(particles.size + 127) / 128, 128>>>(particles.xyzuvw, particles.axayaz, particles.size, dt);

	int newnp = redistribute.stage1(particles.xyzuvw, particles.size);
	
	particles.resize(newnp);

	redistribute.stage2(particles.xyzuvw, particles.size);

	cells.build(particles.xyzuvw, particles.size);
	
	dpd.evaluate(saru_tag, particles.xyzuvw, particles.size, particles.axayaz, cells.start, cells.count);
	 
	update_stage2_and_1<<<(particles.size + 127) / 128, 128>>>(particles.xyzuvw, particles.axayaz, particles.size, dt);
	 
	if (it % 50 == 0)
	    diagnostics(cartcomm, particles.xyzuvw, particles.size, dt, it, L, particles.axayaz);
    }
    
    MPI_CHECK( MPI_Finalize() );

    if (rank == 0)
	printf("simulation is done\n");
    
    return 0;
}
	
