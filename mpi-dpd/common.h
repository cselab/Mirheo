#pragma once

#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cassert>

#include <string>
using namespace std;

#include <unistd.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(ans) do { cudaAssert((ans), __FILE__, __LINE__); } while(0)
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
	fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
	sleep(5);
	if (abort) exit(code);
    }
}

#include <mpi.h>

#define MPI_CHECK(ans) do { mpiAssert((ans), __FILE__, __LINE__); } while(0)

inline void mpiAssert(int code, const char *file, int line, bool abort=true)
{
    if (code != MPI_SUCCESS) 
    {
	char error_string[2048];
	int length_of_error_string = sizeof(error_string);
	MPI_Error_string(code, error_string, &length_of_error_string);
	 
	printf("mpiAssert: %s %d %s\n", file, line, error_string);
	 	 
	MPI_Abort(MPI_COMM_WORLD, code);
    }
}

#include <cuda-dpd.h>

//AoS is the currency for dpd simulations (because of the spatial locality).
//AoS - SoA conversion might be performed within the hpc kernels.
struct Particle
{
    float x[3], u[3];

    static bool initialized;
    static MPI_Datatype mytype;

    static MPI_Datatype datatype()
	{
	    if (!initialized)
	    {
		MPI_CHECK( MPI_Type_contiguous(6, MPI_FLOAT, &mytype));

		MPI_CHECK(MPI_Type_commit(&mytype));

		initialized = true;
	    }

	    return mytype;
	}
};


struct Acceleration
{
    float a[3];

    static bool initialized;
    static MPI_Datatype mytype;

    static MPI_Datatype datatype()
	{
	    if (!initialized)
	    {
		MPI_CHECK( MPI_Type_contiguous(3, MPI_FLOAT, &mytype));

		MPI_CHECK(MPI_Type_commit(&mytype));

		initialized = true;
	    }

	    return mytype;
	}
};

class H5PartDump
{
    MPI_Comm cartcomm;

    string fname;

    float origin[3];

    void * handler;

    int tstamp;
public:

    H5PartDump(const string filename, MPI_Comm cartcomm, const int L);

    void dump(Particle * host_particles, int n);
    
    ~H5PartDump();
};

void xyz_dump(MPI_Comm comm, const char * filename, const char * particlename, Particle * particles, int n, int L, bool append);
void ply_dump(MPI_Comm comm, const char * filename,
	      int (*mesh_indices)[3], const int ninstances, const int ntriangles_per_instance, Particle * _particles, int nvertices_per_instance, int L, bool append);
void diagnostics(MPI_Comm comm, Particle * _particles, int n, float dt, int idstep, int L, Acceleration * _acc, bool particledump);

const int L = 24;
const float dt = 0.001;
const float tend = 100;
const float kBT = 0.1;
const float gammadpd = 45;
const float sigma = sqrt(2 * gammadpd * kBT);
const float sigmaf = sigma / sqrt(dt);
const float aij = 2.5;
const bool walls = true;

//container for the cell lists, which contains only two integer vectors of size ncells.
//the start[cell-id] array gives the entry in the particle array associated to first particle belonging to cell-id
//count[cell-id] tells how many particles are inside cell-id.
//building the cell lists involve a reordering of the particle array (!)
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
	    if (n > 0)
		build_clists((float * )p, n, 1, L, L, L, -L/2, -L/2, -L/2, NULL, start, count,  NULL, 0);
	}
	    	    
    ~CellLists()
	{
	    CUDA_CHECK(cudaFree(start));
	    CUDA_CHECK(cudaFree(count));
	}
};


//container for the gpu particles during the simulation
template<typename T>
struct SimpleDeviceBuffer
{
    int capacity, size;
    
    T * data;
    
SimpleDeviceBuffer(int n = 0): capacity(0), size(0), data(NULL) { resize(n);}
    
    ~SimpleDeviceBuffer()
	{
	    if (data != NULL)
		CUDA_CHECK(cudaFree(data));
	    
	    data = NULL;
	}
    
    void resize(const int n)
	{
	    assert(n >= 0);
	    
	    size = n;
	    
	    if (capacity >= n)
		return;
	    
	    if (data != NULL)
		CUDA_CHECK(cudaFree(data));
	    
	    capacity = n;
	    
	    CUDA_CHECK(cudaMalloc(&data, sizeof(T) * capacity));
	    
#ifndef NDEBUG
	    CUDA_CHECK(cudaMemset(data, 0, sizeof(T) * capacity));
#endif
	}
    
    void preserve_resize(const int n)
	{
	    assert(n >= 0);
	    
	    T * old = data;
	    
	    const int oldsize = size;
	    
	    size = n;
	    
	    if (capacity >= n)
		return;
	    
	    capacity = n;
	    
	    CUDA_CHECK(cudaMalloc(&data, sizeof(T) * capacity));
	    
	    if (old != NULL)
	    {
		CUDA_CHECK(cudaMemcpy(data, old, sizeof(T) * oldsize, cudaMemcpyDeviceToDevice));
		CUDA_CHECK(cudaFree(old));
	    }
	}
};

template<typename T>
struct PinnedHostBuffer
{
    int capacity, size;
    
    T * data, * devptr;
    
PinnedHostBuffer(int n = 0): capacity(0), size(0), data(NULL), devptr(NULL) { resize(n);}

    ~PinnedHostBuffer()
	{
	    if (data != NULL)
		CUDA_CHECK(cudaFreeHost(data));
	    
	    data = NULL;
	}

    void resize(const int n)
	{
	    assert(n >= 0);

	    size = n;
	    
	    if (capacity >= n)
		return;	    

	    if (data != NULL)
		CUDA_CHECK(cudaFreeHost(data));

	    capacity = n;
	    
	    CUDA_CHECK(cudaHostAlloc(&data, sizeof(T) * capacity, cudaHostAllocMapped));
	    
	    CUDA_CHECK(cudaHostGetDevicePointer(&devptr, data, 0));
	}
};
