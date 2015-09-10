/*
 *  common.h
 *  Part of uDeviceX/mpi-dpd/
 *
 *  Created and authored by Diego Rossinelli, on 2014-12-05.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */

#pragma once

#ifdef NDEBUG
#define cuda_printf(...)
#else
#define cuda_printf(...) do { printf(__VA_ARGS__); } while(0)
#endif

enum
{
    XSIZE_SUBDOMAIN = 48,
    YSIZE_SUBDOMAIN = 48,
    ZSIZE_SUBDOMAIN = 48,
    XMARGIN_WALL = 6,
    YMARGIN_WALL = 6,
    ZMARGIN_WALL = 6,
};

const int numberdensity = 4;
const float dt = 0.001;
const float kBT = 0.0945;
const float gammadpd = 45;
const float sigma = sqrt(2 * gammadpd * kBT);
const float sigmaf = sigma / sqrt(dt);
const float aij = 25;
const float hydrostatic_a = 0.05;

extern float tend;
extern bool walls, pushtheflow, doublepoiseuille, rbcs, ctcs, xyz_dumps, hdf5field_dumps, hdf5part_dumps, is_mps_enabled, contactforces;
extern int steps_per_report, steps_per_dump, wall_creation_stepid, nvtxstart, nvtxstop;

#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cassert>

#include <unistd.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(ans) do { cudaAssert((ans), __FILE__, __LINE__); } while(0)
inline void cudaAssert(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess)
    {
	fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);

	abort();
    }
}

#ifdef _USE_NVTX_

#include <nvToolsExt.h>
enum NVTX_COLORS
{
    NVTX_C1 = 0x0000ff00,
    NVTX_C2 = 0x000000ff,
    NVTX_C3 = 0x00ffff00,
    NVTX_C4 = 0x00ff00ff,
    NVTX_C5 = 0x0000ffff,
    NVTX_C6 = 0x00ff0000,
    NVTX_C7 = 0x00ffffff
};

class NvtxTracer
{
    bool active;

public:

    static bool currently_profiling;

NvtxTracer(const char* name, NVTX_COLORS color = NVTX_C1): active(false)
    {
	if (currently_profiling)
	{
	    active = true;

	    nvtxEventAttributes_t eventAttrib = {0};
	    eventAttrib.version = NVTX_VERSION;
	    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
	    eventAttrib.colorType = NVTX_COLOR_ARGB;
	    eventAttrib.color = color;
	    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
	    eventAttrib.message.ascii = name;
	    nvtxRangePushEx(&eventAttrib);
	}
    }

    ~NvtxTracer()
    {
	if (active) nvtxRangePop();
    }
};

#define NVTX_RANGE(arg...) NvtxTracer uniq_name_using_macros(arg);
#else
#define NVTX_RANGE(arg...)
#endif

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

struct ParticlesWrap
{
    const Particle * p;
    Acceleration * a;
    int n;

ParticlesWrap() : p(NULL), a(NULL), n(0){}

ParticlesWrap(const Particle * const p, const int n, Acceleration * a):
    p(p), n(n), a(a) {}
};

struct SolventWrap : ParticlesWrap
{
    const int * cellsstart, * cellscount;

SolventWrap(): cellsstart(NULL), cellscount(NULL), ParticlesWrap() {}

SolventWrap(const Particle * const p, const int n, Acceleration * a, const int * const cellsstart, const int * const cellscount):
    ParticlesWrap(p, n, a), cellsstart(cellsstart), cellscount(cellscount) {}
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

    void dispose()
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

	    const int conservative_estimate = (int)ceil(1.1 * n);
	    capacity = 128 * ((conservative_estimate + 129) / 128);

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

	    const int conservative_estimate = (int)ceil(1.1 * n);
	    capacity = 128 * ((conservative_estimate + 129) / 128);

	    CUDA_CHECK(cudaMalloc(&data, sizeof(T) * capacity));

	    if (old != NULL)
	    {
		CUDA_CHECK(cudaMemcpy(data, old, sizeof(T) * oldsize, cudaMemcpyDeviceToDevice));
		CUDA_CHECK(cudaFree(old));
	    }
	}
};

template<typename T>
struct SimpleHostBuffer
{
    int capacity, size;

    T * data;

SimpleHostBuffer(int n = 0): capacity(0), size(0), data(NULL) { resize(n);}

    ~SimpleHostBuffer()
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

	    const int conservative_estimate = (int)ceil(1.1 * n);
	    capacity = 128 * ((conservative_estimate + 129) / 128);

	    CUDA_CHECK(cudaHostAlloc(&data, sizeof(T) * capacity, cudaHostAllocDefault));
	}

    void preserve_resize(const int n)
	{
	    assert(n >= 0);

	    T * old = data;

	    const int oldsize = size;

	    size = n;

	    if (capacity >= n)
		return;

	    const int conservative_estimate = (int)ceil(1.1 * n);
	    capacity = 128 * ((conservative_estimate + 129) / 128);

	    data = NULL;
	    CUDA_CHECK(cudaHostAlloc(&data, sizeof(T) * capacity, cudaHostAllocDefault));

	    if (old != NULL)
	    {
		memcpy(data, old, sizeof(T) * oldsize);
		CUDA_CHECK(cudaFreeHost(old));
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

	    const int conservative_estimate = (int)ceil(1.1 * n);
	    capacity = 128 * ((conservative_estimate + 129) / 128);

	    CUDA_CHECK(cudaHostAlloc(&data, sizeof(T) * capacity, cudaHostAllocMapped));

	    CUDA_CHECK(cudaHostGetDevicePointer(&devptr, data, 0));
	}

    void preserve_resize(const int n)
	{
	    assert(n >= 0);

	    T * old = data;

	    const int oldsize = size;

	    size = n;

	    if (capacity >= n)
		return;

	    const int conservative_estimate = (int)ceil(1.1 * n);
	    capacity = 128 * ((conservative_estimate + 129) / 128);

	    data = NULL;
	    CUDA_CHECK(cudaHostAlloc(&data, sizeof(T) * capacity, cudaHostAllocMapped));

	    if (old != NULL)
	    {
		CUDA_CHECK(cudaMemcpy(data, old, sizeof(T) * oldsize, cudaMemcpyHostToHost));
		CUDA_CHECK(cudaFreeHost(old));
	    }

	    CUDA_CHECK(cudaHostGetDevicePointer(&devptr, data, 0));
	}
};

#include <cuda-dpd.h>

//container for the cell lists, which contains only two integer vectors of size ncells.
//the start[cell-id] array gives the entry in the particle array associated to first particle belonging to cell-id
//count[cell-id] tells how many particles are inside cell-id.
//building the cell lists involve a reordering of the particle array (!)
struct CellLists
{
    const int ncells, LX, LY, LZ;

    int * start, * count;

CellLists(const int LX, const int LY, const int LZ):
    ncells(LX * LY * LZ + 1), LX(LX), LY(LY), LZ(LZ)
	{
	    CUDA_CHECK(cudaMalloc(&start, sizeof(int) * ncells));
	    CUDA_CHECK(cudaMalloc(&count, sizeof(int) * ncells));
	}

    void build(Particle * const p, const int n, cudaStream_t stream, int * const order = NULL, const Particle * const src = NULL);

    ~CellLists()
	{
	    CUDA_CHECK(cudaFree(start));
	    CUDA_CHECK(cudaFree(count));
	}
};

struct ExpectedMessageSizes
{
    int msgsizes[27];
};

void diagnostics(MPI_Comm comm, MPI_Comm cartcomm, Particle * _particles, int n, float dt, int idstep, Acceleration * _acc);

void report_host_memory_usage(MPI_Comm comm, FILE * foutput);


class LocalComm
{
    MPI_Comm local_comm, active_comm;
    int local_rank, local_nranks;
    int rank, nranks;

    char name[MPI_MAX_PROCESSOR_NAME];
    int len;

public:
    LocalComm();

    void initialize(MPI_Comm active_comm);

    void barrier();

    void print_particles(int np);

    int get_size() { return local_nranks; }

    int get_rank() { return local_rank; }

    MPI_Comm get_comm() { return local_comm;  }
};

extern LocalComm localcomm;


inline MPI_Comm setup_reorder_comm(MPI_Comm init_comm, int rank, int nranks)
{
    MPI_Group init_group;
    MPI_Group new_group;
    MPI_Comm new_comm;

    FILE *fp = fopen("MPICH_RANK_ORDER", "r");
    if (fp == NULL)
    {
	if (rank == 0)
	    printf("\nMPICH_RANK_ORDER.txt not found, returning input communicator\n");

	return init_comm;
    }

    // read the ranks in their new order
    int *p_rank = (int*) malloc(nranks*sizeof(int));

    for(int i = 0; i < nranks; i++)
    {
	int rid;
	int f = fscanf(fp, "%d", &rid);
	if (f != 1)
	{
	    printf("ERROR: not enough ranks!\n");
	    MPI_Abort(init_comm, 1);
	}
	p_rank[rid] = i;
    }

#if VERBOSE
    if (rank == 0)
	for (int i = 0; i < nranks; i++)
	    printf("p_rank[%d] = %d\n", i, p_rank[i]);
#endif

    // get the group under init_comm
    MPI_CHECK( MPI_Comm_group(init_comm, &init_group));

    // create the new group
    MPI_CHECK( MPI_Group_incl(init_group, nranks, p_rank, &new_group));

    free(p_rank);

    // create the new communicator
    MPI_CHECK( MPI_Comm_create(init_comm, new_group, &new_comm));

    return new_comm;
}
