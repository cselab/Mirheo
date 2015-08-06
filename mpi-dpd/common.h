/*
 *  common.h
 *  Part of CTC/mpi-dpd/
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

enum { 
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
extern bool walls, pushtheflow, doublepoiseuille, rbcs, ctcs, xyz_dumps, hdf5field_dumps, hdf5part_dumps, is_mps_enabled;
extern int steps_per_report, steps_per_dump, wall_creation_stepid, nvtxstart, nvtxstop;



__device__ __forceinline__
void read_AOS6f(const float2 * const data, const int nparticles, float2& s0, float2& s1, float2& s2)
{
    if (nparticles == 0)
	return;
    
    int laneid;
    asm volatile ("mov.u32 %0, %%laneid;" : "=r"(laneid));

    const int nfloat2 = 3 * nparticles;
     
    if (laneid < nfloat2)
	s0 = data[laneid];

    if (laneid + 32 < nfloat2)
	s1 = data[laneid + 32];

    if (laneid + 64 < nfloat2)
	s2 = data[laneid + 64];

    const int srclane0 = (3 * laneid + 0) & 0x1f;
    const int srclane1 = (srclane0 + 1) & 0x1f;
    const int srclane2 = (srclane0 + 2) & 0x1f;

    const int start = laneid % 3;
    
    {
	const float t0 = __shfl(start == 0 ? s0.x : start == 1 ? s1.x : s2.x, srclane0);
	const float t1 = __shfl(start == 0 ? s2.x : start == 1 ? s0.x : s1.x, srclane1);
	const float t2 = __shfl(start == 0 ? s1.x : start == 1 ? s2.x : s0.x, srclane2);

	s0.x = t0;
	s1.x = t1;
	s2.x = t2;
    }
    
    {
	const float t0 = __shfl(start == 0 ? s0.y : start == 1 ? s1.y : s2.y, srclane0);
	const float t1 = __shfl(start == 0 ? s2.y : start == 1 ? s0.y : s1.y, srclane1);
	const float t2 = __shfl(start == 0 ? s1.y : start == 1 ? s2.y : s0.y, srclane2);

	s0.y = t0;
	s1.y = t1;
	s2.y = t2;
    }
}

__device__ __forceinline__
void write_AOS6f(float2 * const data, const int nparticles, float2& s0, float2& s1, float2& s2)
{
    if (nparticles == 0)
	return;
    
    int laneid;
    asm volatile ("mov.u32 %0, %%laneid;" : "=r"(laneid));
    
    const int srclane0 = (32 * ((laneid) % 3) + laneid) / 3;
    const int srclane1 = (32 * ((laneid + 1) % 3) + laneid) / 3;
    const int srclane2 = (32 * ((laneid + 2) % 3) + laneid) / 3;

    const int start = laneid % 3;
    
    {
	const float t0 = __shfl(s0.x, srclane0);
	const float t1 = __shfl(s2.x, srclane1);
	const float t2 = __shfl(s1.x, srclane2);
	
	s0.x = start == 0 ? t0 : start == 1 ? t2 : t1;
	s1.x = start == 0 ? t1 : start == 1 ? t0 : t2;
	s2.x = start == 0 ? t2 : start == 1 ? t1 : t0;
    }
    
    {
	const float t0 = __shfl(s0.y, srclane0);
	const float t1 = __shfl(s2.y, srclane1);
	const float t2 = __shfl(s1.y, srclane2);
	
	s0.y = start == 0 ? t0 : start == 1 ? t2 : t1;
	s1.y = start == 0 ? t1 : start == 1 ? t0 : t2;
	s2.y = start == 0 ? t2 : start == 1 ? t1 : t0;
    }
    
    const int nfloat2 = 3 * nparticles;
    
    if (laneid < nfloat2)
	data[laneid] = s0;
    
    if (laneid + 32 < nfloat2)
	data[laneid + 32] = s1;
    
    if (laneid + 64 < nfloat2)
	data[laneid + 64] = s2;
}

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

    void preserve_resize(const int n)
	{
	    assert(n >= 0);
	    
	    T * old = data;
	    
	    const int oldsize = size;
	    
	    size = n;
	    
	    if (capacity >= n)
		return;
	    
	    capacity = n;

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

#include <utility>

class HookedTexture
{
    std::pair<void *, int> registered;
    
    template<typename T>  void _create(T * data, const int n)
    {
	struct cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeLinear;
	resDesc.res.linear.devPtr = (void *)data;
	resDesc.res.linear.sizeInBytes = n * sizeof(T);
	resDesc.res.linear.desc = cudaCreateChannelDesc<T>();
		
	struct cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0]   = cudaAddressModeWrap;
	texDesc.addressMode[1]   = cudaAddressModeWrap;
	texDesc.filterMode       = cudaFilterModePoint;
	texDesc.readMode         = cudaReadModeElementType;
	texDesc.normalizedCoords = 0;
		
	CUDA_CHECK(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));
    }
	
    void _discard()	{  if (texObj != 0)CUDA_CHECK(cudaDestroyTextureObject(texObj)); }
	    
public:
	
    cudaTextureObject_t texObj;
	
HookedTexture(): texObj(0) { }

    template<typename T>
	cudaTextureObject_t acquire(T * data, const int n)
    {
	std::pair<void *, int> target = std::make_pair(const_cast<T *>(data), n);

	if (target.first != registered.first || target.second > registered.second)
	{
	    _discard();
	    _create(data, n);
	    registered = target;
	}

	return texObj;
    }
	
    ~HookedTexture() { _discard(); }
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
	MPI_Comm local_comm;
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
