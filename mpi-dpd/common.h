#pragma once

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
const float tend = 50;
const float kBT = 0.0945;
const float gammadpd = 45;
const float sigma = sqrt(2 * gammadpd * kBT);
const float sigmaf = sigma / sqrt(dt);
const float aij = 2.5;
const float hydrostatic_a = 0.05;
const bool walls = false;
const bool pushtheflow = false;
const bool rbcs = false;
const bool ctcs = false;
const bool xyz_dumps = false;
const bool hdf5field_dumps = false;
const bool hdf5part_dumps = false;
const int steps_per_report = 1000;
const int steps_per_dump = 100;
const int wall_creation_stepid = 5000;

extern bool currently_profiling;

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
    std::pair< void *, int> registered;
    
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
	std::pair< void *, int> target = std::make_pair(data, n);

	if (target != registered)
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
    
CellLists(const int LX, const int LY, const int LZ): ncells(LX * LY * LZ), LX(LX), LY(LY), LZ(LZ)
	{
	    CUDA_CHECK(cudaMalloc(&start, sizeof(int) * ncells));
	    CUDA_CHECK(cudaMalloc(&count, sizeof(int) * ncells));
	}

    void build(Particle * const p, const int n, cudaStream_t stream);
	    	    
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
