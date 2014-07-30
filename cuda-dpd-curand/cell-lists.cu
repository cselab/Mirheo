#include <cstdio>
#include <unistd.h>
#include "cell-lists.h"

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

__global__ void pid2code(int * codes, int * pids, const int np, const float * xyzuvw, const int3 ncells, const float3 domainstart, const float invrc)
{
    const int pid = threadIdx.x + blockDim.x * blockIdx.x;

    if (pid >= np)
	return;

    const float x = (xyzuvw[0 + 6 * pid] - domainstart.x) * invrc;
    const float y = (xyzuvw[1 + 6 * pid] - domainstart.y) * invrc;
    const float z = (xyzuvw[2 + 6 * pid] - domainstart.z) * invrc;
    
    int ix = (int)floor(x);
    int iy = (int)floor(y);
    int iz = (int)floor(z);
    
    if( !(ix >= 0 && ix < ncells.x) ||
	!(iy >= 0 && iy < ncells.y) ||
	!(iz >= 0 && iz < ncells.z))
	printf("pid %d: oops %f %f %f -> %d %d %d\n", pid, x, y, z, ix, iy, iz);

    ix = max(0, min(ncells.x - 1, ix));
    iy = max(0, min(ncells.y - 1, iy));
    iz = max(0, min(ncells.z - 1, iz));
    
    codes[pid] = encode(ix, iy, iz);
    pids[pid] = pid;
};

__global__ void _gather(const float * input, const int * indices, float * output, const int n)
{
    const int tid = threadIdx.x + blockDim.x * blockIdx.x;
    
    if (tid < n)
	output[tid] = input[(tid % 6) + 6 * indices[tid / 6]];
}

__global__ void _generate_cids(int * cids, const int ntotcells, const int offset, const int3 ncells)
{
    const int tid = threadIdx.x + blockDim.x * blockIdx.x;

    if (tid < ntotcells)
    {
	const int xcid = tid % ncells.x;
	const int ycid = (tid / ncells.x) % ncells.y;
	const int zcid = (tid / ncells.x / ncells.y) % ncells.z;

	cids[tid] = encode(xcid, ycid, zcid) + offset;
    }
    else
	if (tid == ntotcells)
	    cids[tid] = 0x7fffffff;
}

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>

using namespace thrust;

template<typename T> T * _ptr(device_vector<T>& v) { return raw_pointer_cast(v.data()); }

void build_clists(float * const xyzuvw, int np, const float rc,
		  const int xcells, const int ycells, const int zcells,
		  const float xstart, const float ystart, const float zstart,
		  int * const order, int * startcell, int * endcell)
{
    device_vector<int> codes(np), pids(np);
    pid2code<<<(np + 127) / 128, 128>>>(_ptr(codes), _ptr(pids), np, xyzuvw, make_int3(xcells, ycells, zcells), make_float3(xstart, ystart, zstart), 1./rc);

    sort_by_key(codes.begin(), codes.end(), pids.begin());
    
    {
	device_vector<float> tmp(np * 6);
	copy(device_ptr<float>(xyzuvw), device_ptr<float>(xyzuvw + 6 * np), tmp.begin());
	
	_gather<<<(6 * np + 127) / 128, 128>>>(_ptr(tmp), _ptr(pids), xyzuvw, 6 * np);
	CUDA_CHECK(cudaPeekAtLastError());
    }
   
    const int ncells = xcells * ycells * zcells;
    device_vector<int> cids(ncells + 1), cidsp1(ncells + 1);
    
    _generate_cids<<< (cids.size() + 127) / 128, 128>>>(_ptr(cids), ncells, 0,  make_int3(xcells, ycells, zcells));
    _generate_cids<<< (cidsp1.size() + 127) / 128, 128>>>(_ptr(cidsp1), ncells, 1, make_int3(xcells, ycells, zcells) );
	
    lower_bound(codes.begin(), codes.end(), cids.begin(), cids.end(), device_ptr<int>(startcell));
    lower_bound(codes.begin(), codes.end(), cidsp1.begin(), cidsp1.end(), device_ptr<int>(endcell));
}

