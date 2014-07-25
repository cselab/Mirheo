#include <cstdio>
#include <unistd.h>
#include "cell-lists.h"

#if 1
// The following encoding/decoding was taken from
// http://fgiesen.wordpress.com/2009/12/13/decoding-morton-codes/
// "Insert" two 0 bits after each of the 10 low bits of x
__device__ inline uint Part1By2(uint x)
{
    x &= 0x000003ff;                  // x = ---- ---- ---- ---- ---- --98 7654 3210
    x = (x ^ (x << 16)) & 0xff0000ff; // x = ---- --98 ---- ---- ---- ---- 7654 3210
    x = (x ^ (x <<  8)) & 0x0300f00f; // x = ---- --98 ---- ---- 7654 ---- ---- 3210
    x = (x ^ (x <<  4)) & 0x030c30c3; // x = ---- --98 ---- 76-- --54 ---- 32-- --10
    x = (x ^ (x <<  2)) & 0x09249249; // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
    return x;
} 

// Inverse of Part1By2 - "delete" all bits not at positions divisible by 3
__device__ uint inline Compact1By2(uint x)
{
    x &= 0x09249249;                  // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
    x = (x ^ (x >>  2)) & 0x030c30c3; // x = ---- --98 ---- 76-- --54 ---- 32-- --10
    x = (x ^ (x >>  4)) & 0x0300f00f; // x = ---- --98 ---- ---- 7654 ---- ---- 3210
    x = (x ^ (x >>  8)) & 0xff0000ff; // x = ---- --98 ---- ---- ---- ---- 7654 3210
    x = (x ^ (x >> 16)) & 0x000003ff; // x = ---- ---- ---- ---- ---- --98 7654 3210
    return x;
}

__device__ int inline encode(int x, int y, int z) 
{
    return (Part1By2(z) << 2) + (Part1By2(y) << 1) + Part1By2(x);
}

__device__ int3 inline decode(int code)
{
    return make_int3(
	Compact1By2(code >> 0),
	Compact1By2(code >> 1),
	Compact1By2(code >> 2)
	);
}
#else
__device__ int encode(int ix, int iy, int iz) 
{
    const int retval = ix + info.ncells.x * (iy + iz * info.ncells.y);

    assert(retval < info.ncells.x * info.ncells.y * info.ncells.z && retval>=0);

    return retval; 
}
	
__device__ int3 decode(int code)
{
    const int ix = code % info.ncells.x;
    const int iy = (code / info.ncells.x) % info.ncells.y;
    const int iz = (code / info.ncells.x/info.ncells.y);

    return make_int3(ix, iy, iz);
}
#endif

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
}

__global__
void _count_particles(const int * const cellsstart, int * const cellscount, const int ncells)
{
    const int tid = threadIdx.x + blockDim.x * blockIdx.x;

    if (tid < ncells)
	cellscount[tid] -= cellsstart[tid];
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
		  int * const order, int * cellsstart, int * cellscount)
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
    device_vector<int> cids(ncells), cidsp1(ncells);
    
    _generate_cids<<< (cids.size() + 127) / 128, 128>>>(_ptr(cids), ncells, 0,  make_int3(xcells, ycells, zcells));
    _generate_cids<<< (cidsp1.size() + 127) / 128, 128>>>(_ptr(cidsp1), ncells, 1, make_int3(xcells, ycells, zcells) );
	
    lower_bound(codes.begin(), codes.end(), cids.begin(), cids.end(), device_ptr<int>(cellsstart));
    lower_bound(codes.begin(), codes.end(), cidsp1.begin(), cidsp1.end(), device_ptr<int>(cellscount));

    _count_particles<<<(ncells + 127) / 128, 128>>> (cellsstart, cellscount, ncells);
}

