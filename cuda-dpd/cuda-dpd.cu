#include <cstdio>
#include <cassert>

//#define _CHECK_

const int collapsefactor = 1;

struct InfoDPD
{
    int3 ncells;
    int np, nsamples, rsamples_start;
    float3 domainsize, domainstart;
    float invrc, aij, gamma, sigmaf;
    float *xyzuvw, *axayaz, *rsamples;
    int * starts;
};

__constant__ InfoDPD info;
 
#if 0
const int depth = 4;
__device__ int encode(int ix, int iy, int iz) 
{
    int idx = 0;
        
    for(int counter = 0; counter < depth; ++counter)
    {
	const int bitmask = 1 << counter;
	const int idx0 = ix&bitmask;
	const int idx1 = iy&bitmask;
	const int idx2 = iz&bitmask;
            
	idx |= ((idx0<<2*counter) | (idx1<<(2*counter+1)) | (idx2<<(2*counter+2)));
    }
        
    return idx; 
}
	
__device__ int3 decode(int code)
{
    int ix = 0, iy = 0, iz = 0;
        
    for(int counter = 0; counter < depth; ++counter)
    {
	const int bitmask_x = 1 << (counter*3+0);
	const int bitmask_y = 1 << (counter*3+1);
	const int bitmask_z = 1 << (counter*3+2);
	
	ix |= (code&bitmask_x)>>2*counter;
	iy |= (code&bitmask_y)>>(2*counter+1);
	iz |= (code&bitmask_z)>>(2*counter+2);
	    
    }
    return make_int3(ix, iy, iz);
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

__global__ void pid2code(int * codes, int * pids)
{
    const int pid = threadIdx.x + blockDim.x * blockIdx.x;

    if (pid >= info.np)
	return;

    const float x = (info.xyzuvw[0 + 6 * pid] - info.domainstart.x) * info.invrc / collapsefactor;
    const float y = (info.xyzuvw[1 + 6 * pid] - info.domainstart.y) * info.invrc / collapsefactor;
    const float z = (info.xyzuvw[2 + 6 * pid] - info.domainstart.z) * info.invrc / collapsefactor;
    
    int ix = (int)floor(x);
    int iy = (int)floor(y);
    int iz = (int)floor(z);
    
    if( !(ix >= 0 && ix < info.ncells.x) ||
	!(iy >= 0 && iy < info.ncells.y) ||
	!(iz >= 0 && iz < info.ncells.z))
	printf("pid %d: oops %f %f %f -> %d %d %d\n", pid, x, y, z, ix, iy, iz);
#if 0 
    assert(ix >= 0 && ix < info.ncells.x);
    assert(iy >= 0 && iy < info.ncells.y);
    assert(iz >= 0 && iz < info.ncells.z);
#else
    ix = max(0, min(info.ncells.x - 1, ix));
    iy = max(0, min(info.ncells.y - 1, iy));
    iz = max(0, min(info.ncells.z - 1, iz));
#endif
    
    codes[pid] = encode(ix, iy, iz);
    pids[pid] = pid;
};

const int xbs = 16;
const int xts = xbs;
const int xbs_l = 3;
const int ybs = 6;
const int yts = 6;
const int ybs_l = 2;

template <bool vertical>
__device__ float3 _reduce(float3 val)
{
    assert(blockDim.x == xbs);
    assert(blockDim.y == ybs);

    __shared__ float buf[3][ybs][xbs];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    buf[0][ty][tx] = val.x;
    buf[1][ty][tx] = val.y;
    buf[2][ty][tx] = val.z;

    __syncthreads();

    if (vertical)
	for(int l = ybs_l; l >= 0; --l)
	{
	    const int L = 1 << l;
	    
	    if (ty < L && ty + L < ybs)
		for(int c = 0; c < 3; ++c)
		    buf[c][ty][tx] += buf[c][ty + L][tx];

	    __syncthreads();
	}
    else
	for(int l = xbs_l; l >= 0; --l)
	{
	    const int L = 1 << l;
	    
	    if (tx < L && tx + L < xbs)
		for(int c = 0; c < 3; ++c)
		    buf[c][ty][tx] += buf[c][ty][tx + L];
	    
	    __syncthreads();
	}	
    
    return make_float3(buf[0][ty][tx], buf[1][ty][tx], buf[2][ty][tx]);
}

__device__ void _ftable(
    float p1[3][yts], float p2[3][xts], float v1[3][yts], float v2[3][xts],
    const int np1, const int np2, const int nonzero_start, const int rsamples_start,
    float a1[3][yts], float a2[3][xts])
{
    assert(np2 <= xts);
    assert(np1 <= yts);
    assert(np1 <= xbs * ybs);
    assert(blockDim.x == xbs && xbs == xts);
    assert(blockDim.y == ybs);

    __shared__ float forces[3][yts][xts];

    const int lx = threadIdx.x;

    if (lx < np2)
	for(int ly = threadIdx.y; ly < np1; ly += blockDim.y)
	{
	    assert(lx < np2 && ly < np1);
	
	    forces[0][ly][lx] = forces[1][ly][lx] = forces[2][ly][lx] = 0;
	
	    if (lx > ly + nonzero_start)
	    {
		float xr = p1[0][ly] - p2[0][lx];
		float yr = p1[1][ly] - p2[1][lx];
		float zr = p1[2][ly] - p2[2][lx];
				
		xr -= info.domainsize.x * floorf(0.5f + xr / info.domainsize.x);
		yr -= info.domainsize.y * floorf(0.5f + yr / info.domainsize.y);
		zr -= info.domainsize.z * floorf(0.5f + zr / info.domainsize.z);

		const float rij2 = xr * xr + yr * yr + zr * zr;
		const float invrij = rsqrtf(rij2);
		const float rij = rij2 * invrij;
		const float wr = max((float)0, 1 - rij * info.invrc);
	
		xr *= invrij;
		yr *= invrij;
		zr *= invrij;

		const float rdotv = xr * (v1[0][ly] - v2[0][lx]) + yr * (v1[1][ly] - v2[1][lx]) + zr * (v1[2][ly] - v2[2][lx]);

		int entry = lx + np2 * ly;
		const float myrandnr = info.rsamples[(info.rsamples_start + rsamples_start + entry) % info.nsamples];
#if 0
		assert(myrandnr != -313);
		info.rsamples[(info.rsamples_start + rsamples_start + entry) % info.nsamples] = -313;
#endif

		const float strength = (info.aij - info.gamma * wr * rdotv + info.sigmaf * myrandnr) * wr;
#ifdef _CHECK_
		forces[0][ly][lx] = rij2 < 1;
		forces[1][ly][lx] = 0;
		forces[2][ly][lx] = 0;
#else
		forces[0][ly][lx] = strength * xr;
		forces[1][ly][lx] = strength * yr;
		forces[2][ly][lx] = strength * zr;
#endif
	    }
	}

    __syncthreads();

    {
	float3 v = make_float3(0, 0, 0);

	if (lx < np2)
	    for(int iy = threadIdx.y; iy < np1; iy += blockDim.y)
	    {
		v.x += forces[0][iy][lx];
		v.y += forces[1][iy][lx];
		v.z += forces[2][iy][lx];
	    }

	v = _reduce<true>(v);
	
	if (lx < np2 && threadIdx.y == 0)
	{
	    a2[0][lx] = v.x;
	    a2[1][lx] = v.y;
	    a2[2][lx] = v.z;
	}
    }

    {
	for(int ly = threadIdx.y, base = 0; base < np1; base += blockDim.y, ly += blockDim.y)
	{
	    float3 h = make_float3(0, 0, 0);       
	
	    if (lx < np2 && ly < np1)
		h = make_float3(forces[0][ly][lx],
				forces[1][ly][lx],
				forces[2][ly][lx]);

	    h = _reduce<false>(h);

	    if (lx == 0 && ly < np1)
	    {
		a1[0][ly] += h.x;
		a1[1][ly] += h.y;
		a1[2][ly] += h.z;
	    }
	}
    }
}

texture<float, cudaTextureType1D> texParticles;

__device__ void _cellscells(const int p1start[4], const int p1count[4], const int p2start[4][4], const int p2scan[4][4],
			    const int maxnp1, const int maxnp2, const bool self, int rsamples_start,
			    float * const axayaz)
{
    __shared__ float pva1[9][yts], pva2[9][xts];

    const int l = threadIdx.x + blockDim.x * threadIdx.y;
    const int BS = blockDim.x * blockDim.y;
      
    for(int ty = 0; ty < maxnp1; ty += yts)
    {
	int np1[4];
	
#pragma unroll
	for(int i = 0; i < 4; ++i)
	    np1[i] = min(yts, p1count[i] - ty);
	
#pragma unroll
	for(int i = 0; i < 4; ++i)
	    assert(BS >= np1[i] * 6);

	float pva1contrib[4];
#pragma unroll
	for(int i = 0; i < 4; ++i)
	    if (l < np1[i] * 6)
		pva1contrib[i] = tex1Dfetch(texParticles, 6 * (p1start[i] + ty) + l);

	float pva1result[4];
#pragma unroll
	for(int i = 0; i < 4; ++i)
	    pva1result[i] = 0;
	
	for(int tx = 0; tx < maxnp2; tx += xts)
	{
	    int np2[4];
#pragma unroll
	    for(int i = 0; i < 4; ++i)
		np2[i] = min(xts, p2scan[i][3] - tx);

	    float pva2contrib[4];
#pragma unroll
	    for(int i = 0; i < 4; ++i)
	    	if (l < np2[i] * 6)
		{
		    const int d = l / 6;
		    const int s = tx + d;
		    const int c = l % 6;
		    const int entry = (s >= p2scan[i][0]) + (s >= p2scan[i][1]) + (s >= p2scan[i][2]);
		    const int pid = s - (entry ? p2scan[i][entry - 1] : 0) + p2start[i][entry];

		    pva2contrib[i] = tex1Dfetch(texParticles, c + 6 * pid);
		}

	    float pva2result[4];
	   
#pragma unroll
	    for(int i = 0; i < 4; ++i)
	    {
		if (l < np1[i] * 6)
		    pva1[l % 6][l / 6] = pva1contrib[i];

		if (l < np1[i] * 3)
		    pva1[6 + (l % 3)][l / 3] = pva1result[i];
	    
		assert(np2[i] * 6 <= BS);
		assert(BS >= np2[i] * 3);
	   
		if (l < np2[i] * 6)
		    pva2[l % 6][l / 6] = pva2contrib[i];

		__syncthreads();

		_ftable(pva1, pva2, &pva1[3], &pva2[3], np1[i], np2[i], i == 0 ? ty - tx : -30000, rsamples_start, &pva1[6], &pva2[6]);
	
		rsamples_start += np1[i] * np2[i];

		if (l < np1[i] * 3)
		     pva1result[i] = pva1[6 + (l % 3)][l / 3];

		if (l < np2[i] * 3)
		     pva2result[i] = pva2[6 + (l % 3)][l / 3];
		
		__syncthreads();
	    }
	    
#pragma unroll
	    for(int i = 0; i < 4; ++i)
	    {
		if (l < np2[i] * 3)
		{
		    const int s = l / 3;
		    const int d = tx + s;
		    const int c = l % 3;
		    const int entry = (d >= p2scan[i][0]) + (d >= p2scan[i][1]) + (d >= p2scan[i][2]);
		    const int pid = d - (entry ? p2scan[i][entry - 1] : 0) + p2start[i][entry];
#ifdef _CHECK_
		    atomicAdd(axayaz + c + 3 * pid, pva2result[i]); //axayaz[c + 3 * pid] += pva2result[i];
#else
		    atomicAdd(axayaz + c + 3 * pid, -pva2result[i]); //axayaz[c + 3 * pid] -= pva2result[i];
#endif
		}
		//	__syncthreads();
	    }
	}

#pragma unroll
	for(int i = 0; i < 4; ++i)
	    assert(np1[i] * 3 <= BS);

	float oldval[4];
#pragma unroll
	for(int i = 0; i < 4; ++i)
	    if (l < np1[i] * 3)
		oldval[i] = axayaz[l + 3 * (p1start[i] + ty)];

#pragma unroll
	for(int i = 0; i < 4; ++i)
	    if (l < np1[i] * 3)
		axayaz[l + 3 * (p1start[i] + ty)] = pva1result[i] + oldval[i];
    }
}

__device__ int _cid(int shiftcode)
{
    int3 indx = decode(blockIdx.x + info.ncells.x * (blockIdx.y + info.ncells.y * blockIdx.z));
	    
    indx.x += (shiftcode & 1);
    indx.y += ((shiftcode >> 1) & 1);
    indx.z += ((shiftcode >> 2) & 1);
	    
    indx.x = (indx.x + info.ncells.x) % info.ncells.x;
    indx.y = (indx.y + info.ncells.y) % info.ncells.y;
    indx.z = (indx.z + info.ncells.z) % info.ncells.z;

    return encode(indx.x, indx.y, indx.z);
}

__constant__ int edgeslutcount[4] = {4, 4, 3, 3};
__constant__ int edgeslutstart[4] = {0, 4, 8, 11};
__constant__ int edgeslut[14] = {0, 1, 2, 7, 2, 4, 6, 7, 4, 5, 7, 4, 0, 7};

texture<int, cudaTextureType1D> texStart;

__global__ void _dpd_forces(float * tmp, int * consumed)
{
    const int idbuf = (blockIdx.x & 1) | ((blockIdx.y & 1) << 1) | ((blockIdx.z & 1) << 2);
    float * const axayaz = tmp + 3 * info.np * idbuf;
    
    const bool master = threadIdx.x + threadIdx.y == 0;
    const int l = threadIdx.x + blockDim.x * threadIdx.y;

    __shared__ int offsetrsamples, rconsumption, maxnp1, maxnp2;
    __shared__ int p1starts[4], p1counts[4];
    __shared__ int p2starts[4][4], p2scans[4][4];

    if (master)
	rconsumption = 0;

    if (l < 4 * 4)
    {
	const int i = l / 4;
	const int j = l % 4;

	if (j == 0)
	{
	    const int cid1 = _cid(i);
	    p1starts[i] = tex1Dfetch(texStart, cid1);
	    p1counts[i] = tex1Dfetch(texStart, cid1 + 1);
	}
		
	if (j < edgeslutcount[i])
	{
	    const int cid2 = _cid(edgeslut[j + edgeslutstart[i]]);
	    
	    p2starts[i][j] = tex1Dfetch(texStart, cid2);
	    p2scans[i][j] = tex1Dfetch(texStart, cid2 + 1);
	}
	else
	    p2scans[i][j] = p2starts[i][j] = 0;

	if (j == 0)
	    p1counts[i] -= p1starts[i];
	
	int myp1count = __shfl(p1counts[i], i * 4 + 0);
	myp1count = max(myp1count, __shfl_xor(myp1count, 8));
	myp1count = max(myp1count, __shfl_xor(myp1count, 4));
	
	if (master)
	    maxnp1 = myp1count;
			
	int entryscan = p2scans[i][j] - p2starts[i][j];
	
	entryscan += (j >= 1) * __shfl_up(entryscan, 1);
	entryscan += (j >= 2) * __shfl_up(entryscan, 2);
	p2scans[i][j] = entryscan;
	
	const int r0 = entryscan * p1counts[i];
	const int e1m = __shfl_xor(entryscan, 4);
	const int e1r = __shfl_xor(r0, 4);
	const int m1 = max(entryscan, e1m);
	const int r1 = r0 + e1r;
	const int e2m = __shfl_xor(m1, 8);
	const int e2r = __shfl_xor(r1, 8);
	const int m2 = max(m1, e2m);
	const int r2 = r1 + e2r;
	
	if (l == 3)
	{
	    maxnp2 = m2;
	    rconsumption = r2;
	    offsetrsamples = atomicAdd(consumed, rconsumption);
	}
    }

    __syncthreads();    
    
    if (offsetrsamples + rconsumption >= info.nsamples)
	return;
    
    _cellscells(p1starts, p1counts, p2starts, p2scans, maxnp1, maxnp2, true, offsetrsamples, axayaz);
    /*  _cellcells(p1starts[0], p1counts[0], p2starts[0], p2scans[0], maxnp1, maxnp2, true, offsetrsamples, axayaz);
	_cellcells(p1starts[1], p1counts[1], p2starts[1], p2scans[1], maxnp1, maxnp2, false, offsetrsamples, axayaz);
	_cellcells(p1starts[2], p1counts[2], p2starts[2], p2scans[2], maxnp1, maxnp2, false , offsetrsamples, axayaz);
	_cellcells(p1starts[3], p1counts[3], p2starts[3], p2scans[3], maxnp1, maxnp2, false , offsetrsamples, axayaz);*/
}

__global__ void _reduce(float * tmp)
{
    assert(gridDim.x * blockDim.x >= info.np);
    
    const int tid = threadIdx.x + blockDim.x * blockIdx.x;

    if (tid < info.np * 3)
    {
	const int nbufs = 8;

	float s = 0;
	for(int idbuf = 0; idbuf < nbufs ; ++idbuf)
	    s += tmp[tid + 3 * info.np * idbuf];
	
	info.axayaz[tid] = s;
    }
}

__global__ void _gather(const float * input, const int * indices, float * output, const int n)
{
    const int tid = threadIdx.x + blockDim.x * blockIdx.x;
    
    if (tid < n)
	output[tid] = input[(tid % 6) + 6 * indices[tid / 6]];
}

#include <cmath>
#include <unistd.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>

#include "profiler-dpd.h"
#include "rring-buffer.h"

using namespace thrust;

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

template<typename T> T * _ptr(device_vector<T>& v) { return raw_pointer_cast(v.data()); }

ProfilerDPD * myprof = NULL;
RRingBuffer * rrbuf = NULL;

void forces_dpd_cuda(float * const _xyzuvw, float * const _axayaz,
		     int * const order, const int np,
		     const float rc,
		     const float XL, const float YL, const float ZL,
		     const float aij,
		     const float gamma,
		     const float sigma,
		     const float invsqrtdt,
		     float * const _rsamples, int nsamples)
{
    static bool initialized = false;

    if (!initialized)
    {
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	if (!prop.canMapHostMemory)
	{
	    printf("Capability zero-copy not there! Aborting now.\n");
	    abort();
	}
	else
	{
	    cudaSetDeviceFlags(cudaDeviceMapHost);
	    cudaDeviceSetCacheConfig (cudaFuncCachePreferShared);
	    //cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
	}

	initialized = true;
    }

    if (rrbuf == NULL)
	rrbuf = new RRingBuffer(50 * np * 3 * collapsefactor * collapsefactor * collapsefactor);

    if (myprof == NULL)
#ifdef _PROFILE_
	myprof = new ProfilerDPD(true);
#else
    myprof = new ProfilerDPD(false);
#endif
    
    int nx = (int)ceil(XL / (collapsefactor *rc));
    int ny = (int)ceil(YL / (collapsefactor *rc));
    int nz = (int)ceil(ZL / (collapsefactor *rc));
    const int ncells = nx * ny * nz;
    
    device_vector<int> starts(ncells + 1);
    device_vector<float> xyzuvw(_xyzuvw, _xyzuvw + np * 6), axayaz(np * 3);
    
    InfoDPD c;
    c.ncells = make_int3(nx, ny, nz);
    c.np = np;
    c.domainsize = make_float3(XL, YL, ZL);
    c.domainstart = make_float3(-XL * 0.5, -YL * 0.5, -ZL * 0.5);
    c.invrc = 1.f / rc;
    c.aij = aij;
    c.gamma = gamma;
    c.sigmaf = sigma * invsqrtdt;
    c.xyzuvw = _ptr(xyzuvw);
    c.axayaz = _ptr(axayaz);
    c.nsamples = rrbuf->nsamples();
    c.rsamples = rrbuf->buffer();
    c.rsamples_start = rrbuf->start();

    device_vector<float> rsamples;
    if (_rsamples != NULL)
    {
	rsamples.resize(nsamples);
	copy(_rsamples, _rsamples + nsamples, rsamples.begin());

	c.nsamples = nsamples;
	c.rsamples = _ptr(rsamples);
	c.rsamples_start = 0;
    }
    else
	nsamples = rrbuf->nsamples();
    
    c.starts = _ptr(starts);
    
    CUDA_CHECK(cudaMemcpyToSymbol(info, &c, sizeof(c)));

    device_vector<int> codes(np), pids(np);
    pid2code<<<(np + 127) / 128, 128>>>(_ptr(codes), _ptr(pids));

    sort_by_key(codes.begin(), codes.end(), pids.begin());

    {
	device_vector<float> tmp(xyzuvw.begin(), xyzuvw.end());
	_gather<<<(6 * np + 127) / 128, 128>>>(_ptr(tmp), _ptr(pids), _ptr(xyzuvw), 6 * np);
	CUDA_CHECK(cudaPeekAtLastError());
    }
    
    device_vector<int> cids(ncells + 1);
    sequence(cids.begin(), cids.end());

    lower_bound(codes.begin(), codes.end(), cids.begin(), cids.end(), starts.begin());

    int * consumed = NULL;
    cudaHostAlloc((void **)&consumed, sizeof(int), cudaHostAllocMapped);
    assert(consumed != NULL);
    *consumed = 0;
    
    {
	size_t textureoffset = 0;
	cudaChannelFormatDesc fmt =  cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindSigned);
	texStart.channelDesc = fmt;
	texStart.filterMode = cudaFilterModePoint;
	texStart.mipmapFilterMode = cudaFilterModePoint;
	texStart.normalized = 0;
	cudaBindTexture(&textureoffset, &texStart, c.starts, &fmt, sizeof(int) * (ncells + 1));
	
	fmt =  cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	texParticles.channelDesc = fmt;
	texParticles.filterMode = cudaFilterModePoint;
	texParticles.mipmapFilterMode = cudaFilterModePoint;
	texParticles.normalized = 0;
	cudaBindTexture(&textureoffset, &texParticles, c.xyzuvw, &fmt, sizeof(float) * 6 * np);

	float * tmp;

	const int nreplica = 24;

	CUDA_CHECK(cudaMalloc(&tmp, sizeof(float) * np * nreplica));
	CUDA_CHECK(cudaMemset(tmp, 0, sizeof(float) * np * nreplica));
	
	int * dconsumed = NULL;
	cudaHostGetDevicePointer(&dconsumed, consumed, 0);

	myprof->start();
	
	_dpd_forces<<<dim3(c.ncells.x, c.ncells.y, c.ncells.z), dim3(xbs, ybs, 1)>>>(tmp, dconsumed);

	myprof->force();
	CUDA_CHECK(cudaPeekAtLastError());

	_reduce<<<(3 * np + 127) / 128, 128>>>(tmp);
	myprof->reduce();
	CUDA_CHECK(cudaPeekAtLastError());
	
	CUDA_CHECK(cudaFree(tmp));
	
	if (*consumed >= nsamples)
	{
	    printf("done with code %d: consumed: %d\n", 7, *consumed);
	    printf("not a nice situation.\n");
	    abort();
	}
    }
	
    myprof->report();
    
    if (_rsamples == NULL)
	rrbuf->update(*consumed);
    
    cudaFreeHost(consumed);
   
    copy(xyzuvw.begin(), xyzuvw.end(), _xyzuvw);
    copy(axayaz.begin(), axayaz.end(), _axayaz);

    if (order != NULL)
	copy(pids.begin(), pids.end(), order);

#ifdef _CHECK_
    CUDA_CHECK(cudaThreadSynchronize());
    
    for(int i = 0; i < np; ++i)
    {
	printf("pid %d -> %f %f %f\n", i, (float)axayaz[0 + 3 * i], (float)axayaz[1 + 3* i], (float)axayaz[2 + 3 *i]);

	int cnt = 0;
	//const int pid = pids[i];

	printf("devi coords are %f %f %f\n", (float)xyzuvw[0 + 6 * i], (float)xyzuvw[1 + 6 * i], (float)xyzuvw[2 + 6 * i]);
	printf("host coords are %f %f %f\n", (float)_xyzuvw[0 + 6 * i], (float)_xyzuvw[1 + 6 * i], (float)_xyzuvw[2 + 6 * i]);
	
	for(int j = 0; j < np; ++j)
	{
	    if (i == j)
		continue;
 
	    float xr = _xyzuvw[0 + 6 *i] - _xyzuvw[0 + 6 * j];
	    float yr = _xyzuvw[1 + 6 *i] - _xyzuvw[1 + 6 * j];
	    float zr = _xyzuvw[2 + 6 *i] - _xyzuvw[2 + 6 * j];

	    xr -= c.domainsize.x *  ::floor(0.5f + xr / c.domainsize.x);
	    yr -= c.domainsize.y *  ::floor(0.5f + yr / c.domainsize.y);
	    zr -= c.domainsize.z *  ::floor(0.5f + zr / c.domainsize.z);

	    const float rij2 = xr * xr + yr * yr + zr * zr;
	    

	    cnt += rij2 < 1;
	}
	printf("i found %d host interactions and with cuda i found %d\n", cnt, (int)axayaz[0 + 3 * i]);
	assert(cnt == (float)axayaz[0 + 3 * i]);
    }
    
    printf("test done.\n");
    sleep(1);
    exit(0);
#endif
}

void forces_dpd_cuda(float * const xp, float * const yp, float * const zp,
		     float * const xv, float * const yv, float * const zv,
		     float * const xa, float * const ya, float * const za,
		     int * const order, const int np,
		     const float rc,
		     const float LX, const float LY, const float LZ,
		     const float aij,
		     const float gamma,
		     const float sigma,
		     const float invsqrtdt,
		     float * const rsamples, int nsamples)
{
    float * pv = new float[6 * np];

    for(int i = 0; i < np; ++i)
    {
	pv[0 + 6 * i] = xp[i];
	pv[1 + 6 * i] = yp[i];
	pv[2 + 6 * i] = zp[i];
	pv[3 + 6 * i] = xv[i];
	pv[4 + 6 * i] = yv[i];
	pv[5 + 6 * i] = zv[i];
    }

    float * a = new float[3 * np];
    
    forces_dpd_cuda(pv, a, order, np, rc, LX, LY, LZ,
		    aij, gamma, sigma, invsqrtdt, rsamples,  nsamples);

    for(int i = 0; i < np; ++i)
    {
	xp[i] = pv[0 + 6 * i]; 
	yp[i] = pv[1 + 6 * i]; 
	zp[i] = pv[2 + 6 * i]; 
	xv[i] = pv[3 + 6 * i]; 
	yv[i] = pv[4 + 6 * i]; 
	zv[i] = pv[5 + 6 * i];
    }

    delete [] pv;
     
    for(int i = 0; i < np; ++i)
    {
	xa[i] = a[0 + 3 * i];
	ya[i] = a[1 + 3 * i];
	za[i] = a[2 + 3 * i];
    }

    delete [] a;
}