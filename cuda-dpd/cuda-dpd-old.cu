#include <cstdio>
#include <cassert>

//#define _CHECK_

const int collapsefactor = 1;
 
struct InfoDPD
{
    int3 ncells;
    int np, nsamples, rsamples_start;
    float3 domainsize, invdomainsize, domainstart;
    float invrc, aij, gamma, sigmaf;
    float *xyzuvw, *axayaz, *rsamples;
};

__constant__ InfoDPD info;
 
#include "cell-lists.h"

const int xbs = 16;
const int ybs = 6;

__device__ void _ftable(
    float p1[3][ybs], float p2[3][xbs], float v1[3][ybs], float v2[3][xbs],
    const int np1, const int np2, const int nonzero_start, const int rsamples_start,
    float a1[3][ybs], float a2[3][xbs])
{
    assert(np2 <= xbs);
    assert(np1 <= ybs);
    assert(np1 <= xbs * ybs);
    assert(blockDim.x == xbs && xbs == xbs);
    assert(blockDim.y == ybs);

    const int lx = threadIdx.x;
    const int ly = threadIdx.y;

    float xmyforce = 0, ymyforce = 0, zmyforce = 0;
    
    {
	const bool valid = (lx < np2 && ly < np1) * (lx > ly + nonzero_start);
	
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

	if (valid)
	{
#ifdef _CHECK_
	    xmyforce = (rij2 < 1);
	    ymyforce = 0;
	    zmyforce = 0;
#else		    
	    xmyforce = strength * xr;
	    ymyforce = strength * yr;
	    zmyforce = strength * zr;
#endif
	}
    }

    {
	assert(xbs == 16 && warpSize == 32);

	float xmysum = xmyforce, ymysum = ymyforce, zmysum = zmyforce;
	
#pragma unroll
	for(int l = 1; l < 16; l <<= 1)
	{
	    const float xother = __shfl_xor(xmysum, l);
	    const float yother = __shfl_xor(ymysum, l);
	    const float zother = __shfl_xor(zmysum, l);

	    xmysum += xother;
	    ymysum += yother;
	    zmysum += zother;
	}

	if (lx == 0 && ly < np1)
	{
	    a1[0][ly] = xmysum;
	    a1[1][ly] = ymysum;
	    a1[2][ly] = zmysum;
	}
    }

    {
	assert(xbs == 16 && warpSize == 32 && ybs == 6);

	__shared__ float buf[3][3][16];

	xmyforce += __shfl_xor(xmyforce, 16);
	ymyforce += __shfl_xor(ymyforce, 16);
	zmyforce += __shfl_xor(zmyforce, 16);
	
	if ((ly & 1) == 0)
	{
	    const int entry = ly >> 1;
	    buf[0][entry][lx] = xmyforce;
	    buf[1][entry][lx] = ymyforce;
	    buf[2][entry][lx] = zmyforce;
	}

	__syncthreads();

	if (lx < np2 && threadIdx.y == 0)
	{
	    a2[0][lx] = buf[0][0][lx] + buf[0][1][lx] + buf[0][2][lx];
	    a2[1][lx] = buf[1][0][lx] + buf[1][1][lx] + buf[1][2][lx];
	    a2[2][lx] = buf[2][0][lx] + buf[2][1][lx] + buf[2][2][lx];
	}
    }
}

texture<float, cudaTextureType1D> texParticles;

__device__ void _cellscells(const int p1start[4], const int p1count[4], const int p2start[4][4], const int p2scan[4][4],
			    const int maxnp1, const int maxnp2, const bool self, int rsamples_start,
			    float * const axayaz)
{
    __shared__ float pva1[9][ybs], pva2[9][xbs];

    const int l = threadIdx.x + blockDim.x * threadIdx.y;
    const int BS = blockDim.x * blockDim.y;
      
    for(int ty = 0; ty < maxnp1; ty += ybs)
    {
	int np1[4];
	
#pragma unroll
	for(int i = 0; i < 4; ++i)
	    np1[i] = max(0, min(ybs, p1count[i] - ty));
	
#pragma unroll
	for(int i = 0; i < 4; ++i)
	    assert(BS >= np1[i] * 6);

	float pva1contrib[4];
#pragma unroll
	for(int i = 0; i < 4; ++i)
	    if (l < np1[i] * 6)
		pva1contrib[i] = tex1Dfetch(texParticles, 6 * (p1start[i] + ty) + l);

	float pva1result[4] = {0, 0, 0, 0};
	
	for(int tx = 0; tx < maxnp2; tx += xbs)
	{
	    int np2[4];
#pragma unroll
	    for(int i = 0; i < 4; ++i)
		np2[i] = max(0, min(xbs, p2scan[i][3] - tx));

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

	    float pva2result[4] = {0, 0, 0, 0};
	   
#pragma unroll
	    for(int i = 0; i < 4; ++i)
	    {
		if (np1[i] * np2[i] == 0)
		    continue;
		
		if (l < np1[i] * 6)
		    pva1[l % 6][l / 6] = pva1contrib[i];

		assert(np2[i] * 6 <= BS);
		assert(BS >= np2[i] * 3);
	   
		if (l < np2[i] * 6)
		    pva2[l % 6][l / 6] = pva2contrib[i];

		__syncthreads();

		_ftable(pva1, pva2, &pva1[3], &pva2[3], np1[i], np2[i], i == 0 ? ty - tx : -30000, rsamples_start, &pva1[6], &pva2[6]);

		__syncthreads();
		
		rsamples_start += np1[i] * np2[i];

		if (l < np1[i] * 3)
		    pva1result[i] += pva1[6 + (l % 3)][l / 3];
		
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
		    atomicAdd(axayaz + c + 3 * pid, pva2result[i]);
#else
		    atomicAdd(axayaz + c + 3 * pid, -pva2result[i]);
#endif
		}
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
    int3 indx = make_int3(blockIdx.x, blockIdx.y, blockIdx.z); 
	    
    indx.x += (shiftcode & 1);
    indx.y += ((shiftcode >> 1) & 1);
    indx.z += ((shiftcode >> 2) & 1);
	    
    indx.x = (indx.x + info.ncells.x) % info.ncells.x;
    indx.y = (indx.y + info.ncells.y) % info.ncells.y;
    indx.z = (indx.z + info.ncells.z) % info.ncells.z;

    return indx.x + info.ncells.x * (indx.y + info.ncells.y * indx.z);//encode(indx.x, indx.y, indx.z);
}

__constant__ int edgeslutcount[4] = {4, 4, 3, 3};
__constant__ int edgeslutstart[4] = {0, 4, 8, 11};
__constant__ int edgeslut[14] = {0, 1, 2, 7, 2, 4, 6, 7, 4, 5, 7, 4, 0, 7};

texture<int, cudaTextureType1D> texStart, texEnd;

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
	    p1counts[i] = tex1Dfetch(texEnd, cid1);
	    
	    assert( tex1Dfetch(texEnd, cid1) - tex1Dfetch(texStart, cid1) >= 0);
	}
		
	if (j < edgeslutcount[i])
	{
	    const int cid2 = _cid(edgeslut[j + edgeslutstart[i]]);
	    
	    p2starts[i][j] = tex1Dfetch(texStart, cid2);
	    p2scans[i][j] = tex1Dfetch(texEnd, cid2);

	    assert( tex1Dfetch(texEnd, cid2) - tex1Dfetch(texStart, cid2) >= 0);
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

__device__ float saru(unsigned int seed1, unsigned int seed2, unsigned int seed3)
{
    seed3 ^= (seed1<<7)^(seed2>>6);
    seed2 += (seed1>>4)^(seed3>>15);
    seed1 ^= (seed2<<9)+(seed3<<8);
    seed3 ^= 0xA5366B4D*((seed2>>11) ^ (seed1<<1));
    seed2 += 0x72BE1579*((seed1<<4)  ^ (seed3>>16));
    seed1 ^= 0X3F38A6ED*((seed3>>5)  ^ (((signed int)seed2)>>22));
    seed2 += seed1*seed3;
    seed1 += seed3 ^ (seed2>>2);
    seed2 ^= ((signed int)seed2)>>17;
    
    int state  = 0x79dedea3*(seed1^(((signed int)seed1)>>14));
    int wstate = (state + seed2) ^ (((signed int)state)>>8);
    state  = state + (wstate*(wstate^0xdddf97f5));
    wstate = 0xABCB96F7 + (wstate>>1);
    
    state  = 0x4beb5d59*state + 0x2600e1f7;             // LCG
    wstate = wstate + 0x8009d14b + ((((signed int)wstate)>>31)&0xda879add); // OWS
    
    unsigned int v = (state ^ (state>>26))+wstate;
    unsigned int r = (v^(v>>20))*0x6957f5a7;
    
    double res = r / (4294967295.0f);
    return res;
}

#define COLS 8
#define ROWS (32 / COLS)
#define _XCPB_ 2
#define _YCPB_ 2
#define _ZCPB_ 1
#define _CPB_ (_XCPB_ * _YCPB_ * _ZCPB_)
__global__ void _dpd_forces_saru(int idtimestep)
{
    assert(warpSize == COLS * ROWS);
    assert(blockDim.x == warpSize && blockDim.y == _CPB_ && blockDim.z == 1);
    assert(ROWS * 3 <= warpSize);

    const int tid = threadIdx.x;
    const int subtid = tid % COLS;
    const int slot = tid / COLS;
    const int wid = threadIdx.y;
     
    __shared__ int volatile starts[_CPB_][32], scan[_CPB_][32];

    int mycount = 0; 
    if (tid < 27)
    {
	const int dx = (1 + tid) % 3;
	const int dy = (1 + (tid / 3)) % 3;
	const int dz = (1 + (tid / 9)) % 3;

	const int xcid = (blockIdx.x * _XCPB_ + ((threadIdx.y) % _XCPB_) + dx - 1 + info.ncells.x) % info.ncells.x;
	const int ycid = (blockIdx.y * _YCPB_ + ((threadIdx.y / _XCPB_) % _YCPB_) + dy - 1 + info.ncells.y) % info.ncells.y;
	const int zcid = (blockIdx.z * _ZCPB_ + ((threadIdx.y / (_XCPB_ * _YCPB_)) % _ZCPB_) + dz - 1 + info.ncells.z) % info.ncells.z;
	const int cid = xcid + info.ncells.x * (ycid + info.ncells.y * zcid);

	starts[wid][tid] = tex1Dfetch(texStart, cid);
	mycount = tex1Dfetch(texEnd, cid) - starts[wid][tid];
    }

#pragma unroll
    for(int L = 1; L < 32; L <<= 1)
	mycount += (tid >= L) * __shfl_up(mycount, L) ;

    if (tid < 27)
	scan[wid][tid] = mycount;

    const int dststart = starts[wid][0];
    const int nsrc = scan[wid][26], ndst = scan[wid][0];
    
    float f[3];
    __shared__ volatile float dpv[_CPB_][ROWS][6], spv[_CPB_][6][COLS];
    __shared__ volatile int spid[_CPB_][COLS];
    
    for(int d = 0; d < ndst; d += ROWS)
    {
	const int np1 = min(ndst - d, ROWS);
	
	for(int i = tid; i < np1 * 6 ; i += warpSize)
	    dpv[wid][i / 6][i % 6] = tex1Dfetch(texParticles, i + 6 * (d + dststart));

#pragma unroll
	for(int c = 0; c < 3; ++c)
		f[c] = 0;
	
	for(int s = 0; s < nsrc; s += COLS)
	{
	    const int np2 = min(nsrc - s, COLS);

	    for(int i = tid; i < np2 * 6; i += warpSize)
	    {
		const int pid = s + i / 6;
		const int key9 = 9 * (pid >= scan[wid][8]) + 9 * (pid >= scan[wid][17]);
		const int key3 = 3 * (pid >= scan[wid][key9 + 2]) + 3 * (pid >= scan[wid][key9 + 5]);
		const int key1 = (pid >= scan[wid][key9 + key3]) + (pid >= scan[wid][key9 + key3 + 1]);
		const int key = key9 + key3 + key1;
		assert(pid >= (key ? scan[wid][key - 1] : 0) && pid < scan[wid][key]);
		
		const int localid = pid - s;
		const int c = i % 6;
		const int myspid = starts[wid][key] + pid - (key ? scan[wid][key - 1] : 0);
		spv[wid][c][localid % COLS] = tex1Dfetch(texParticles, c + 6 * myspid);
				
		if (c == 0)
		    spid[wid][localid % COLS] = myspid;
	    }

	    
	    {		
		const float xforce = f[0];
		const float yforce = f[1];
		const float zforce = f[2];
			    
		const float xdiff = dpv[wid][slot][0] - spv[wid][0][subtid];
		const float ydiff = dpv[wid][slot][1] - spv[wid][1][subtid];
		const float zdiff = dpv[wid][slot][2] - spv[wid][2][subtid];

		const float _xr = xdiff - info.domainsize.x * floorf(0.5f + xdiff * info.invdomainsize.x);
		const float _yr = ydiff - info.domainsize.y * floorf(0.5f + ydiff * info.invdomainsize.y);
		const float _zr = zdiff - info.domainsize.z * floorf(0.5f + zdiff * info.invdomainsize.z);

		const float rij2 = _xr * _xr + _yr * _yr + _zr * _zr;
		const float invrij = rsqrtf(rij2);
		const float rij = rij2 * invrij;
		const float wr = max((float)0, 1 - rij * info.invrc);
		
		const float xr = _xr * invrij;
		const float yr = _yr * invrij;
		const float zr = _zr * invrij;
		
		const float rdotv =
		    xr * (dpv[wid][slot][3] - spv[wid][3][subtid]) +
		    yr * (dpv[wid][slot][4] - spv[wid][4][subtid]) +
		    zr * (dpv[wid][slot][5] - spv[wid][5][subtid]);
		    
		const int gd = dststart + d + slot;
		const int gs = spid[wid][subtid];
		const float mysaru = saru(min(gs, gd), max(gs, gd), idtimestep);
		const float myrandnr = 3.4641016151377544f * mysaru - 1.7320508075688772f;
		
		const float strength = (info.aij - info.gamma * wr * rdotv + info.sigmaf * myrandnr) * wr;
		const bool valid = (d + slot != s + subtid) && (slot < np1) && (subtid < np2);
		
		if (valid)
		{
#ifdef _CHECK_
		    f[0] = xforce + (rij2 < 1);
		    f[1] = yforce + wr;
		    f[2] = zforce + 0;
#else		    	     
		    f[0] = xforce + strength * xr;
		    f[1] = yforce + strength * yr;
		    f[2] = zforce + strength * zr;
#endif
		}
	    } 
	} //end for s

#pragma unroll
	for(int L = COLS / 2; L > 0; L >>=1)
#pragma unroll
		for(int c = 0; c < 3; ++c)
		    f[c] += __shfl_xor(f[c], L);

	const float fcontrib = f[subtid % 3];
	
	if (slot < np1)
	{
	    const int dstpid = dststart + d + slot;
	    const int c = (subtid % 3);

	    info.axayaz[c + 3 * dstpid] = fcontrib;
	    //printf("tid %d: dst %d -> f %f (np1 is %d)\n", tid, c + 3 * (slot + 2 * pass), fcontrib, np1);
	}
    } //end for d
}

#include <cmath>
#include <unistd.h>

#include <thrust/device_vector.h>
using namespace thrust;

#include "profiler-dpd.h"
#include "rring-buffer.h"

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
	    cudaSetDeviceFlags(cudaDeviceMapHost);

//	CUDA_CHECK(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));  
	
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
        
    device_vector<float> xyzuvw(_xyzuvw, _xyzuvw + np * 6), axayaz(np * 3);
    
    InfoDPD c;
    c.ncells = make_int3(nx, ny, nz);
    c.np = np;
    c.domainsize = make_float3(XL, YL, ZL);
    c.invdomainsize = make_float3(1 / XL, 1 / YL, 1 / ZL);
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
    
    CUDA_CHECK(cudaMemcpyToSymbol(info, &c, sizeof(c)));

    device_vector<int> starts(ncells + 1), ends(ncells + 1);
    build_clists(_ptr(xyzuvw), np, rc, c.ncells.x, c.ncells.y, c.ncells.z,
		 c.domainstart.x, c.domainstart.y, c.domainstart.z,
		 order, _ptr(starts), _ptr(ends));

    {
	size_t textureoffset = 0;
	cudaChannelFormatDesc fmt =  cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindSigned);
	texStart.channelDesc = fmt;
	texStart.filterMode = cudaFilterModePoint;
	texStart.mipmapFilterMode = cudaFilterModePoint;
	texStart.normalized = 0;
	cudaBindTexture(&textureoffset, &texStart, _ptr(starts), &fmt, sizeof(int) * (ncells + 1));

	texEnd.channelDesc = fmt;
	texEnd.filterMode = cudaFilterModePoint;
	texEnd.mipmapFilterMode = cudaFilterModePoint;
	texEnd.normalized = 0;
	cudaBindTexture(&textureoffset, &texEnd, _ptr(ends), &fmt, sizeof(int) * (ncells + 1));
	
	fmt =  cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	texParticles.channelDesc = fmt;
	texParticles.filterMode = cudaFilterModePoint;
	texParticles.mipmapFilterMode = cudaFilterModePoint;
	texParticles.normalized = 0;
	cudaBindTexture(&textureoffset, &texParticles, c.xyzuvw, &fmt, sizeof(float) * 6 * np);
    }
    
    int * consumed = NULL;
    cudaHostAlloc((void **)&consumed, sizeof(int), cudaHostAllocMapped);
    assert(consumed != NULL);
    *consumed = 0;

     if (true)
     {
	myprof->start();

	static int tid = 0;

	_dpd_forces_saru<<<dim3(c.ncells.x / _XCPB_,
				c.ncells.y / _YCPB_,
				c.ncells.z / _ZCPB_), dim3(32, _CPB_)>>>(tid);

/*	_dpd_forces_saru<<<dim3(1,
				1,
				1), dim3(32, _CPB_)>>>(tid);
*/
	++tid;

	CUDA_CHECK(cudaPeekAtLastError());
	
	myprof->force();
	myprof->reduce();
    }
    else
    {
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
    
#ifdef _CHECK_
    CUDA_CHECK(cudaThreadSynchronize());
    
    for(int i = 0; i < np; ++i)
    {
	printf("pid %d -> %f %f %f\n", i, (float)axayaz[0 + 3 * i], (float)axayaz[1 + 3* i], (float)axayaz[2 + 3 *i]);

	int cnt = 0;
	float fc = 0;
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
	    const float invrij = rsqrtf(rij2);
	    const float rij = rij2 * invrij;
	    const float wr = max((float)0, 1 - rij * c.invrc);
	
	    const bool collision =  rij2 < 1;

	    if (collision)
		fc += wr;//	printf("ref p %d colliding with %d\n", i, j);
	    
	    cnt += collision;
	}
	printf("i found %d host interactions and with cuda i found %d\n", cnt, (int)axayaz[0 + 3 * i]);
	assert(cnt == (float)axayaz[0 + 3 * i]);
	printf("fc aij ref %f vs cuda %e\n", fc,  (float)axayaz[1 + 3 * i]);
	assert(fabs(fc - (float)axayaz[1 + 3 * i]) < 1e-4);
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