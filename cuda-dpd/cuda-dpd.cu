#include <cstdio>
#include <cassert>

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
    
    state  = 0x4beb5d59*state + 0x2600e1f7; // LCG
    wstate = wstate + 0x8009d14b + ((((signed int)wstate)>>31)&0xda879add); // OWS
    
    unsigned int v = (state ^ (state>>26))+wstate;
    unsigned int r = (v^(v>>20))*0x6957f5a7;
    
    double res = r / (4294967295.0f);
    return res;
}

#ifndef NDEBUG
#define _CHECK_
#endif

struct InfoDPD
{
    int3 ncells;
    int np;
    float3 domainsize, invdomainsize, domainstart;
    float invrc, aij, gamma, sigmaf;
    float *xyzuvw, *axayaz;
};

__constant__ InfoDPD info;
 
texture<float2, cudaTextureType1D> texParticles;
texture<int, cudaTextureType1D> texStart, texEnd;

#define COLS 8
#define ROWS (32 / COLS)
#define _XCPB_ 2
#define _YCPB_ 2
#define _ZCPB_ 2
#define CPB (_XCPB_ * _YCPB_ * _ZCPB_)

__global__ void _dpd_forces_saru(int idtimestep)
{
    assert(warpSize == COLS * ROWS);
    assert(blockDim.x == warpSize && blockDim.y == CPB && blockDim.z == 1);
    assert(ROWS * 3 <= warpSize);

    const int tid = threadIdx.x;
    const int subtid = tid % COLS;
    const int slot = tid / COLS;
    const int wid = threadIdx.y;
     
    __shared__ int volatile starts[CPB][32], scan[CPB][32];

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

    for(int L = 1; L < 32; L <<= 1)
	mycount += (tid >= L) * __shfl_up(mycount, L) ;

    if (tid < 27)
	scan[wid][tid] = mycount;

    const int dststart = starts[wid][0];
    const int nsrc = scan[wid][26], ndst = scan[wid][0];
    
    __shared__ volatile float dpv[CPB][ROWS][6], spv[CPB][COLS][6];
    __shared__ volatile int spid[CPB][COLS];

    for(int d = 0; d < ndst; d += ROWS)
    {
	const int np1 = min(ndst - d, ROWS);
	
	for(int i = tid; i < np1 ; i += warpSize)
	    for(int c = 0; c < 3; ++c)
	    {
		float2 tmp = tex1Dfetch(texParticles, c + 3 * (d + dststart + i));;
		dpv[wid][i][2 * c + 0] = tmp.x;
		dpv[wid][i][2 * c + 1] = tmp.y;
	    }

	float f[3];
	for(int c = 0; c < 3; ++c)
		f[c] = 0;

	for(int s = 0; s < nsrc; s += COLS)
	{
	    const int np2 = min(nsrc - s, COLS);
  
	    for(int i = tid; i < np2; i += warpSize)
	    {
		const int pid = s + i;
		const int key9 = 9 * (pid >= scan[wid][8]) + 9 * (pid >= scan[wid][17]);
		const int key3 = 3 * (pid >= scan[wid][key9 + 2]) + 3 * (pid >= scan[wid][key9 + 5]);
		const int key1 = (pid >= scan[wid][key9 + key3]) + (pid >= scan[wid][key9 + key3 + 1]);
		const int key = key9 + key3 + key1;
		assert(pid >= (key ? scan[wid][key - 1] : 0) && pid < scan[wid][key]);
		
		const int localid = pid - s;
		
		const int myspid = starts[wid][key] + pid - (key ? scan[wid][key - 1] : 0);
		for(int c = 0; c < 3; ++c)
		{
		    float2 tmp = tex1Dfetch(texParticles, c + 3 * myspid);
		    spv[wid][localid % COLS][2 * c + 0] = tmp.x;
		    spv[wid][localid % COLS][2 * c + 1] = tmp.y;
		}		
	
		spid[wid][localid % COLS] =  myspid;
	    }
	    
	    {		
		const float xforce = f[0];
		const float yforce = f[1];
		const float zforce = f[2];
			    
		const float xdiff = dpv[wid][slot][0] - spv[wid][subtid][0];
		const float ydiff = dpv[wid][slot][1] - spv[wid][subtid][1];
		const float zdiff = dpv[wid][slot][2] - spv[wid][subtid][2];

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
		    xr * (dpv[wid][slot][3] - spv[wid][subtid][3]) +
		    yr * (dpv[wid][slot][4] - spv[wid][subtid][4]) +
		    zr * (dpv[wid][slot][5] - spv[wid][subtid][5]);
		    
		const int gd = dststart + d + slot;
		const int gs = spid[wid][subtid];
		const float mysaru = saru(min(gs, gd), max(gs, gd), idtimestep);
		const float myrandnr = 3.464101615f * mysaru - 1.732050807f;
		 
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
	}
	
	for(int L = COLS / 2; L > 0; L >>=1)
		for(int c = 0; c < 3; ++c)
		    f[c] += __shfl_xor(f[c], L);

	const float fcontrib = f[subtid % 3];
	const int dstpid = dststart + d + slot;
	const int c = (subtid % 3);

	if (slot < np1)
	    info.axayaz[c + 3 * dstpid] = fcontrib;
    }
}

#include <cmath>
#include <unistd.h>

#include <thrust/device_vector.h>
using namespace thrust;

#include "profiler-dpd.h"
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

template<typename T> T * _ptr(device_vector<T>& v) { return raw_pointer_cast(v.data()); }

ProfilerDPD * myprof = NULL;

void forces_dpd_cuda(float * const _xyzuvw, float * const _axayaz,
		     int * const order, const int np,
		     const float rc,
		     const float XL, const float YL, const float ZL,
		     const float aij,
		     const float gamma,
		     const float sigma,
		     const float invsqrtdt)
{
    if (myprof == NULL)
	myprof = new ProfilerDPD();
    
    int nx = (int)ceil(XL / rc);
    int ny = (int)ceil(YL / rc);
    int nz = (int)ceil(ZL / rc);
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
	
	fmt = cudaCreateChannelDesc<float2>();
	texParticles.channelDesc = fmt;
	texParticles.filterMode = cudaFilterModePoint;
	texParticles.mipmapFilterMode = cudaFilterModePoint;
	texParticles.normalized = 0;
	cudaBindTexture(&textureoffset, &texParticles, c.xyzuvw, &fmt, sizeof(float) * 6 * np);
    }
    
    myprof->start();

    static int tid = 0;

    _dpd_forces_saru<<<dim3(c.ncells.x / _XCPB_,
			    c.ncells.y / _YCPB_,
			    c.ncells.z / _ZCPB_), dim3(32, CPB)>>>(tid);

    ++tid;

    CUDA_CHECK(cudaPeekAtLastError());
	
    myprof->force();	
    myprof->report();
    
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
		     const float invsqrtdt)
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
		    aij, gamma, sigma, invsqrtdt);

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