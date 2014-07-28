#include <cassert>

struct InfoDPD
{
    int3 ncells;
    float3 domainsize, invdomainsize, domainstart;
    float invrc, aij, gamma, sigmaf;
};

__constant__ InfoDPD info;

__device__
__forceinline__ float saru(unsigned int seed1, unsigned int seed2, unsigned int seed3)
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
 
#define COLS 8
#define ROWS (32 / COLS)
#define CPB 4
 
__global__ __launch_bounds__(32 * CPB, 16) 
    void _dpd_forces_saru(float * const axayaz,
			  const int idtimestep,
			  cudaTextureObject_t texDstStart, cudaTextureObject_t texDstCount,
			  cudaTextureObject_t texSrcStart, cudaTextureObject_t texSrcCount,
			  cudaTextureObject_t texDstParticles, cudaTextureObject_t texSrcParticles,
			  const int * const nonempty_destcells, const int nonempties, const int gdpid_start, const int gspid_start)
{
    assert(warpSize == COLS * ROWS);
    assert(blockDim.x == warpSize && blockDim.y == CPB && blockDim.z == 1);
    assert(ROWS * 3 <= warpSize);

    const int mycidentry = blockIdx.x * CPB + threadIdx.y;

    if (mycidentry >= nonempties)
	return;

    const int mycid = nonempty_destcells[mycidentry];
    
    const int xmycid = mycid % info.ncells.x;
    const int ymycid = (mycid / info.ncells.x) % info.ncells.y;
    const int zmycid = (mycid / info.ncells.x / info.ncells.y) % info.ncells.z;

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

	const int xcid = (xmycid + dx - 1 + info.ncells.x) % info.ncells.x;
	const int ycid = (ymycid + dy - 1 + info.ncells.y) % info.ncells.y;
	const int zcid = (zmycid + dz - 1 + info.ncells.z) % info.ncells.z;
	const int cid = xcid + info.ncells.x * (ycid + info.ncells.y * zcid);

	starts[wid][tid] = tex1Dfetch<int>(texSrcStart, cid);
	mycount = tex1Dfetch<int>(texSrcCount, cid);
    }

    for(int L = 1; L < 32; L <<= 1)
	mycount += (tid >= L) * __shfl_up(mycount, L) ;

    if (tid < 27)
	scan[wid][tid] = mycount;

    const int dststart = tex1Dfetch<int>(texDstStart, mycid);
    const int nsrc = scan[wid][26], ndst = tex1Dfetch<int>(texDstCount, mycid);
 
    for(int d = 0; d < ndst; d += ROWS)
    {
	const int np1 = min(ndst - d, ROWS);

	const int dpid = dststart + d + slot;
	const int gdpid = gdpid_start + dpid;
	const int entry = 3 * dpid;
	float2 dtmp0 = tex1Dfetch<float2>(texDstParticles, entry);
	float2 dtmp1 = tex1Dfetch<float2>(texDstParticles, entry + 1);
	float2 dtmp2 = tex1Dfetch<float2>(texDstParticles, entry + 2);
	
	float f[3] = {0, 0, 0};

	for(int s = 0; s < nsrc; s += COLS)
	{
	    const int np2 = min(nsrc - s, COLS);
  
	    const int pid = s + subtid;
	    const int key9 = 9 * (pid >= scan[wid][8]) + 9 * (pid >= scan[wid][17]);
	    const int key3 = 3 * (pid >= scan[wid][key9 + 2]) + 3 * (pid >= scan[wid][key9 + 5]);
	    const int key1 = (pid >= scan[wid][key9 + key3]) + (pid >= scan[wid][key9 + key3 + 1]);
	    const int key = key9 + key3 + key1;
	    assert(subtid >= np2 || pid >= (key ? scan[wid][key - 1] : 0) && pid < scan[wid][key]);

	    const int spid = starts[wid][key] + pid - (key ? scan[wid][key - 1] : 0);
	    const int gspid = gspid_start + spid;
	    const int sentry = 3 * spid;
	    const float2 stmp0 = tex1Dfetch<float2>(texSrcParticles, sentry);
	    const float2 stmp1 = tex1Dfetch<float2>(texSrcParticles, sentry + 1);
	    const float2 stmp2 = tex1Dfetch<float2>(texSrcParticles, sentry + 2);
	    
	    {
		const float xforce = f[0];
		const float yforce = f[1];
		const float zforce = f[2];
			    
		const float xdiff = dtmp0.x - stmp0.x;
		const float ydiff = dtmp0.y - stmp0.y;
		const float zdiff = dtmp1.x - stmp1.x;

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
		    xr * (dtmp1.y - stmp1.y) +
		    yr * (dtmp2.x - stmp2.x) +
		    zr * (dtmp2.y - stmp2.y);
	
		const float mysaru = saru(min(gspid, gdpid), max(spid, dpid), idtimestep);
		const float myrandnr = 3.464101615f * mysaru - 1.732050807f;
		 
		const float strength = (info.aij - info.gamma * wr * rdotv + info.sigmaf * myrandnr) * wr;
		const bool valid = (slot < np1) && (subtid < np2);
		
		if (valid)
		{
#ifdef _CHECK_
		    f[0] = xforce + (rij2 < 1);
		    f[1] = yforce + wr;
		    f[2] = zforce + rdotv *  (rij2 < 1);
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
	    axayaz[c + 3 * dstpid] = fcontrib;
    }
}

#include <cstdio>
#include <unistd.h>
#include <thrust/device_vector.h>

#include "cuda-dpd.h"
#include "../profiler-dpd.h"
#include "../cell-lists.h"

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

struct TextureWrap
{
    cudaTextureObject_t texObj;

    template<typename ElementType>
    TextureWrap(ElementType * data, const int n):
	texObj(0)
	{
	    struct cudaResourceDesc resDesc;
	    memset(&resDesc, 0, sizeof(resDesc));
	    resDesc.resType = cudaResourceTypeLinear;
	    resDesc.res.linear.devPtr = data;
	    resDesc.res.linear.sizeInBytes = n * sizeof(ElementType);
	    resDesc.res.linear.desc = cudaCreateChannelDesc<ElementType>();
    
	    struct cudaTextureDesc texDesc;
	    memset(&texDesc, 0, sizeof(texDesc));
	    texDesc.addressMode[0]   = cudaAddressModeWrap;
	    texDesc.addressMode[1]   = cudaAddressModeWrap;
	    texDesc.filterMode       = cudaFilterModePoint;
	    texDesc.readMode         = cudaReadModeElementType;
	    texDesc.normalizedCoords = 1;

	    texObj = 0;
	    CUDA_CHECK(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));
	}

    ~TextureWrap()
	{
	    CUDA_CHECK(cudaDestroyTextureObject(texObj));
	}
};

extern int tid;

void forces_dpd_cuda_bipartite(float * const _xyzuvw1, float * const _axayaz1, int * const order1, const int np1, const int gp1id_start,
			       float * const _xyzuvw2, float * const _axayaz2, int * const order2, const int np2, const int gp2id_start,
			       const float rc, const float XL, const float YL, const float ZL,
			       const float aij, const float gamma, const float sigma, const float invsqrtdt)
{
    int nx = (int)ceil(XL / rc);
    int ny = (int)ceil(YL / rc);
    int nz = (int)ceil(ZL / rc);
    const int ncells = nx * ny * nz;

#ifdef _CHECK_
    for(int i = 0; i < np1; ++i)
	for(int c = 3; c < 6; ++c)
	    _xyzuvw1[c + 6 * i] = 1e-1 * (drand48()* 2 - 1);
    
    for(int i = 0; i < np2; ++i)
	for(int c = 3; c < 6; ++c)
	    _xyzuvw2[c + 6 * i] = 1e-1 * (drand48()* 2 - 1);
#endif
    
    device_vector<float> xyzuvw1(_xyzuvw1, _xyzuvw1 + np1 * 6), axayaz1(np1 * 3);
    device_vector<float> xyzuvw2(_xyzuvw2, _xyzuvw2 + np2 * 6), axayaz2(np2 * 3);
    
    InfoDPD c;
    c.ncells = make_int3(nx, ny, nz);
    c.domainsize = make_float3(XL, YL, ZL);
    c.invdomainsize = make_float3(1 / XL, 1 / YL, 1 / ZL);
    c.domainstart = make_float3(-XL * 0.5, -YL * 0.5, -ZL * 0.5);
    c.invrc = 1.f / rc;
    c.aij = aij;
    c.gamma = gamma;
    c.sigmaf = sigma * invsqrtdt;
        
    CUDA_CHECK(cudaMemcpyToSymbol(info, &c, sizeof(c)));

    device_vector<int> starts1(ncells), ends1(ncells), nonempty1(ncells);
    device_vector<int> starts2(ncells), ends2(ncells), nonempty2(ncells);

    std::pair<int, int *> nonemptycells1;
    std::pair<int, int *> nonemptycells2;
    
    nonemptycells1.second = _ptr(nonempty1);
    nonemptycells2.second = _ptr(nonempty2);
    
    build_clists(_ptr(xyzuvw1), np1, rc, c.ncells.x, c.ncells.y, c.ncells.z,
		 c.domainstart.x, c.domainstart.y, c.domainstart.z,
		 order1, _ptr(starts1), _ptr(ends1), &nonemptycells1);

    build_clists(_ptr(xyzuvw2), np2, rc, c.ncells.x, c.ncells.y, c.ncells.z,
		 c.domainstart.x, c.domainstart.y, c.domainstart.z,
		 order2, _ptr(starts2), _ptr(ends2), &nonemptycells2);
  
    TextureWrap texStart1(_ptr(starts1), ncells), texCount1(_ptr(ends1), ncells);
    TextureWrap texStart2(_ptr(starts2), ncells), texCount2(_ptr(ends2), ncells);

    TextureWrap texParticles1((float2*)_ptr(xyzuvw1), 3 * np1);
    TextureWrap texParticles2((float2*)_ptr(xyzuvw2), 3 * np2);

    cudaStream_t stream1, stream2;
    
    CUDA_CHECK(cudaStreamCreate(&stream1));
    CUDA_CHECK(cudaStreamCreate(&stream2));
    
    ProfilerDPD::singletone().start();

    _dpd_forces_saru<<<(nonemptycells1.first + CPB - 1) / CPB, dim3(32, CPB), 0, stream1>>>(
	_ptr(axayaz1), tid, texStart1.texObj, texCount1.texObj, texStart2.texObj, texCount2.texObj,
	texParticles1.texObj, texParticles2.texObj, nonemptycells1.second, nonemptycells1.first, gp1id_start, gp2id_start);
    
    CUDA_CHECK(cudaPeekAtLastError());
    
    _dpd_forces_saru<<<(nonemptycells2.first + CPB - 1) / CPB, dim3(32, CPB), 0, stream2>>>(
	_ptr(axayaz2), tid, texStart2.texObj, texCount2.texObj, texStart1.texObj, texCount1.texObj,
	texParticles2.texObj, texParticles1.texObj, nonemptycells2.second, nonemptycells2.first, gp2id_start, gp1id_start);
    
    CUDA_CHECK(cudaPeekAtLastError());
    tid += 2;

    ProfilerDPD::singletone().force();	
    ProfilerDPD::singletone().report();

    CUDA_CHECK(cudaStreamDestroy(stream1));
    CUDA_CHECK(cudaStreamDestroy(stream2));
    
    copy(axayaz1.begin(), axayaz1.end(), _axayaz1);
    copy(axayaz2.begin(), axayaz2.end(), _axayaz2);

#ifdef _CHECK_
    printf("hello check: np1 %d np2: %d\n", np1, np2);
    printf("nonempty in p1: %d\n", nonemptycells1.first);
    sleep(2);
    for(int ii = 0; ii < np1; ++ii)
    {
	printf("pid %d -> %f %f %f\n", ii, (float)axayaz1[0 + 3 * ii], (float)axayaz1[1 + 3* ii], (float)axayaz1[2 + 3 *ii]);
	const int i = order1[ii];
	int cnt = 0;
	float fc = 0, fc2 = 0;
	printf("devi coords are %f %f %f\n", (float)xyzuvw1[0 + 6 * ii], (float)xyzuvw1[1 + 6 * ii], (float)xyzuvw1[2 + 6 * ii]);
	printf("host coords are %f %f %f\n", (float)_xyzuvw1[0 + 6 * i], (float)_xyzuvw1[1 + 6 * i], (float)_xyzuvw1[2 + 6 * i]);
	
	for(int j = 0; j < np2; ++j)
	{ 
	    float xr = _xyzuvw1[0 + 6 *i] - _xyzuvw2[0 + 6 * j];
	    float yr = _xyzuvw1[1 + 6 *i] - _xyzuvw2[1 + 6 * j];
	    float zr = _xyzuvw1[2 + 6 *i] - _xyzuvw2[2 + 6 * j];

	    xr -= c.domainsize.x *  ::floor(0.5f + xr / c.domainsize.x);
	    yr -= c.domainsize.y *  ::floor(0.5f + yr / c.domainsize.y);
	    zr -= c.domainsize.z *  ::floor(0.5f + zr / c.domainsize.z);

	    const float rij2 = xr * xr + yr * yr + zr * zr;
	    const float invrij = rsqrtf(rij2);
	    const float rij = rij2 * invrij;
	    const float wr = max((float)0, 1 - rij * c.invrc);
	
	    const bool collision =  rij2 < 1;

	    if (collision)
		fc += wr;
	    
	    cnt += collision;

	    xr /= rij;
	    yr /= rij;
	    zr /= rij;
	    
	    float xdv = _xyzuvw1[3 + 6 *i] - _xyzuvw2[3 + 6 * j];
	    float ydv = _xyzuvw1[4 + 6 *i] - _xyzuvw2[4 + 6 * j];
	    float zdv = _xyzuvw1[5 + 6 *i] - _xyzuvw2[5 + 6 * j];
	    float rdotv = xr * xdv + yr * ydv + zr * zdv;
	    fc2 += rdotv * (rij2 < 1);
	}
	printf("i found %d host interactions and with cuda i found %d\n", cnt, (int)axayaz1[0 + 3 * ii]);
	
	printf("fc aij ref %f vs cuda %e\n", fc,  (float)axayaz1[1 + 3 * ii]);
	printf("rdotv ref %f vs cuda %e\n", fc2,  (float)axayaz1[2 + 3 * ii]);
	assert(cnt == (float)axayaz1[0 + 3 * ii]);
	assert(fabs(fc - (float)axayaz1[1 + 3 * ii]) < 1e-4);
	assert(fabs(fc2 - (float)axayaz1[2 + 3 * ii]) < 1e-4);
    }

    printf("done with the first part\n");
 
    for(int ii = 0; ii < np2; ++ii)
    {
	const int i = order2[ii];
	
	printf("pid %d -> %f %f %f\n", i, (float)axayaz2[0 + 3 * ii], (float)axayaz2[1 + 3* ii], (float)axayaz2[2 + 3 *ii]);

	int cnt = 0;
	float fc = 0, fc2 = 0;
	printf("devi coords are %f %f %f\n", (float)xyzuvw2[0 + 6 * ii], (float)xyzuvw2[1 + 6 * ii], (float)xyzuvw2[2 + 6 * ii]);
	printf("host coords are %f %f %f\n", (float)_xyzuvw2[0 + 6 * i], (float)_xyzuvw2[1 + 6 * i], (float)_xyzuvw2[2 + 6 * i]);
	
	for(int j = 0; j < np1; ++j)
	{ 
	    float xr = _xyzuvw2[0 + 6 *i] - _xyzuvw1[0 + 6 * j];
	    float yr = _xyzuvw2[1 + 6 *i] - _xyzuvw1[1 + 6 * j];
	    float zr = _xyzuvw2[2 + 6 *i] - _xyzuvw1[2 + 6 * j];

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

	    xr /= rij;
	    yr /= rij;
	    zr /= rij;
	    
	    float xdv = _xyzuvw2[3 + 6 *i] - _xyzuvw1[3 + 6 * j];
	    float ydv = _xyzuvw2[4 + 6 *i] - _xyzuvw1[4 + 6 * j];
	    float zdv = _xyzuvw2[5 + 6 *i] - _xyzuvw1[5 + 6 * j];
	    float rdotv = xr * xdv + yr * ydv + zr * zdv;
	    fc2 += rdotv * (rij2 < 1);
	}
	
	printf("i found %d host interactions and with cuda i found %d\n", cnt, (int)axayaz2[0 + 3 * ii]);
	
	printf("fc aij ref %f vs cuda %e\n", fc,  (float)axayaz2[1 + 3 *ii]);
	printf("rdotv ref %f vs cuda %e\n", fc2,  (float)axayaz2[2 + 3 * ii]);
	assert(cnt == (float)axayaz2[0 + 3 * ii]);
	assert(fabs(fc - (float)axayaz2[1 + 3 * ii]) < 1e-4);
	assert(fabs(fc2 - (float)axayaz2[2 + 3 * ii]) < 1e-4);
    }
    printf("test done.\n");
    sleep(1);
    exit(0);
#endif
}

void forces_dpd_cuda_bipartite(float * const xp1, float * const yp1, float * const zp1,
			       float * const xv1, float * const yv1, float * const zv1,
			       float * const xa1, float * const ya1, float * const za1,
			       const int np1, const int gp1id_start,
			       float * const xp2, float * const yp2, float * const zp2,
			       float * const xv2, float * const yv2, float * const zv2,
			       float * const xa2, float * const ya2, float * const za2,
			       const int np2, const int gp2id_start,
			       const float rc, const float LX, const float LY, const float LZ,
			       const float a, const float gamma, const float sigma, const float invsqrtdt)
{
    float * pv1 = new float[6 * np1];
    float * pv2 = new float[6 * np2];

    for(int i = 0; i < np1; ++i)
    {
	pv1[0 + 6 * i] = xp1[i];
	pv1[1 + 6 * i] = yp1[i];
	pv1[2 + 6 * i] = zp1[i];
	pv1[3 + 6 * i] = xv1[i];
	pv1[4 + 6 * i] = yv1[i];
	pv1[5 + 6 * i] = zv1[i];
    }
    
    for(int i = 0; i < np2; ++i)
    {
	pv2[0 + 6 * i] = xp2[i];
	pv2[1 + 6 * i] = yp2[i];
	pv2[2 + 6 * i] = zp2[i];
	pv2[3 + 6 * i] = xv2[i];
	pv2[4 + 6 * i] = yv2[i];
	pv2[5 + 6 * i] = zv2[i];
    }

    float * a1 = new float[3 * np1];
    float * a2 = new float[3 * np2];
    
    memset(a1, 0, sizeof(float) * 3 * np1);
    memset(a2, 0, sizeof(float) * 3 * np2);

    int * order1 = new int [np1];
    int * order2 = new int [np2];

    forces_dpd_cuda_bipartite(pv1, a1, order1, np1, gp1id_start,
		    pv2, a2, order2, np2, gp2id_start,
		    rc, LX, LY, LZ,
		    a, gamma, sigma, invsqrtdt);
     
    for(int i = 0; i < np1; ++i)
    {
	assert(xp1[order1[i]] == pv1[0 + 6 * i]);
	
	xa1[order1[i]] += a1[0 + 3 * i];
	ya1[order1[i]] += a1[1 + 3 * i]; 
	za1[order1[i]] += a1[2 + 3 * i];
    }
    
    for(int i = 0; i < np2; ++i)
    {
	xa2[order2[i]] += a2[0 + 3 * i];
	ya2[order2[i]] += a2[1 + 3 * i];
	za2[order2[i]] += a2[2 + 3 * i];
    }
   
    delete [] a1;
    delete [] a2;

    delete [] order1;
    delete [] order2;
}