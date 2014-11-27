#include <cassert>

#include "../saru.cuh"

struct BipartiteInfoDPD
{
    int3 ncells;
    float3 domainsize, invdomainsize, domainstart;
    float invrc, aij, gamma, sigmaf;
};

__constant__ BipartiteInfoDPD bipart_info;


#ifndef NDEBUG
//#define _CHECK_
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
    
    const int xmycid = mycid % bipart_info.ncells.x;
    const int ymycid = (mycid / bipart_info.ncells.x) % bipart_info.ncells.y;
    const int zmycid = (mycid / bipart_info.ncells.x / bipart_info.ncells.y) % bipart_info.ncells.z;

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

	const int xcid = (xmycid + dx - 1 + bipart_info.ncells.x) % bipart_info.ncells.x;
	const int ycid = (ymycid + dy - 1 + bipart_info.ncells.y) % bipart_info.ncells.y;
	const int zcid = (zmycid + dz - 1 + bipart_info.ncells.z) % bipart_info.ncells.z;
	const int cid = xcid + bipart_info.ncells.x * (ycid + bipart_info.ncells.y * zcid);

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

		const float _xr = xdiff - bipart_info.domainsize.x * floorf(0.5f + xdiff * bipart_info.invdomainsize.x);
		const float _yr = ydiff - bipart_info.domainsize.y * floorf(0.5f + ydiff * bipart_info.invdomainsize.y);
		const float _zr = zdiff - bipart_info.domainsize.z * floorf(0.5f + zdiff * bipart_info.invdomainsize.z);

		const float rij2 = _xr * _xr + _yr * _yr + _zr * _zr;
		const float invrij = rsqrtf(rij2);
		const float rij = rij2 * invrij;
		const float wr = max((float)0, 1 - rij * bipart_info.invrc);
		
		const float xr = _xr * invrij;
		const float yr = _yr * invrij;
		const float zr = _zr * invrij;
		
		const float rdotv = 
		    xr * (dtmp1.y - stmp1.y) +
		    yr * (dtmp2.x - stmp2.x) +
		    zr * (dtmp2.y - stmp2.y);
	
		const float mysaru = saru(min(gspid, gdpid), max(gspid, gdpid), idtimestep);
		const float myrandnr = 3.464101615f * mysaru - 1.732050807f;
		 
		const float strength = (bipart_info.aij - bipart_info.gamma * wr * rdotv + bipart_info.sigmaf * myrandnr) * wr;
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
#include "../hacks.h"

using namespace thrust;

void forces_dpd_cuda_bipartite(float * const _xyzuvw1, float * const _axayaz1, int * const order1, const int np1, const int gp1id_start,
			       float * const _xyzuvw2, float * const _axayaz2, int * const order2, const int np2, const int gp2id_start,
			       const float rc, const float XL, const float YL, const float ZL,
			       const float aij, const float gamma, const float sigma, const float invsqrtdt)
{
    const bool second_partition = _axayaz2 != NULL;
    
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
    
    BipartiteInfoDPD c;
    c.ncells = make_int3(nx, ny, nz);
    c.domainsize = make_float3(XL, YL, ZL);
    c.invdomainsize = make_float3(1 / XL, 1 / YL, 1 / ZL);
    c.domainstart = make_float3(-XL * 0.5, -YL * 0.5, -ZL * 0.5);
    c.invrc = 1.f / rc;
    c.aij = aij;
    c.gamma = gamma;
    c.sigmaf = sigma * invsqrtdt;
        
    CUDA_CHECK(cudaMemcpyToSymbol(bipart_info, &c, sizeof(c)));

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
    if (second_partition)
	CUDA_CHECK(cudaStreamCreate(&stream2));
    
    ProfilerDPD::singletone().start();

    _dpd_forces_saru<<<(nonemptycells1.first + CPB - 1) / CPB, dim3(32, CPB), 0, stream1>>>(
	_ptr(axayaz1), saru_tid, texStart1.texObj, texCount1.texObj, texStart2.texObj, texCount2.texObj,
	texParticles1.texObj, texParticles2.texObj, nonemptycells1.second, nonemptycells1.first, gp1id_start, gp2id_start);
    
    CUDA_CHECK(cudaPeekAtLastError());
    
    if (second_partition)
    {
	_dpd_forces_saru<<<(nonemptycells2.first + CPB - 1) / CPB, dim3(32, CPB), 0, stream2>>>(
	    _ptr(axayaz2), saru_tid, texStart2.texObj, texCount2.texObj, texStart1.texObj, texCount1.texObj,
	    texParticles2.texObj, texParticles1.texObj, nonemptycells2.second, nonemptycells2.first, gp2id_start, gp1id_start);
    
	CUDA_CHECK(cudaPeekAtLastError());
    }
    saru_tid += 1;  

    ProfilerDPD::singletone().force();	
    ProfilerDPD::singletone().report();
 
    copy(axayaz1.begin(), axayaz1.end(), _axayaz1);

    if (second_partition)
	copy(axayaz2.begin(), axayaz2.end(), _axayaz2);

    CUDA_CHECK(cudaStreamDestroy(stream1));
    
    if (second_partition)
	CUDA_CHECK(cudaStreamDestroy(stream2));
    
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



__global__
//template<int ILP>
void _bipartite_dpd_directforces(float * const axayaz, const int np, const int np_src,
				 const int saru_tag1, const int saru_tag2, const bool saru_mask, const float * xyzuvw, const float * xyzuvw_src,
				 const float invrc, const float aij, const float gamma, const float sigmaf)
{
    assert(blockDim.x % warpSize == 0);
    assert(blockDim.x * gridDim.x >= np);
    
    const int tid = threadIdx.x % warpSize;
    const int pid = threadIdx.x + blockDim.x * blockIdx.x;
    const bool valid = pid < np;

    float xp, yp, zp, up, vp, wp;

    if (valid)
    {
	xp = xyzuvw[0 + pid * 6];
	yp = xyzuvw[1 + pid * 6];
	zp = xyzuvw[2 + pid * 6];
	up = xyzuvw[3 + pid * 6];
	vp = xyzuvw[4 + pid * 6];
	wp = xyzuvw[5 + pid * 6];
    }

    float xforce = 0, yforce = 0, zforce = 0;
    
    for(int s = 0; s < np_src; s += warpSize)
    {
	float my_xq, my_yq, my_zq, my_uq, my_vq, my_wq;

	const int batchsize = min(warpSize, np_src - s);

	if (tid < batchsize)
	{
	    my_xq = xyzuvw_src[0 + (tid + s) * 6];
	    my_yq = xyzuvw_src[1 + (tid + s) * 6];
	    my_zq = xyzuvw_src[2 + (tid + s) * 6];
	    my_uq = xyzuvw_src[3 + (tid + s) * 6];
	    my_vq = xyzuvw_src[4 + (tid + s) * 6];
	    my_wq = xyzuvw_src[5 + (tid + s) * 6];
	}
	
	for(int l = 0; l < batchsize; ++l)
	{
	    const float xq = __shfl(my_xq, l);
	    const float yq = __shfl(my_yq, l);
	    const float zq = __shfl(my_zq, l);
	    const float uq = __shfl(my_uq, l);
	    const float vq = __shfl(my_vq, l);
	    const float wq = __shfl(my_wq, l);

	    //necessary to force the execution shuffles here below
	    //__syncthreads();
	    
	    //if (valid)
	    {
		const float _xr = xp - xq;
		const float _yr = yp - yq;
		const float _zr = zp - zq;
		
		const float rij2 = _xr * _xr + _yr * _yr + _zr * _zr;
		
		const float invrij = rsqrtf(rij2);
		 
		const float rij = rij2 * invrij;
		const float wr = max((float)0, 1 - rij * invrc);
		
		const float xr = _xr * invrij;
		const float yr = _yr * invrij;
		const float zr = _zr * invrij;

		const float rdotv = 
		    xr * (up - uq) +
		    yr * (vp - vq) +
		    zr * (wp - wq);
		
		const float mysaru = saru(saru_tag1, saru_tag2, saru_mask ? pid + np * (s + l) : (s + l) + np_src * pid);
	
		const float myrandnr = 3.464101615f * mysaru - 1.732050807f;
		 
		const float strength = (aij - gamma * wr * rdotv + sigmaf * myrandnr) * wr;

		xforce += strength * xr;
		yforce += strength * yr;
		zforce += strength * zr;

		/*	assert(fabs(xforce) < 100);
			assert(fabs(yforce) < 100);		
			assert(fabs(zforce) < 100);*/
	    }
	}
    }

    if (valid)
    {
	assert(!isnan(xforce));
	assert(!isnan(yforce));
	assert(!isnan(zforce));
    
	axayaz[0 + 3 * pid] = xforce;
	axayaz[1 + 3 * pid] = yforce;
	axayaz[2 + 3 * pid] = zforce;
/*
  for(int c = 0; c < 3; ++c)
  {
  assert(fabs(axayaz[c + 3 * pid]) < 100);
  }*/
    }
}

void directforces_dpd_cuda_bipartite_nohost(
    const float * const xyzuvw, float * const axayaz, const int np,
    const float * const xyzuvw_src, const int np_src,
    const float aij, const float gamma, const float sigma, const float invsqrtdt,
    const int saru_tag1, const int saru_tag2, const bool sarumask, cudaStream_t stream)
{
    if (np == 0 || np_src == 0)
    {
	printf("warning: directforces_dpd_cuda_bipartite_nohost called with ZERO!\n");
	return;
    }
 
    _bipartite_dpd_directforces<<<(np + 127) / 128, 128, 0, stream>>>(axayaz, np, np_src, saru_tag1, saru_tag2, sarumask,
								      xyzuvw, xyzuvw_src, 1, aij, gamma, sigma * invsqrtdt);
   
    CUDA_CHECK(cudaPeekAtLastError());
}

void directforces_dpd_cuda_bipartite(
    const float * const xyzuvw, float * const axayaz, const int np,
    const float * const xyzuvw_src, const int np_src,
    const float aij, const float gamma, const float sigma, const float invsqrtdt,
    const int saru_tag1, const int saru_tag2, const bool sarumask)
{
    float * xyzuvw_d = NULL;
    CUDA_CHECK(cudaMalloc(&xyzuvw_d, sizeof(float) * 6 * np));
    CUDA_CHECK(cudaMemcpy(xyzuvw_d, xyzuvw, sizeof(float) * 6 * np, cudaMemcpyHostToDevice));

    float * xyzuvw_src_d = NULL;
    CUDA_CHECK(cudaMalloc(&xyzuvw_src_d, sizeof(float) * 6 * np_src));
    CUDA_CHECK(cudaMemcpy(xyzuvw_src_d, xyzuvw_src, sizeof(float) * 6 * np_src, cudaMemcpyHostToDevice));

    float * axayaz_d = NULL;
    CUDA_CHECK(cudaMalloc(&axayaz_d, sizeof(float) * 3 * np));
    CUDA_CHECK(cudaMemset(axayaz_d, 0, sizeof(float) * 3 * np));
    
    directforces_dpd_cuda_bipartite_nohost(xyzuvw_d, axayaz_d, np, xyzuvw_src_d,  np_src,
					   aij,  gamma,  sigma, invsqrtdt, saru_tag1, saru_tag2, sarumask, 0);

    CUDA_CHECK( cudaMemcpy(axayaz, axayaz_d, sizeof(float) * 3 * np, cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(xyzuvw_d));
    CUDA_CHECK(cudaFree(xyzuvw_src_d));
    CUDA_CHECK(cudaFree(axayaz_d));
}

void forces_dpd_cuda_bipartite(const float * const xp1, const float * const yp1, const float * const zp1,
			       const float * const xv1, const float * const yv1, const float * const zv1,
			       float * const xa1, float * const ya1,  float * const za1,
			       const int np1, const int gp1id_start,	 
			       const float * const xp2, const float * const yp2, const float * const zp2,
			       const float * const xv2, const float * const yv2, const float * const zv2,
			       float * const xa2,  float * const ya2,  float * const za2,
			       const int np2, const int gp2id_start,
			       const float rc, const float LX, const float LY, const float LZ,
			       const float a, const float gamma, const float sigma, const float invsqrtdt)
{
    const bool second_partition = xa2 != NULL && ya2 != NULL && za2 != NULL;
    
    if (np1 * np2 <= 0) return;

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
    float * a2 = NULL;

    if (second_partition)
	a2 = new float[3 * np2];
    
    memset(a1, 0, sizeof(float) * 3 * np1);
    if (second_partition)
	memset(a2, 0, sizeof(float) * 3 * np2);

    int * order1 = new int [np1];
    int * order2 = NULL;

    if (second_partition)
	order2 = new int [np2];

    forces_dpd_cuda_bipartite(pv1, a1, order1, np1, gp1id_start,
			      pv2, a2, order2, np2, gp2id_start,
			      rc, LX, LY, LZ,
			      a, gamma, sigma, invsqrtdt);

    delete [] pv1;
    delete [] pv2;
     
    for(int i = 0; i < np1; ++i)
    {
	assert(xp1[order1[i]] == pv1[0 + 6 * i]);
	
	xa1[order1[i]] += a1[0 + 3 * i];
	ya1[order1[i]] += a1[1 + 3 * i]; 
	za1[order1[i]] += a1[2 + 3 * i];
    }

    if (second_partition)
	for(int i = 0; i < np2; ++i)
	{
	    xa2[order2[i]] += a2[0 + 3 * i];
	    ya2[order2[i]] += a2[1 + 3 * i];
	    za2[order2[i]] += a2[2 + 3 * i];
	}
   
    delete [] a1;

    if (second_partition)
	delete [] a2;

    delete [] order1;

    if (second_partition)
	delete [] order2;
}

__global__ __launch_bounds__(32 * CPB, 16) 
    void _dpd_forces_saru(const float2 * const xyzuvw, const int np, cudaTextureObject_t texDstStart,
			  cudaTextureObject_t texSrcStart,  cudaTextureObject_t texSrcParticles, const int np_src, const int3 halo_ncells,
			  const float aij, const float gamma, const float sigmaf,
			  const int saru_tag1, const int saru_tag2, const bool sarumask, float * const axayaz)
{
    assert(warpSize == COLS * ROWS);
    assert(blockDim.x == warpSize && blockDim.y == CPB && blockDim.z == 1);
    assert(ROWS * 3 <= warpSize);

    const int mycid = blockIdx.x * CPB + threadIdx.y;

    if (mycid >= halo_ncells.x * halo_ncells.y * halo_ncells.z)
	return;

    const int xmycid = mycid % halo_ncells.x;
    const int ymycid = (mycid / halo_ncells.x) % halo_ncells.y;
    const int zmycid = (mycid / halo_ncells.x / halo_ncells.y) % halo_ncells.z;

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

	const int xcid = xmycid + dx - 1;
	const int ycid = ymycid + dy - 1;
	const int zcid = zmycid + dz - 1;
	
	const bool bad_cid =
	    xcid < 0 || xcid >= halo_ncells.x ||
	    ycid < 0 || ycid >= halo_ncells.y ||
	    zcid < 0 || zcid >= halo_ncells.z ;
	    
	const int cid = xcid + halo_ncells.x * (ycid + halo_ncells.y * zcid);

	starts[wid][tid] = bad_cid ? -10000 : tex1Dfetch<int>(texSrcStart, cid);
	mycount = bad_cid ? 0 : (tex1Dfetch<int>(texSrcStart, cid + 1) - tex1Dfetch<int>(texSrcStart, cid));
    }

    for(int L = 1; L < 32; L <<= 1)
	mycount += (tid >= L) * __shfl_up(mycount, L) ;

    if (tid < 27)
	scan[wid][tid] = mycount;

    const int dststart = tex1Dfetch<int>(texDstStart, mycid);
    const int nsrc = scan[wid][26], ndst = tex1Dfetch<int>(texDstStart, mycid + 1) - tex1Dfetch<int>(texDstStart, mycid);
    
    for(int d = 0; d < ndst; d += ROWS)
    {
	const int np1 = min(ndst - d, ROWS);

	const int dpid = dststart + d + slot;

	const int entry = 3 * dpid;
	float2 dtmp0 = xyzuvw[entry];
	float2 dtmp1 = xyzuvw[entry + 1];
	float2 dtmp2 = xyzuvw[entry + 2];
	
	float f[3] = {0, 0, 0};

	for(int s = 0; s < nsrc; s += COLS)
	{
	    const int np2 = min(nsrc - s, COLS);
  
	    const int pid = s + subtid;
	    const int key9 = 9 * (pid >= scan[wid][8]) + 9 * (pid >= scan[wid][17]);
	    const int key3 = 3 * (pid >= scan[wid][key9 + 2]) + 3 * (pid >= scan[wid][key9 + 5]);
	    const int key1 = (pid >= scan[wid][key9 + key3]) + (pid >= scan[wid][key9 + key3 + 1]);
	    const int key = key9 + key3 + key1;
	    assert(key >= 0 && key < 27);
	    assert(subtid >= np2 || pid >= (key ? scan[wid][key - 1] : 0) && pid < scan[wid][key]);

	    const int spid = starts[wid][key] + pid - (key ? scan[wid][key - 1] : 0);
	    assert(subtid >= np2 || starts[wid][key] >= 0);
	    
	    const int sentry = 3 * spid;
	    const float2 stmp0 = tex1Dfetch<float2>(texSrcParticles, sentry);
	    const float2 stmp1 = tex1Dfetch<float2>(texSrcParticles, sentry + 1);
	    const float2 stmp2 = tex1Dfetch<float2>(texSrcParticles, sentry + 2);
	    
	    {
		const float xforce = f[0];
		const float yforce = f[1];
		const float zforce = f[2];
			    
		const float _xr = dtmp0.x - stmp0.x;
		const float _yr = dtmp0.y - stmp0.y;
		const float _zr = dtmp1.x - stmp1.x;

		const float rij2 = _xr * _xr + _yr * _yr + _zr * _zr;
		const float invrij = rsqrtf(rij2);
		const float rij = rij2 * invrij;
		const float wr = max((float)0, 1 - rij);
		
		const float xr = _xr * invrij;
		const float yr = _yr * invrij;
		const float zr = _zr * invrij;
		
		const float rdotv = 
		    xr * (dtmp1.y - stmp1.y) +
		    yr * (dtmp2.x - stmp2.x) +
		    zr * (dtmp2.y - stmp2.y);
	
		const float mysaru = saru(saru_tag1, saru_tag2, sarumask ? dpid + np * spid : spid + np_src * dpid);
		const float myrandnr = 3.464101615f * mysaru - 1.732050807f;
		 
		const float strength = (aij - gamma * wr * rdotv + sigmaf * myrandnr) * wr;
		const bool valid = (slot < np1) && (subtid < np2);

		assert( (dpid >= 0 && dpid < np && spid >= 0 && spid < np_src) || ! valid); 
		
		if (valid)
		{
		    f[0] = xforce + strength * xr;
		    f[1] = yforce + strength * yr;
		    f[2] = zforce + strength * zr;
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

void directforces_dpd_cuda_bipartite_nohost(cudaStream_t stream, const float2 * const xyzuvw, const int np, cudaTextureObject_t texDstStart,
					    cudaTextureObject_t texSrcStart, cudaTextureObject_t texSrcParticles, const int np_src,
					    const int3 halo_ncells,
					    const float aij, const float gamma, const float sigmaf,
					    const int saru_tag1, const int saru_tag2, const bool sarumask, float * const axayaz)
{ 
    const int ncells = halo_ncells.x * halo_ncells.y * halo_ncells.z;
    
    _dpd_forces_saru<<<(ncells + CPB - 1) / CPB, dim3(32, CPB), 0, stream>>>(
	xyzuvw, np, texDstStart, texSrcStart, texSrcParticles, np_src,
	halo_ncells, aij, gamma, sigmaf, saru_tag1, saru_tag2, sarumask,
	axayaz);
}