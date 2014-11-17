#include <cstdio>
#include <cassert>

#include "../saru.cuh"

#ifndef NDEBUG
//#define _CHECK_
#endif

struct InfoDPD
{
    int3 ncells;
    float3 domainsize, invdomainsize, domainstart;
    float invrc, aij, gamma, sigmaf;
};

__constant__ InfoDPD info;

texture<float2, cudaTextureType1D> texParticles2;
texture<int, cudaTextureType1D> texStart, texCount;
 
#define COLS 32
#define ROWS (32 / COLS)
#define _XCPB_ 2
#define _YCPB_ 2
#define _ZCPB_ 1
#define CPB (_XCPB_ * _YCPB_ * _ZCPB_)

__global__ __launch_bounds__(32 * CPB, 16) 
    void _dpd_forces_saru(float * const axayaz,
			  const int idtimestep)
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
	mycount = tex1Dfetch(texCount, cid);
    }

    for(int L = 1; L < 32; L <<= 1)
	mycount += (tid >= L) * __shfl_up(mycount, L) ;

    if (tid < 27)
	scan[wid][tid] = mycount;

    const int dststart = starts[wid][0];
    const int nsrc = scan[wid][26], ndst = scan[wid][0];
 
    for(int d = 0; d < ndst; d += ROWS)
    {
	const int np1 = min(ndst - d, ROWS);

	const int dpid = dststart + d + slot;
	const int entry = 3 * dpid;
	float2 dtmp0 = tex1Dfetch(texParticles2, entry);
	float2 dtmp1 = tex1Dfetch(texParticles2, entry + 1);
	float2 dtmp2 = tex1Dfetch(texParticles2, entry + 2);
	
	float xforce = 0, yforce = 0, zforce = 0;

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
	    const int sentry = 3 * spid;
	    const float2 stmp0 = tex1Dfetch(texParticles2, sentry);
	    const float2 stmp1 = tex1Dfetch(texParticles2, sentry + 1);
	    const float2 stmp2 = tex1Dfetch(texParticles2, sentry + 2);
	    
	    {
		const float xdiff = dtmp0.x - stmp0.x;
		const float ydiff = dtmp0.y - stmp0.y;
		const float zdiff = dtmp1.x - stmp1.x;

#ifndef _NONPERIODIC_KERNEL_
		asdasda
		const float _xr = xdiff - info.domainsize.x * floorf(0.5f + xdiff * info.invdomainsize.x);
		const float _yr = ydiff - info.domainsize.y * floorf(0.5f + ydiff * info.invdomainsize.y);
		const float _zr = zdiff - info.domainsize.z * floorf(0.5f + zdiff * info.invdomainsize.z);
#else
		const float _xr = xdiff;
		const float _yr = ydiff;
		const float _zr = zdiff;
#endif
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
		  
		const float mysaru = saru(min(spid, dpid), max(spid, dpid), idtimestep);
		const float myrandnr = 3.464101615f * mysaru - 1.732050807f;
		 
		const float strength = (info.aij - info.gamma * wr * rdotv + info.sigmaf * myrandnr) * wr;
		const bool valid = (d + slot != s + subtid) && (slot < np1) && (subtid < np2);
		
		if (valid)
		{
#ifdef _CHECK_
		    xforce += (rij2 < 1);
		    yforce += wr;
		    zforce += 0;
#else		    	     
		    xforce += strength * xr;
		    yforce += strength * yr;
		    zforce += strength * zr;
#endif
		}
	    } 
	}
	
	for(int L = COLS / 2; L > 0; L >>=1)
	{
	    xforce += __shfl_xor(xforce, L);
	    yforce += __shfl_xor(yforce, L);
	    zforce += __shfl_xor(zforce, L);
	}

	const int c = (subtid % 3);       
	const float fcontrib = (c == 0) * xforce + (c == 1) * yforce + (c == 2) * zforce;//f[subtid % 3];
	const int dstpid = dststart + d + slot;


	if (slot < np1)
	    axayaz[c + 3 * dstpid] = fcontrib;
    }
}

bool fdpd_init = false;

#include "../hacks.h"

void forces_dpd_cuda_nohost(const float * const xyzuvw, float * const axayaz,  const int np,
			    const int * const cellsstart, const int * const cellscount, 
			    const float rc,
			    const float XL, const float YL, const float ZL,
			    const float aij,
			    const float gamma,
			    const float sigma,
			    const float invsqrtdt,
			    const int saru_tag)
{
    if (np == 0)
    {
	printf("WARNING: forces_dpd_cuda_nohost called with np = %d\n", np);
	return;
    }
    
    int nx = (int)ceil(XL / rc);
    int ny = (int)ceil(YL / rc);
    int nz = (int)ceil(ZL / rc);
    const int ncells = nx * ny * nz;

    if (!fdpd_init)
    {
	texStart.channelDesc = cudaCreateChannelDesc<int>();
	texStart.filterMode = cudaFilterModePoint;
	texStart.mipmapFilterMode = cudaFilterModePoint;
	texStart.normalized = 0;
    
	texCount.channelDesc = cudaCreateChannelDesc<int>();
	texCount.filterMode = cudaFilterModePoint;
	texCount.mipmapFilterMode = cudaFilterModePoint;
	texCount.normalized = 0;

	texParticles2.channelDesc = cudaCreateChannelDesc<float2>();
	texParticles2.filterMode = cudaFilterModePoint;
	texParticles2.mipmapFilterMode = cudaFilterModePoint;
	texParticles2.normalized = 0;

	fdpd_init = true;
    }

    size_t textureoffset;
    CUDA_CHECK(cudaBindTexture(&textureoffset, &texParticles2, xyzuvw, &texParticles2.channelDesc, sizeof(float) * 6 * np));
    assert(textureoffset == 0);
    CUDA_CHECK(cudaBindTexture(&textureoffset, &texStart, cellsstart, &texStart.channelDesc, sizeof(int) * ncells));
    assert(textureoffset == 0);
    CUDA_CHECK(cudaBindTexture(&textureoffset, &texCount, cellscount, &texCount.channelDesc, sizeof(int) * ncells));
    assert(textureoffset == 0);
      
    InfoDPD c;
    c.ncells = make_int3(nx, ny, nz);
    c.domainsize = make_float3(XL, YL, ZL);
    c.invdomainsize = make_float3(1 / XL, 1 / YL, 1 / ZL);
    c.domainstart = make_float3(-XL * 0.5, -YL * 0.5, -ZL * 0.5);
    c.invrc = 1.f / rc;
    c.aij = aij;
    c.gamma = gamma;
    c.sigmaf = sigma * invsqrtdt;
      
    CUDA_CHECK(cudaMemcpyToSymbolAsync(info, &c, sizeof(c)));
   
    _dpd_forces_saru<<<dim3(c.ncells.x / _XCPB_,
			    c.ncells.y / _YCPB_,
			    c.ncells.z / _ZCPB_), dim3(32, CPB)>>>(axayaz, saru_tag);

    CUDA_CHECK(cudaPeekAtLastError());	
}

#include <cmath>
#include <unistd.h>

//#include <thrust/device_vector.h>
//using namespace thrust;

#include "../profiler-dpd.h"
#include "../cell-lists.h"




int fdpd_oldnp = 0, fdpd_oldnc = 0;

float * fdpd_xyzuvw = NULL, * fdpd_axayaz = NULL;
int * fdpd_start = NULL, * fdpd_count = NULL;

void forces_dpd_cuda_aos(float * const _xyzuvw, float * const _axayaz,
		     int * const order, const int np,
		     const float rc,
		     const float XL, const float YL, const float ZL,
		     const float aij,
		     const float gamma,
		     const float sigma,
		     const float invsqrtdt,
			 const int saru_tag,
			 const bool nohost)
{
    if (np == 0)
    {
	printf("WARNING: forces_dpd_cuda_aos called with np = %d\n", np);
	return;
    }
    
    int nx = (int)ceil(XL / rc);
    int ny = (int)ceil(YL / rc);
    int nz = (int)ceil(ZL / rc);
    const int ncells = nx * ny * nz;

    if (!fdpd_init)
    {
	texStart.channelDesc = cudaCreateChannelDesc<int>();
	texStart.filterMode = cudaFilterModePoint;
	texStart.mipmapFilterMode = cudaFilterModePoint;
	texStart.normalized = 0;
    
	texCount.channelDesc = cudaCreateChannelDesc<int>();
	texCount.filterMode = cudaFilterModePoint;
	texCount.mipmapFilterMode = cudaFilterModePoint;
	texCount.normalized = 0;

	texParticles2.channelDesc = cudaCreateChannelDesc<float2>();
	texParticles2.filterMode = cudaFilterModePoint;
	texParticles2.mipmapFilterMode = cudaFilterModePoint;
	texParticles2.normalized = 0;

	fdpd_init = true;
    }
    
    if (fdpd_oldnp < np)
    {
	if (fdpd_oldnp > 0)
	{
	    CUDA_CHECK(cudaFree(fdpd_xyzuvw));
	    CUDA_CHECK(cudaFree(fdpd_axayaz));
	}

	CUDA_CHECK(cudaMalloc(&fdpd_xyzuvw, sizeof(float) * 6 * np));
	CUDA_CHECK(cudaMalloc(&fdpd_axayaz, sizeof(float) * 3 * np));

	size_t textureoffset;
	CUDA_CHECK(cudaBindTexture(&textureoffset, &texParticles2, fdpd_xyzuvw, &texParticles2.channelDesc, sizeof(float) * 6 * np));
	
	fdpd_oldnp = np;
    }

    if (fdpd_oldnc < ncells)
    {
	if (fdpd_oldnc > 0)
	{
	    CUDA_CHECK(cudaFree(fdpd_start));
	    CUDA_CHECK(cudaFree(fdpd_count));
	}

	CUDA_CHECK(cudaMalloc(&fdpd_start, sizeof(int) * ncells));
	CUDA_CHECK(cudaMalloc(&fdpd_count, sizeof(int) * ncells));

	size_t textureoffset = 0;
	CUDA_CHECK(cudaBindTexture(&textureoffset, &texStart, fdpd_start, &texStart.channelDesc, sizeof(int) * ncells));
	CUDA_CHECK(cudaBindTexture(&textureoffset, &texCount, fdpd_count, &texCount.channelDesc, sizeof(int) * ncells));
	
	fdpd_oldnc = ncells;
    }

    CUDA_CHECK(cudaMemcpyAsync(fdpd_xyzuvw, _xyzuvw, sizeof(float) * np * 6, nohost ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice));
    
    InfoDPD c;
    c.ncells = make_int3(nx, ny, nz);
    c.domainsize = make_float3(XL, YL, ZL);
    c.invdomainsize = make_float3(1 / XL, 1 / YL, 1 / ZL);
    c.domainstart = make_float3(-XL * 0.5, -YL * 0.5, -ZL * 0.5);
    c.invrc = 1.f / rc;
    c.aij = aij;
    c.gamma = gamma;
    c.sigmaf = sigma * invsqrtdt;
        
    build_clists(fdpd_xyzuvw, np, rc, c.ncells.x, c.ncells.y, c.ncells.z,
		 c.domainstart.x, c.domainstart.y, c.domainstart.z,
		 order, fdpd_start, fdpd_count, NULL);

    //TextureWrap texStart(_ptr(starts), ncells), texCount(_ptr(counts), ncells);
    //TextureWrap texParticles((float2*)_ptr(xyzuvw), 3 * np);
    
    CUDA_CHECK(cudaMemcpyToSymbolAsync(info, &c, sizeof(c)));
   
    ProfilerDPD::singletone().start();

    if (saru_tag >= 0)
	saru_tid = saru_tag;
    
    _dpd_forces_saru<<<dim3(c.ncells.x / _XCPB_,
			    c.ncells.y / _YCPB_,
			    c.ncells.z / _ZCPB_), dim3(32, CPB)>>>(fdpd_axayaz, saru_tid);
 
    ++saru_tid;

    CUDA_CHECK(cudaPeekAtLastError());
	
    ProfilerDPD::singletone().force();
    
//copy xyzuvw as well?!?
    if (nohost)
    {
	CUDA_CHECK(cudaMemcpy(_xyzuvw, fdpd_xyzuvw, sizeof(float) * 6 * np, cudaMemcpyDeviceToDevice));
	CUDA_CHECK(cudaMemcpy(_axayaz, fdpd_axayaz, sizeof(float) * 3 * np, cudaMemcpyDeviceToDevice));
    }
    else
	CUDA_CHECK(cudaMemcpy(_axayaz, fdpd_axayaz, sizeof(float) * 3 * np, cudaMemcpyDeviceToHost));

    ProfilerDPD::singletone().report();

    //copy(axayaz.begin(), axayaz.end(), _axayaz);
     
#ifdef _CHECK_
    CUDA_CHECK(cudaThreadSynchronize());
    
    for(int ii = 0; ii < np; ++ii)
    { 
	printf("pid %d -> %f %f %f\n", ii, (float)axayaz[0 + 3 * ii], (float)axayaz[1 + 3* ii], (float)axayaz[2 + 3 *ii]);

	int cnt = 0;
	float fc = 0;
	const int i = order[ii];
	printf("devi coords are %f %f %f\n", (float)xyzuvw[0 + 6 * ii], (float)xyzuvw[1 + 6 * ii], (float)xyzuvw[2 + 6 * ii]);
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
	printf("i found %d host interactions and with cuda i found %d\n", cnt, (int)axayaz[0 + 3 * ii]);
	assert(cnt == (float)axayaz[0 + 3 * ii]);
	printf("fc aij ref %f vs cuda %e\n", fc,  (float)axayaz[1 + 3 * ii]);
	assert(fabs(fc - (float)axayaz[1 + 3 * ii]) < 1e-4);
    }
    
    printf("test done.\n");
    sleep(1);
    exit(0);
#endif
}

int * fdpd_order = NULL;
float * fdpd_pv = NULL, *fdpd_a = NULL;

void forces_dpd_cuda(const float * const xp, const float * const yp, const float * const zp,
		     const float * const xv, const float * const yv, const float * const zv,
		     float * const xa, float * const ya, float * const za,
		     const int np,
		     const float rc,
		     const float LX, const float LY, const float LZ,
		     const float aij,
		     const float gamma,
		     const float sigma,
		     const float invsqrtdt,
		     const int input_saru_tag)
{
    if (np <= 0) return;

    if (np > fdpd_oldnp)
    {
	if (fdpd_oldnp > 0)
	{
	    CUDA_CHECK(cudaFreeHost(fdpd_pv));
	    CUDA_CHECK(cudaFreeHost(fdpd_order));
	    CUDA_CHECK(cudaFreeHost(fdpd_a));
	}

	CUDA_CHECK(cudaHostAlloc(&fdpd_pv, sizeof(float) * np * 6, cudaHostAllocDefault));
	CUDA_CHECK(cudaHostAlloc(&fdpd_order, sizeof(int) * np, cudaHostAllocDefault));
	CUDA_CHECK(cudaHostAlloc(&fdpd_a, sizeof(float) * np * 3, cudaHostAllocDefault));

	//this will be done by forces_dpd_cuda
	//fdpd_oldnp = np;
    }
    
    for(int i = 0; i < np; ++i)
    {
	fdpd_pv[0 + 6 * i] = xp[i];
	fdpd_pv[1 + 6 * i] = yp[i];
	fdpd_pv[2 + 6 * i] = zp[i];
	fdpd_pv[3 + 6 * i] = xv[i];
	fdpd_pv[4 + 6 * i] = yv[i];
	fdpd_pv[5 + 6 * i] = zv[i];
    }

    forces_dpd_cuda_aos(fdpd_pv, fdpd_a, fdpd_order, np, rc, LX, LY, LZ,
			aij, gamma, sigma, invsqrtdt, input_saru_tag, false);
    
    //delete [] pv;
     
    for(int i = 0; i < np; ++i)
    {
	xa[fdpd_order[i]] += fdpd_a[0 + 3 * i];
	ya[fdpd_order[i]] += fdpd_a[1 + 3 * i];
	za[fdpd_order[i]] += fdpd_a[2 + 3 * i];
    }

    //delete [] a;

    //delete [] order;
}