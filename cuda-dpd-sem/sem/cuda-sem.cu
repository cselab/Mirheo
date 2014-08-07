#include <cstdio>
#include <cassert>

#include "../saru.cuh"

#ifndef NDEBUG
#define _CHECK_
#endif

struct InfoSEM
{
    int3 ncells;
    int  npart;
    float3 domainsize, invdomainsize, domainstart;
    float invrc, A0, A1, A2, gamma, B0, rc2;
};

__constant__ InfoSEM info;
 
#define COLS 8
#define ROWS (32 / COLS)
#define _XCPB_ 2
#define _YCPB_ 2
#define _ZCPB_ 1
#define CPB (_XCPB_ * _YCPB_ * _ZCPB_)

__global__ __launch_bounds__(32 * CPB, 16) 
    void _sem_forces_saru(float * const axayaz,
		       const int idtimestep, 
		       cudaTextureObject_t texStart, cudaTextureObject_t texCount, cudaTextureObject_t texParticles)
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

	starts[wid][tid] = tex1Dfetch<int>(texStart, cid);
	mycount = tex1Dfetch<int>(texCount, cid);
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
	float2 dtmp0 = tex1Dfetch<float2>(texParticles, entry);
	float2 dtmp1 = tex1Dfetch<float2>(texParticles, entry + 1);
	float2 dtmp2 = tex1Dfetch<float2>(texParticles, entry + 2);
	
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
	    const int sentry = 3 * spid;
	    const float2 stmp0 = tex1Dfetch<float2>(texParticles, sentry);
	    const float2 stmp1 = tex1Dfetch<float2>(texParticles, sentry + 1);
	    const float2 stmp2 = tex1Dfetch<float2>(texParticles, sentry + 2);
	    
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
		  
		const float mysaru = saru(min(spid, dpid), max(spid, dpid), idtimestep);
		const float myrandnr = 3.464101615f * mysaru - 1.732050807f;
#if 0
		const float e = info.A1 * (1  + rij2 * info.A2);
#else
		const float e = info.A1 * expf(rij2 * info.A2);
#endif		
		const float strength = -info.A0 * e * (1 - e) + (- info.gamma * wr * rdotv + info.B0 * myrandnr) * wr;
		const bool valid = (d + slot != s + subtid) && (slot < np1) && (subtid < np2);
		
		if (valid)
		{
#ifdef _CHECK_
		    f[0] = xforce + (rij2 < 1);
		    f[1] = yforce + wr*(rij2 < 1);
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
	    axayaz[c + 3 * dstpid] = fcontrib;
    }
}

#include <cmath>
#include <unistd.h>

#include <thrust/device_vector.h>
using namespace thrust;

#include "../profiler-dpd.h"
#include "../cell-lists.h"
#include "../hacks.h"

void forces_sem_cuda_nohost(
    float *  device_xyzuvw, float * device_axayaz,
		     int * const order, const int np,
		     const float rcutoff,
		     const float XL, const float YL, const float ZL,
		     const double gamma, const double temp, const double dt, const double u0, const double rho, const double req, const double D, const double rc)
{ 
	if (np <= 0) return;

    int nx = (int)ceil(XL / rcutoff);
    int ny = (int)ceil(YL / rcutoff);
    int nz = (int)ceil(ZL / rcutoff);
    const int ncells = nx * ny * nz;
        
    InfoSEM c;
    c.ncells = make_int3(nx, ny, nz);
    c.domainsize = make_float3(XL, YL, ZL);
    c.invdomainsize = make_float3(1 / XL, 1 / YL, 1 / ZL);
    c.domainstart = make_float3(-XL * 0.5, -YL * 0.5, -ZL * 0.5);
    c.invrc = 1.f / rc; 
    c.A0 = 4 * u0 * rho / (req * req);
    c.A1 = exp(rho);
    c.A2 = -1 / (req * req);
    c.B0 = sqrt(2 * gamma * temp / dt) * D;
    c.gamma = gamma;
    c.rc2 = rcutoff * rcutoff;
        
    CUDA_CHECK(cudaMemcpyToSymbol(info, &c, sizeof(c)));

    device_vector<int> starts(ncells), counts(ncells);
    build_clists(device_xyzuvw, np, rcutoff, c.ncells.x, c.ncells.y, c.ncells.z,
		 c.domainstart.x, c.domainstart.y, c.domainstart.z,
		 order, _ptr(starts), _ptr(counts), NULL);

    TextureWrap texStart(_ptr(starts), ncells), texCount(_ptr(counts), ncells);
    TextureWrap texParticles((float2*)device_xyzuvw, 3 * np);
    
    ProfilerDPD::singletone().start();
    
    _sem_forces_saru<<<dim3(c.ncells.x / _XCPB_,
			    c.ncells.y / _YCPB_,
			    c.ncells.z / _ZCPB_), dim3(32, CPB)>>>( device_axayaz, saru_tid, texStart.texObj, texCount.texObj, texParticles.texObj);

    ++saru_tid;

    CUDA_CHECK(cudaPeekAtLastError());
	
    ProfilerDPD::singletone().force();	
    ProfilerDPD::singletone().report();
     
#ifdef _CHECK_
    host_vector<float> axayaz(device_ptr<float>(device_axayaz), device_ptr<float>(device_axayaz + np * 3)),
	_xyzuvw(device_ptr<float>(device_xyzuvw), device_ptr<float>(device_xyzuvw + np * 6));
    
    CUDA_CHECK(cudaThreadSynchronize());
    
    for(int i = 0; i < np; ++i)
    { 
	printf("pid %d -> %f %f %f\n", i, (float)axayaz[0 + 3 * i], (float)axayaz[1 + 3* i], (float)axayaz[2 + 3 *i]);

	int cnt = 0;
	float fc = 0;
	//const int i = order[ii];
	//printf("devi coords are %f %f %f\n", (float)xyzuvw[0 + 6 * ii], (float)xyzuvw[1 + 6 * ii], (float)xyzuvw[2 + 6 * ii]);
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

void forces_sem_cuda(float * const xp, float * const yp, float * const zp,
		     float * const xv, float * const yv, float * const zv,
		     float * const xa, float * const ya, float * const za,
		     const int np,
		     const float rcutoff,
		     const float LX, const float LY, const float LZ,
		     const double gamma, const double temp, const double dt, const double u0, const double rho, const double req, const double D, const double rc)
{
	if (np <= 0) return;

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
    memset(a, 0, sizeof(float) * 3 * np);

    int * order = new int [np];

    device_vector<float> xyzuvw(pv, pv + np * 6),  axayaz(np * 3);
	
    forces_sem_cuda_nohost(_ptr(xyzuvw), _ptr(axayaz), order, np, rcutoff, LX, LY, LZ,
		    gamma, temp, dt, u0, rho, req, D, rc);

    copy(axayaz.begin(), axayaz.end(), a);
	
    delete [] pv;
     
    for(int i = 0; i < np; ++i)
    {
	xa[order[i]] += a[0 + 3 * i];
	ya[order[i]] += a[1 + 3 * i];
	za[order[i]] += a[2 + 3 * i];
    }

    delete [] a;

    delete [] order;
}

//===============================================================================================
//===============================================================================================

__inline__ __device__
float3 warpReduceSum(float3 val)
{
	for (int offset = warpSize/2; offset > 0; offset /= 2)
	{
		val.x += __shfl_down(val.x, offset);
		val.y += __shfl_down(val.y, offset);
		val.z += __shfl_down(val.z, offset);
	}
	return val;
}

__inline__ __device__
float3 blockReduceSum(float3 val)
{
  static __shared__ float3 sh[32]; // Shared mem for 32 partial sums
  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  val = warpReduceSum(val);     // Each warp performs partial reduction

  if (lane==0) sh[wid]=val;	// Write reduced value to shared memory

  __syncthreads();              // Wait for all partial reductions

  //read from shared memory only if that warp existed
  val = (threadIdx.x < (blockDim.x / warpSize)) ? sh[lane] : make_float3(0.0f, 0.0f, 0.0f);

  if (wid == 0)
	  val = warpReduceSum(val); //Final reduce within first warp

  return val;
}

__global__ void _sem_forces_saru_direct(float * const axayaz, cudaTextureObject_t texParticles, const int idtimestep)
{
	float3 a;

	int dstId = blockIdx.x;
	if (dstId >= info.npart) return;
	int tid   = threadIdx.x;

	a.x = a.y = a.z = 0;

	const float2 dtmp0 = tex1Dfetch<float2>(texParticles, 3*dstId);
	const float2 dtmp1 = tex1Dfetch<float2>(texParticles, 3*dstId + 1);
	const float2 dtmp2 = tex1Dfetch<float2>(texParticles, 3*dstId + 2);
	const float x  = dtmp0.x;
	const float y  = dtmp0.y;
	const float z  = dtmp1.x;
	const float vx = dtmp1.y;
	const float vy = dtmp2.x;
	const float vz = dtmp2.y;

	for (int srcId = tid; srcId < info.npart; srcId += blockDim.x)
		if (srcId != dstId)
		{
			const float2 stmp0 = tex1Dfetch<float2>(texParticles, 3*srcId);
			const float2 stmp1 = tex1Dfetch<float2>(texParticles, 3*srcId + 1);
			const float2 stmp2 = tex1Dfetch<float2>(texParticles, 3*srcId + 2);
			const float xn  = stmp0.x;
			const float yn  = stmp0.y;
			const float zn  = stmp1.x;
			const float vxn = stmp1.y;
			const float vyn = stmp2.x;
			const float vzn = stmp2.y;

			const float xdiff = x - xn;
			const float ydiff = y - yn;
			const float zdiff = z - zn;

			const float _xr = xdiff - info.domainsize.x * floorf(0.5f + xdiff * info.invdomainsize.x);
			const float _yr = ydiff - info.domainsize.y * floorf(0.5f + ydiff * info.invdomainsize.y);
			const float _zr = zdiff - info.domainsize.z * floorf(0.5f + zdiff * info.invdomainsize.z);

			const float rij2 = _xr * _xr + _yr * _yr + _zr * _zr;

			if (rij2 > info.rc2) continue;

			const float invrij = rsqrtf(rij2);
			const float rij = rij2 * invrij;
			const float wr = max((float)0, 1 - rij * info.invrc);

			const float xr = _xr * invrij;
			const float yr = _yr * invrij;
			const float zr = _zr * invrij;

			const float rdotv =
				xr * (vx - vxn) +
				yr * (vy - vyn) +
				zr * (vz - vzn);

			const float mysaru = saru(min(srcId, dstId), max(srcId, dstId), idtimestep);
			const float myrandnr = 3.464101615f * mysaru - 1.732050807f;
#if 0
			const float e = info.A1 * (1  + rij2 * info.A2);
#else
			const float e = info.A1 * expf(rij2 * info.A2);
#endif
			float strength = -info.A0 * e * (1 - e) + (- info.gamma * wr * rdotv + info.B0 * myrandnr) * wr;

			a.x += strength * xr;
			a.y += strength * yr;
			a.z += strength * zr;
		}

	a = blockReduceSum(a);

	if (tid == 0)
	{
		axayaz[3*dstId + 0] = a.x;
		axayaz[3*dstId + 1] = a.y;
		axayaz[3*dstId + 2] = a.z;
	}
}

#define DIRBS 128
void forces_sem_cuda_direct_nohost(
			 float *  device_xyzuvw, float * device_axayaz,
		     const int np,
		     const float rcutoff,
		     const float XL, const float YL, const float ZL,
		     const double gamma, const double temp, const double dt, const double u0, const double rho, const double req, const double D, const double rc)
{
	if (np <= 0) return;

    InfoSEM c;
    c.domainsize = make_float3(XL, YL, ZL);
    c.invdomainsize = make_float3(1 / XL, 1 / YL, 1 / ZL);
    c.domainstart = make_float3(-XL * 0.5, -YL * 0.5, -ZL * 0.5);
    c.invrc = 1.f / rc;
    c.A0 = 4 * u0 * rho / (req * req);
    c.A1 = exp(rho);
    c.A2 = -1 / (req * req);
    c.B0 = sqrt(2 * gamma * temp / dt) * D;
    c.gamma = gamma;
    c.npart = np;
    c.rc2 = rcutoff * rcutoff;

    CUDA_CHECK(cudaMemcpyToSymbol(info, &c, sizeof(c)));

    TextureWrap texParticles((float2*)device_xyzuvw, 3 * np);

    ProfilerDPD::singletone().start();

    _sem_forces_saru_direct<<<np, DIRBS>>>( device_axayaz, texParticles.texObj, saru_tid );

    ++saru_tid;

    CUDA_CHECK(cudaPeekAtLastError());

    ProfilerDPD::singletone().force();
    ProfilerDPD::singletone().report();

#ifdef _CHECK_
    host_vector<float> axayaz(device_ptr<float>(device_axayaz), device_ptr<float>(device_axayaz + np * 3));

    host_vector<float> _xyzuvw(device_ptr<float>(device_xyzuvw), device_ptr<float>(device_xyzuvw + np * 6));


    CUDA_CHECK(cudaThreadSynchronize());

    for(int i = 0; i < np; ++i)
    {
	//printf("pid %d -> %f %f %f\n", i, (float)axayaz[0 + 3 * i], (float)axayaz[1 + 3* i], (float)axayaz[2 + 3 *i]);

	int cnt = 0;
	float fc = 0;
	//const int i = order[ii];
	//printf("devi coords are %f %f %f\n", (float)xyzuvw[0 + 6 * ii], (float)xyzuvw[1 + 6 * ii], (float)xyzuvw[2 + 6 * ii]);
	//printf("host coords are %f %f %f\n", (float)_xyzuvw[0 + 6 * i], (float)_xyzuvw[1 + 6 * i], (float)_xyzuvw[2 + 6 * i]);

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

	    const float _xr = xr * invrij;
	    const float _yr = yr * invrij;
	    const float _zr = zr * invrij;

	    			const float rdotv =
	    				_xr * (_xyzuvw[3 + 6 *i] - _xyzuvw[3 + 6 * j]) +
	    				_yr * (_xyzuvw[4 + 6 *i] - _xyzuvw[4 + 6 * j]) +
	    				_zr * (_xyzuvw[5 + 6 *i] - _xyzuvw[5 + 6 * j]);

	    			const float mysaru = 0.5;//saru(min(i, j), max(i, j), saru_tid);
	    			const float myrandnr = 3.464101615f * mysaru - 1.732050807f;
	    #if 0
	    			const float e = c.A1 * (1  + rij2 * c.A2);
	    #else
	    			const float e = c.A1 * expf(rij2 * c.A2);
	    #endif
	    			float strength = -c.A0 * e * (1 - e) + (- c.gamma * wr * rdotv + c.B0 * myrandnr) * wr;

	    const bool collision =  rij2 < 6.25;

	    if (collision)
		fc += strength * _yr;//	printf("ref p %d colliding with %d\n", i, j);

	    cnt += collision;
	}
	if (cnt != (int)axayaz[0 + 3 * i]) printf("i found %d host interactions and with cuda i found %d\n", cnt, (int)axayaz[0 + 3 * i]);
	//assert((float)cnt == (float)axayaz[0 + 3 * i]);
	if (fabs((float)fc - (float)axayaz[1 + 3 * i]) > 1e-4) printf("fc aij ref %f vs cuda %f\n", fc,  (float)axayaz[1 + 3 * i]);
	//assert(fabs((float)fc - (float)axayaz[1 + 3 * i]) < 1e-4);
    }

    printf("test done.\n");
    //sleep(1);
    exit(0);
#endif
}

void forces_sem_cuda_direct(float * const xp, float * const yp, float * const zp,
		     float * const xv, float * const yv, float * const zv,
		     float * const xa, float * const ya, float * const za,
		     const int np,
		     const float rcutoff,
		     const float LX, const float LY, const float LZ,
		     const double gamma, const double temp, const double dt, const double u0, const double rho, const double req, const double D, const double rc)
{
	if (np <= 0) return;

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
    memset(a, 0, sizeof(float) * 3 * np);

    device_vector<float> xyzuvw(pv, pv + np * 6),  axayaz(np * 3);

    forces_sem_cuda_direct_nohost(_ptr(xyzuvw), _ptr(axayaz), np, rcutoff, LX, LY, LZ,
		    gamma, temp, dt, u0, rho, req, D, rc);

    copy(axayaz.begin(), axayaz.end(), a);

    for(int i = 0; i < np; ++i)
    {
    	xa[i] += a[0 + 3 * i];
    	ya[i] += a[1 + 3 * i];
    	za[i] += a[2 + 3 * i];
    }

    delete [] pv;
    delete [] a;
}




