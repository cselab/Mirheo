/*
 *  rbc-interactions.cu
 *  Part of CTC/mpi-dpd/
 *
 *  Created and authored by Diego Rossinelli on 2014-12-02.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */

#include <set>
#include <../dpd-rng.h>

#include "rbc-interactions.h"
#include "minmax-massimo.h"

namespace KernelsRBC
{
    struct ParamsFSI
    {
	float aij, gamma, sigmaf;
    };

    __constant__ ParamsFSI params;

    texture<float2, cudaTextureType1D> texSolventParticles, texSoluteParticles;
    texture<int, cudaTextureType1D> texCellsStart, texCellsCount, texSoluteCellsStart, texSoluteCellsCount;

    static bool firsttime = true;

    __global__ void shift_send_particles_kernel(const Particle * const src, const int n, const int code, Particle * const dst)
    {
	const int gid = threadIdx.x + blockDim.x * blockIdx.x;

	const int d[3] = { (code + 2) % 3 - 1, (code / 3 + 2) % 3 - 1, (code / 9 + 2) % 3 - 1 };
	const int L[3] = { XSIZE_SUBDOMAIN, YSIZE_SUBDOMAIN, ZSIZE_SUBDOMAIN };

	if (gid < n)
	{
	    Particle p = src[gid];

	    for(int c = 0; c < 3; ++c)
		p.x[c] -= d[c] * L[c];

	    dst[gid] = p;
	}
    }

    static const int cmaxnrbcs = 64;
    __constant__ float * csources[cmaxnrbcs], * cdestinations[cmaxnrbcs];
    __constant__ int ccodes[cmaxnrbcs];

    template <bool from_cmem>
    __global__ void shift_all_send_particles(const int nrbcs, const int nvertices,
					     const float ** const dsources, const int * dcodes, float ** const ddestinations)
    {
	const int nfloats_per_rbc = 6 * nvertices;

	assert(nfloats_per_rbc * nrbcs <= blockDim.x * gridDim.x);

	const int gid = threadIdx.x + blockDim.x * blockIdx.x;

	if (gid >= nfloats_per_rbc * nrbcs)
	    return;

	const int idrbc = gid / nfloats_per_rbc;
	assert(idrbc < nrbcs);

	const int offset = gid % nfloats_per_rbc;

	float val;
	if (from_cmem)
	    val = csources[idrbc][offset];
	else
	    val = dsources[idrbc][offset];

	int code;
	if (from_cmem)
	    code = ccodes[idrbc];
	else
	    code = dcodes[idrbc];

	const int c = gid % 6;

	val -=
	    (c == 0) * ((code     + 2) % 3 - 1) * XSIZE_SUBDOMAIN +
	    (c == 1) * ((code / 3 + 2) % 3 - 1) * YSIZE_SUBDOMAIN +
	    (c == 2) * ((code / 9 + 2) % 3 - 1) * ZSIZE_SUBDOMAIN ;

	if (from_cmem)
	    cdestinations[idrbc][offset] = val;
	else
	    ddestinations[idrbc][offset] = val;
    }

    SimpleDeviceBuffer<float *> _ddestinations;
    SimpleDeviceBuffer<const float *> _dsources;
    SimpleDeviceBuffer<int> _dcodes;

    void dispose()
    {
	_ddestinations.dispose();
	_dsources.dispose();
	_dcodes.dispose();
    }

    void shift_send_particles(cudaStream_t stream, const int nrbcs, const int nvertices,
			      const float ** const sources, const int * codes, float ** const destinations)
    {
	if (nrbcs == 0)
	    return;

	const int nthreads = nrbcs * nvertices * 6;

	if (nrbcs < cmaxnrbcs)
	{
	    if (!is_mps_enabled) {
		CUDA_CHECK(cudaMemcpyToSymbolAsync(ccodes, codes, sizeof(int) * nrbcs, 0, cudaMemcpyHostToDevice, stream));
		CUDA_CHECK(cudaMemcpyToSymbolAsync(cdestinations, destinations, sizeof(float *) * nrbcs, 0, cudaMemcpyHostToDevice, stream));
		CUDA_CHECK(cudaMemcpyToSymbolAsync(csources, sources, sizeof(float *) * nrbcs, 0, cudaMemcpyHostToDevice, stream));
	    } else {
		CUDA_CHECK(cudaMemcpyToSymbol(ccodes, codes, sizeof(int) * nrbcs, 0, cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpyToSymbol(cdestinations, destinations, sizeof(float *) * nrbcs, 0, cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpyToSymbol(csources, sources, sizeof(float *) * nrbcs, 0, cudaMemcpyHostToDevice));
	    }

	    shift_all_send_particles<true><<<(nthreads + 127) / 128, 128, 0, stream>>>
		(nrbcs, nvertices, NULL, NULL, NULL);

	    CUDA_CHECK(cudaPeekAtLastError());
	}
	else
	{
	    _dcodes.resize(nrbcs);
	    _ddestinations.resize(nrbcs);
	    _dsources.resize(nrbcs);

	    if (!is_mps_enabled) {
		CUDA_CHECK(cudaMemcpyAsync(_dcodes.data, codes, sizeof(int) * nrbcs, cudaMemcpyHostToDevice, stream));
		CUDA_CHECK(cudaMemcpyAsync(_ddestinations.data, destinations, sizeof(float *) * nrbcs, cudaMemcpyHostToDevice, stream));
		CUDA_CHECK(cudaMemcpyAsync(_dsources.data, sources, sizeof(float *) * nrbcs, cudaMemcpyHostToDevice, stream));
	    } else {
		CUDA_CHECK(cudaMemcpy(_dcodes.data, codes, sizeof(int) * nrbcs, cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(_ddestinations.data, destinations, sizeof(float *) * nrbcs, cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(_dsources.data, sources, sizeof(float *) * nrbcs, cudaMemcpyHostToDevice));
	    }

	    shift_all_send_particles<false><<<(nthreads + 127) / 128, 128, 0, stream>>>
		(nrbcs, nvertices, _dsources.data, _dcodes.data, _ddestinations.data);
	}
    }

    template <bool from_cmem>
    __global__ void merge_all_acc(const int nrbcs, const int nvertices,
				  const float ** const dsources, float ** const ddestinations)
    {
	if (nrbcs == 0)
	    return;

	const int nfloats_per_rbc = 3 * nvertices;

	assert(nfloats_per_rbc * nrbcs <= blockDim.x * gridDim.x);

	const int gid = threadIdx.x + blockDim.x * blockIdx.x;

	if (gid >= nfloats_per_rbc * nrbcs)
	    return;

	const int idrbc = gid / nfloats_per_rbc;
	assert(idrbc < nrbcs);

	const int offset = gid % nfloats_per_rbc;

	float val;
	if (from_cmem)
	    val = csources[idrbc][offset];
	else
	    val = dsources[idrbc][offset];

	if (from_cmem)
	    atomicAdd(cdestinations[idrbc] + offset, val);
	else
	    atomicAdd(ddestinations[idrbc] + offset, val);
    }

    void merge_all_accel(cudaStream_t stream, const int nrbcs, const int nvertices,
			 const float ** const sources, float ** const destinations)
    {
	if (nrbcs == 0)
	    return;

	const int nthreads = nrbcs * nvertices * 3;

	CUDA_CHECK(cudaPeekAtLastError());

	if (nrbcs < cmaxnrbcs)
	{
	    if (!is_mps_enabled) {
		CUDA_CHECK(cudaMemcpyToSymbolAsync(cdestinations, destinations, sizeof(float *) * nrbcs, 0, cudaMemcpyHostToDevice, stream));
		CUDA_CHECK(cudaMemcpyToSymbolAsync(csources, sources, sizeof(float *) * nrbcs, 0, cudaMemcpyHostToDevice, stream));
	    } else {
		CUDA_CHECK(cudaMemcpyToSymbol(cdestinations, destinations, sizeof(float *) * nrbcs, 0, cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpyToSymbol(csources, sources, sizeof(float *) * nrbcs, 0, cudaMemcpyHostToDevice));
	    }

	    merge_all_acc<true><<<(nthreads + 127) / 128, 128, 0, stream>>>(nrbcs, nvertices, NULL, NULL);

	    CUDA_CHECK(cudaPeekAtLastError());
	}
	else
	{
	    _ddestinations.resize(nrbcs);
	    _dsources.resize(nrbcs);

	    if (!is_mps_enabled) {
		CUDA_CHECK(cudaMemcpyAsync(_ddestinations.data, destinations, sizeof(float *) * nrbcs, cudaMemcpyHostToDevice, stream));
		CUDA_CHECK(cudaMemcpyAsync(_dsources.data, sources, sizeof(float *) * nrbcs, cudaMemcpyHostToDevice, stream));
	    } else {
		CUDA_CHECK(cudaMemcpy(_ddestinations.data, destinations, sizeof(float *) * nrbcs, cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(_dsources.data, sources, sizeof(float *) * nrbcs, cudaMemcpyHostToDevice));
	    }

	    merge_all_acc<false><<<(nthreads + 127) / 128, 128, 0, stream>>>(nrbcs, nvertices, _dsources.data, _ddestinations.data);
	}
    }

    __device__ bool fsi_kernel(const float seed,
			       const int dpid, const float3 xp, const float3 up, const int spid,
			       float& xforce, float& yforce, float& zforce)
    {
	xforce = yforce = zforce = 0;

	const int sentry = 3 * spid;

	const float2 stmp0 = tex1Dfetch(texSolventParticles, sentry);
	const float2 stmp1 = tex1Dfetch(texSolventParticles, sentry + 1);
	const float2 stmp2 = tex1Dfetch(texSolventParticles, sentry + 2);

	const float _xr = xp.x - stmp0.x;
	const float _yr = xp.y - stmp0.y;
	const float _zr = xp.z - stmp1.x;

	const float rij2 = _xr * _xr + _yr * _yr + _zr * _zr;

	if (rij2 > 1)
	    return false;

	const float invrij = rsqrtf(rij2);

	const float rij = rij2 * invrij;
	const float argwr = max((float)0, 1 - rij);
	const float wr = powf(argwr, powf(0.5f, -VISCOSITY_S_LEVEL));

	const float xr = _xr * invrij;
	const float yr = _yr * invrij;
	const float zr = _zr * invrij;

	const float rdotv =
	    xr * (up.x - stmp1.y) +
	    yr * (up.y - stmp2.x) +
	    zr * (up.z - stmp2.y);

	const float myrandnr = Logistic::mean0var1(seed, dpid, spid);

	const float strength = params.aij * argwr +  (- params.gamma * wr * rdotv + params.sigmaf * myrandnr) * wr;

	xforce = strength * xr;
	yforce = strength * yr;
	zforce = strength * zr;

	return true;
    }

    __device__ float3 fsi_interaction(const float seed,
				      const int dpid, const float3 up, const int spid,
				      const float2 stmp1, const float2 stmp2,
				      const float _xr, const float _yr, const float _zr, const float rij2)
    {
	const float invrij = rsqrtf(rij2);

	const float rij = rij2 * invrij;
	const float argwr = 1 - rij;
	const float wr = viscosity_function<-VISCOSITY_S_LEVEL>(argwr);

	const float xr = _xr * invrij;
	const float yr = _yr * invrij;
	const float zr = _zr * invrij;

	const float rdotv =
	    xr * (up.x - stmp1.y) +
	    yr * (up.y - stmp2.x) +
	    zr * (up.z - stmp2.y);

	const float myrandnr = Logistic::mean0var1(seed, dpid, spid);

	const float strength = params.aij * argwr +  (- params.gamma * wr * rdotv + params.sigmaf * myrandnr) * wr;

	return make_float3(strength * xr, strength * yr, strength * zr);
    }


    template<int XCPB, int YCPB, int ZCPB, int COLS, int ROWS>
    __global__ void fsi_forces(const float seed,
			       float * const accsolute, const int nsolute,
			       float * const accsolvent, const int nsolvent)
    {
	enum { CPB = XCPB * YCPB * ZCPB };

	assert(warpSize == COLS * ROWS);
	assert(blockDim.x == warpSize && blockDim.y == CPB && blockDim.z == 1);
	assert(ROWS * 3 <= warpSize);

	const int tid = threadIdx.x;
	const int wid = threadIdx.y;

	const int subtid = tid % COLS;
	const int slot = tid / COLS;

	__shared__ int volatile starts[CPB][32], scan[CPB][32];

	const int xmycid = blockIdx.x * XCPB + ((wid) % XCPB);
	const int ymycid = blockIdx.y * YCPB + ((wid / XCPB) % YCPB);
	const int zmycid = blockIdx.z * ZCPB + ((wid / (XCPB * YCPB)) % ZCPB);

	assert(xmycid >= 0 && xmycid < XSIZE_SUBDOMAIN);
	assert(ymycid >= 0 && ymycid < YSIZE_SUBDOMAIN);
	assert(zmycid >= 0 && zmycid < ZSIZE_SUBDOMAIN);

	const int mycid = xmycid + XSIZE_SUBDOMAIN * (ymycid + YSIZE_SUBDOMAIN * zmycid);

	int mycount = 0, myscan = 0;

	if (tid < 27)
	{
	    const int dx = tid % 3;
	    const int dy = (tid / 3) % 3;
	    const int dz = (tid / 9) % 3;

	    int xcid = xmycid + dx - 1;
	    int ycid = ymycid + dy - 1;
	    int zcid = zmycid + dz - 1;

	    const bool valid_cid =
		xcid >= 0 && xcid < XSIZE_SUBDOMAIN &&
		ycid >= 0 && ycid < YSIZE_SUBDOMAIN &&
		zcid >= 0 && zcid < ZSIZE_SUBDOMAIN ;

	    xcid = min(XSIZE_SUBDOMAIN - 1, max(0, xcid));
	    ycid = min(YSIZE_SUBDOMAIN - 1, max(0, ycid));
	    zcid = min(ZSIZE_SUBDOMAIN - 1, max(0, zcid));

	    const int cid = max(0, xcid + XSIZE_SUBDOMAIN * (ycid + YSIZE_SUBDOMAIN * zcid));

	    starts[wid][tid] = tex1Dfetch(texCellsStart, cid);

	    myscan = mycount = valid_cid * tex1Dfetch(texCellsCount, cid);
	}

	for(int L = 1; L < 32; L <<= 1)
	    myscan += (tid >= L) * __shfl_up(myscan, L);

	if (tid < 28)
	    scan[wid][tid] = myscan - mycount;

	const int nsrc = scan[wid][27];

	const int dststart = tex1Dfetch(texSoluteCellsStart, mycid);
	const int lastdst = dststart + tex1Dfetch(texSoluteCellsCount, mycid);

	for(int dpid = dststart + slot; dpid < lastdst; dpid += ROWS)
	{
	    float3 xdest, udest, force = make_float3(0, 0, 0);

	    float2 dtmp0 = tex1Dfetch(texSoluteParticles, 3 * dpid);
	    xdest.x = dtmp0.x;
	    xdest.y = dtmp0.y;

	    dtmp0 = tex1Dfetch(texSoluteParticles, 3 * dpid + 1);
	    xdest.z = dtmp0.x;
	    udest.x = dtmp0.y;

	    dtmp0 = tex1Dfetch(texSoluteParticles, 3 * dpid + 2);
	    udest.y = dtmp0.x;
	    udest.z = dtmp0.y;

	    for(int pid = subtid; pid < nsrc; pid += COLS)
	    {
		const int key9 = 9 * ((pid >= scan[wid][9]) + (pid >= scan[wid][18]));
		const int key3 = 3 * ((pid >= scan[wid][key9 + 3]) + (pid >= scan[wid][key9 + 6]));
		const int key = key9 + key3;

		const int spid = pid - scan[wid][key] + starts[wid][key];
		const int sentry = 3 * spid;
		const float2 stmp0 = tex1Dfetch(texSolventParticles, sentry);
		const float2 stmp1 = tex1Dfetch(texSolventParticles, sentry + 1);

		const float xr = xdest.x - stmp0.x;
		const float yr = xdest.y - stmp0.y;
		const float zr = xdest.z - stmp1.x;
		const float rij2 = xr * xr + yr * yr + zr * zr;

		if (rij2 < 1.0f)
		{
		    const float2 stmp2 = tex1Dfetch(texSolventParticles, sentry + 2);
		    const float3 f = fsi_interaction(seed, dpid, udest, spid, stmp1, stmp2, xr, yr, zr, rij2);

		    /*atomicAdd(accsolute + 3 * dpid    , f.x);
		      atomicAdd(accsolute + 3 * dpid + 1, f.y);
		      atomicAdd(accsolute + 3 * dpid + 2, f.z);*/
		    force.x += f.x;
		    force.y += f.y;
		    force.z += f.z;

		    atomicAdd(accsolvent + 3 * spid    , -f.x);
		    atomicAdd(accsolvent + 3 * spid + 1, -f.y);
		    atomicAdd(accsolvent + 3 * spid + 2, -f.z);
		}
	    }

	    atomicAdd(accsolute + 3 * dpid    , force.x);
	    atomicAdd(accsolute + 3 * dpid + 1, force.y);
	    atomicAdd(accsolute + 3 * dpid + 2, force.z);
	}
    }

    __global__ void fsi_forces_old(const float seed,
				   Acceleration * accsolvent, const int npsolvent,
				   const Particle * const particle, const int nparticles, Acceleration * accrbc)
    {
	const int dpid = threadIdx.x + blockDim.x * blockIdx.x;

	if (dpid >= nparticles)
	    return;

	const Particle p = particle[dpid];

	const float3 xp = make_float3(p.x[0], p.x[1], p.x[2]);
	const float3 up = make_float3(p.u[0], p.u[1], p.u[2]);

	const int L[3] = { XSIZE_SUBDOMAIN, YSIZE_SUBDOMAIN, ZSIZE_SUBDOMAIN };

	int mycid[3];
	for(int c = 0; c < 3; ++c)
	    mycid[c] = L[c]/2 + (int)floor(p.x[c]);

	for(int c = 0; c < 3; ++c)
	    if (mycid[c] < -1 || mycid[c] >= L[c] + 1)
	    {
		for(int c = 0; c < 3; ++c)
		    accrbc[dpid].a[c] = 0;

		return;
	    }

	float fsum[3] = {0, 0, 0};

	for(int code = 0; code < 27; ++code)
	{
	    const int d[3] = {
		(code % 3) - 1,
		(code/3 % 3) - 1,
		(code/9 % 3) - 1
	    };

	    int vcid[3];
	    for(int c = 0; c < 3; ++c)
		vcid[c] = mycid[c] + d[c];

	    bool validcid = true;
	    for(int c = 0; c < 3; ++c)
		validcid &= vcid[c] >= 0 && vcid[c] < L[c];

	    if (!validcid)
		continue;

	    const int cid = vcid[0] + XSIZE_SUBDOMAIN * (vcid[1] + YSIZE_SUBDOMAIN * vcid[2]);
	    const int mystart = tex1Dfetch(texCellsStart, cid);
	    const int myend = mystart + tex1Dfetch(texCellsCount, cid);

	    assert(mystart >= 0 && mystart <= myend);
	    assert(myend <= npsolvent);

#pragma unroll 4
	    for(int s = mystart; s < myend; ++s)
	    {
		float f[3];
		const bool nonzero = fsi_kernel(seed, dpid, xp, up, s, f[0], f[1], f[2]);

		if (nonzero)
		{
		    for(int c = 0; c < 3; ++c)
			fsum[c] += f[c];

		    for(int c = 0; c < 3; ++c)
			atomicAdd(c + (float *)(accsolvent + s), -f[c]);
		}
	    }
	}

	for(int c = 0; c < 3; ++c)
	    accrbc[dpid].a[c] = fsum[c];
    }

    __constant__ int packstarts[27];
    __constant__ Particle * packstates[26];
    __constant__ Acceleration * packresults[26];

    __device__ float3 __shfl_float3(float3 f, int l) {

	return make_float3(__shfl(f.x, l),
			   __shfl(f.y, l),
			   __shfl(f.z, l));
    }

    __constant__ char4 tid2ind[32] = {{-1, -1, -1, 0}, {0, -1, -1, 0}, {1, -1, -1, 0},
				      {-1,  0, -1, 0}, {0,  0, -1, 0}, {1,  0, -1, 0},
				      {-1 , 1, -1, 0}, {0,  1, -1, 0}, {1,  1, -1, 0},
				      {-1, -1,  0, 0}, {0, -1,  0, 0}, {1, -1,  0, 0},
				      {-1,  0,  0, 0}, {0,  0,  0, 0}, {1,  0,  0, 0},
				      {-1,  1,  0, 0}, {0,  1,  0, 0}, {1,  1,  0, 0},
				      {-1, -1,  1, 0}, {0, -1,  1, 0}, {1, -1,  1, 0},
				      {-1,  0,  1, 0}, {0,  0,  1, 0}, {1,  0,  1, 0},
				      {-1,  1,  1, 0}, {0,  1,  1, 0}, {1,  1,  1, 0},
				      { 0,  0,  0, 0}, {0,  0,  0, 0}, {0,  0,  0, 0},
				      { 0,  0,  0, 0}, {0,  0,  0, 0}};

    template<int BLOCKSIZE> __global__  __launch_bounds__(32 * 4, 16)
	void fsi_forces_all_nopref(const float seed, Acceleration * accsolvent, const int npsolvent, const int nremote)
    {
	assert(blockDim.x == BLOCKSIZE);
	assert(blockDim.x * gridDim.x >= nremote);

	__shared__ float tmp[BLOCKSIZE * 3];
	__shared__ int volatile starts[BLOCKSIZE];
	__shared__ int volatile scan[BLOCKSIZE];

	const int tid = threadIdx.x;
	const int gidstart =  BLOCKSIZE * blockIdx.x;

	const int lid = threadIdx.x%32;
	const int wof = threadIdx.x&(~31);
	const int wst = gidstart+wof;

	const int nlocal = min(BLOCKSIZE, nremote - gidstart);

	float3 xp, up;

#ifndef NDEBUG
	xp = make_float3(-313.313f, -313.313f, -313.313f); //che e' poi l'auto di paperino
	up = make_float3(-313.313f, -313.313f, -313.313f);
#endif

	{
	    const int n = nlocal * 6;
	    const int h = nlocal * 3;

	    for(int base = 0; base < n; base += h)
	    {
#pragma unroll 3
		for(int x = tid; x < h; x += BLOCKSIZE)
		{
		    const int l = base + x;
		    const int gid = gidstart + l / 6;

		    const int key9 = 9 * ((gid >= packstarts[9]) + (gid >= packstarts[18]));
		    const int key3 = 3 * ((gid >= packstarts[key9 + 3]) + (gid >= packstarts[key9 + 6]));
		    const int key1 = (gid >= packstarts[key9 + key3 + 1]) + (gid >= packstarts[key9 + key3 + 2]);

		    const int code = key9 + key3 + key1;
		    const int lpid = gid - packstarts[code];

		    assert(x < BLOCKSIZE * 3);
		    tmp[x] = *((l % 6) + (float *)&packstates[code][lpid]);
		}

		__syncthreads();

		const int xstart = tid * 6 - base;

		if (0 <= xstart && xstart + 3 <= h)
		{
		    xp.x = tmp[0 + xstart];
		    xp.y = tmp[1 + xstart];
		    xp.z = tmp[2 + xstart];

		    assert(0 + 6 * tid - base >= 0);
		    assert(2 + 6 * tid - base < 3 * BLOCKSIZE);
		}

		const int ustart = 3 + 6 * tid - base;

		if (0 <= ustart && ustart + 3 <= h)
		{
		    up.x = tmp[0 + ustart];
		    up.y = tmp[1 + ustart];
		    up.z = tmp[2 + ustart];

		    assert(3 + 6 * tid - base >= 0);
		    assert(5 + 6 * tid - base < 3 * BLOCKSIZE);
		}
	    }
	}

#ifndef NDEBUG
	assert(xp.x != -313.313f || gidstart + tid >= nremote);
	assert(xp.y != -313.313f || gidstart + tid >= nremote);
	assert(xp.z != -313.313f || gidstart + tid >= nremote);
	assert(up.x != -313.313f || gidstart + tid >= nremote);
	assert(up.y != -313.313f || gidstart + tid >= nremote);
	assert(up.z != -313.313f || gidstart + tid >= nremote);
#endif

	assert(!isnan(xp.x) && !isnan(xp.y) && !isnan(xp.z));
	assert(!isnan(up.x) && !isnan(up.y) && !isnan(up.z));

	__syncthreads();

	if (wst < nremote)
	{
	    char mycid[4] = {-2,-2,-2,0};
	    if (tid + gidstart < nremote) {
	        mycid[0] = XSIZE_SUBDOMAIN / 2 + (int)floor(xp.x);
	        mycid[1] = YSIZE_SUBDOMAIN / 2 + (int)floor(xp.y);
	        mycid[2] = ZSIZE_SUBDOMAIN / 2 + (int)floor(xp.z);
		mycid[3] = 1;

	        if (mycid[0] < -1 || mycid[0] >= XSIZE_SUBDOMAIN + 1 ||
		    mycid[1] < -1 || mycid[1] >= YSIZE_SUBDOMAIN + 1 ||
		    mycid[2] < -1 || mycid[2] >= ZSIZE_SUBDOMAIN + 1)
		    mycid[3] = 0;
            }

	    float fsum[3] = {0, 0, 0};
	    const char4 offs = tid2ind[lid];

	    for(int l = 0; l < 32; l++) {

		char ccel[4];
		*((int *)ccel) = __shfl(*((int *)mycid), l);
		if (!ccel[3]) continue;

    		int mycount=0, myscan=0;
		if (lid < 27) {

		    ccel[0] += offs.x;
		    ccel[1] += offs.y;
		    ccel[2] += offs.z;

		    bool validcid = ccel[0] >= 0 && ccel[0] < XSIZE_SUBDOMAIN &&
			ccel[1] >= 0 && ccel[1] < YSIZE_SUBDOMAIN &&
			ccel[2] >= 0 && ccel[2] < ZSIZE_SUBDOMAIN;

		    const int cid = (validcid) ? (ccel[0] + XSIZE_SUBDOMAIN*(ccel[1] + YSIZE_SUBDOMAIN*ccel[2])) : 0;
		    starts[threadIdx.x] = (validcid) ? tex1Dfetch(texCellsStart, cid) : 0;
		    myscan = mycount = (validcid) ? tex1Dfetch(texCellsCount, cid) : 0;
		}
#pragma unroll
		for(int L = 1; L < 32; L <<= 1)
		    myscan += (lid >= L)*__shfl_up(myscan, L);

		if (lid < 28) scan[threadIdx.x] = myscan - mycount;

		float ftmp[3] = {0, 0, 0};
		float3 dxp = __shfl_float3(xp, l);
		float3 dup = __shfl_float3(up, l);

		const int did = wst+l;

		const int nsrc = scan[wof+27];
		for(int sid = lid; sid < nsrc; sid += 32) {

		    const int key9 = 9*((sid >= scan[wof + 9]) + (sid >= scan[wof + 18]));
		    const int key3 = 3*((sid >= scan[wof + key9+3]) + (sid >= scan[wof + key9+6]));
		    const int key1 = (sid >= scan[wof + key9+key3+1]) + (sid >= scan[wof + key9+key3+2]);
		    int s = sid - scan[wof + key3+key9+key1] + starts[wof + key3+key9+key1];

		    float f[3];
		    const bool nonzero = fsi_kernel(seed, did, dxp, dup, s, f[0], f[1], f[2]);

		    if (nonzero) {
			ftmp[0] += f[0];
			ftmp[1] += f[1];
			ftmp[2] += f[2];

			atomicAdd((float *)(accsolvent + s),   -f[0]);
			atomicAdd((float *)(accsolvent + s)+1, -f[1]);
			atomicAdd((float *)(accsolvent + s)+2, -f[2]);
		    }
		}
#pragma unroll
		for(int z = 16; z; z >>= 1) {
		    ftmp[0] += __shfl_xor(ftmp[0], z);
		    ftmp[1] += __shfl_xor(ftmp[1], z);
		    ftmp[2] += __shfl_xor(ftmp[2], z);
		}
		if (l == lid) {
		    fsum[0] = ftmp[0];
		    fsum[1] = ftmp[1];
		    fsum[2] = ftmp[2];
		}
	    }

	    for(int c = 0; c < 3;  ++c)
		assert(!isnan(fsum[c]));

	    tmp[0 + 3 * tid] = fsum[0];
	    tmp[1 + 3 * tid] = fsum[1];
	    tmp[2 + 3 * tid] = fsum[2];
	}

	__syncthreads();

	{
	    const int n = nlocal * 3;

#pragma unroll 3
	    for(int l = tid; l < n; l += BLOCKSIZE)
	    {
		const int gid = gidstart + l / 3;

		const int key9 = 9 * ((gid >= packstarts[9]) + (gid >= packstarts[18]));
		const int key3 = 3 * ((gid >= packstarts[key9 + 3]) + (gid >= packstarts[key9 + 6]));
		const int key1 = (gid >= packstarts[key9 + key3 + 1]) + (gid >= packstarts[key9 + key3 + 2]);

		const int code = key9 + key3 + key1;
		const int lpid = gid - packstarts[code];

		packresults[code][lpid].a[l % 3] = tmp[l];
	    }
	}
    }

    __global__ void merge_accelerations(const Acceleration * const src, const int n, Acceleration * const dst)
    {
	const int gid = threadIdx.x + blockDim.x * blockIdx.x;

	if (gid < n)
	    for(int c = 0; c < 3; ++c)
		dst[gid].a[c] += src[gid].a[c];
    }

    __global__ void merge_accelerations_float(const Acceleration * const src, const int n, Acceleration * const dst)
    {
	assert(blockDim.x * gridDim.x >= n * 3);

	const int gid = threadIdx.x + blockDim.x * blockIdx.x;

	const int pid = gid / 3;
	const int c = gid % 3;

	if (pid < n)
	    dst[pid].a[c] += src[pid].a[c];
    }

    template<bool accumulation>
    __global__ void merge_accelerations_scattered_float(const int * const reordering, const Acceleration * const src,
							const int n, Acceleration * const dst)
    {
	assert(blockDim.x * gridDim.x >= n * 3);

	const int gid = threadIdx.x + blockDim.x * blockIdx.x;

	const int pid = gid / 3;
	const int c = gid % 3;

	if (pid < n)
	{
	    const int actualpid = reordering[pid];

	    if (accumulation)
		dst[actualpid].a[c] += src[pid].a[c];
	    else
		dst[actualpid].a[c] = src[pid].a[c];
	}
    }


    void setup(const Particle * const solvent, const int npsolvent, const int * const cellsstart, const int * const cellscount,
	       const Particle * const solute, const int npsolute, const int * const solute_cellsstart, const int * const solute_cellscount)
    {
	if (firsttime)
	{
	    texCellsStart.channelDesc = cudaCreateChannelDesc<int>();
	    texCellsStart.filterMode = cudaFilterModePoint;
	    texCellsStart.mipmapFilterMode = cudaFilterModePoint;
	    texCellsStart.normalized = 0;

	    texCellsCount.channelDesc = cudaCreateChannelDesc<int>();
	    texCellsCount.filterMode = cudaFilterModePoint;
	    texCellsCount.mipmapFilterMode = cudaFilterModePoint;
	    texCellsCount.normalized = 0;

	    texSoluteCellsStart.channelDesc = cudaCreateChannelDesc<int>();
	    texSoluteCellsStart.filterMode = cudaFilterModePoint;
	    texSoluteCellsStart.mipmapFilterMode = cudaFilterModePoint;
	    texSoluteCellsStart.normalized = 0;

	    texSoluteCellsCount.channelDesc = cudaCreateChannelDesc<int>();
	    texSoluteCellsCount.filterMode = cudaFilterModePoint;
	    texSoluteCellsCount.mipmapFilterMode = cudaFilterModePoint;
	    texSoluteCellsCount.normalized = 0;

	    texSolventParticles.channelDesc = cudaCreateChannelDesc<float2>();
	    texSolventParticles.filterMode = cudaFilterModePoint;
	    texSolventParticles.mipmapFilterMode = cudaFilterModePoint;
	    texSolventParticles.normalized = 0;

	    texSoluteParticles.channelDesc = cudaCreateChannelDesc<float2>();
	    texSoluteParticles.filterMode = cudaFilterModePoint;
	    texSoluteParticles.mipmapFilterMode = cudaFilterModePoint;
	    texSoluteParticles.normalized = 0;

	    firsttime = false;
	}

	size_t textureoffset;
	if (npsolvent)
	    CUDA_CHECK(cudaBindTexture(&textureoffset, &texSolventParticles, solvent, &texSolventParticles.channelDesc,
				       sizeof(float) * 6 * npsolvent));
	assert(textureoffset == 0);


	if (npsolute)
	    CUDA_CHECK(cudaBindTexture(&textureoffset, &texSoluteParticles, solute, &texSoluteParticles.channelDesc, sizeof(float) * 6 * npsolute));
	assert(textureoffset == 0);


	const int ncells = XSIZE_SUBDOMAIN * YSIZE_SUBDOMAIN * ZSIZE_SUBDOMAIN;

	assert(textureoffset == 0);
	CUDA_CHECK(cudaBindTexture(&textureoffset, &texCellsStart, cellsstart, &texCellsStart.channelDesc, sizeof(int) * ncells));
	assert(textureoffset == 0);
	CUDA_CHECK(cudaBindTexture(&textureoffset, &texCellsCount, cellscount, &texCellsCount.channelDesc, sizeof(int) * ncells));
	assert(textureoffset == 0);

	if (solute_cellsstart)
	    CUDA_CHECK(cudaBindTexture(&textureoffset, &texSoluteCellsStart, solute_cellsstart, &texSoluteCellsStart.channelDesc,
				       sizeof(int) * ncells));
	assert(textureoffset == 0);
	
	if (solute_cellscount)
	    CUDA_CHECK(cudaBindTexture(&textureoffset, &texSoluteCellsCount, solute_cellscount, &texSoluteCellsCount.channelDesc,
				       sizeof(int) * ncells));
	assert(textureoffset == 0);


	CUDA_CHECK(cudaFuncSetCacheConfig(fsi_forces<2, 2, 1, 8, 4>, cudaFuncCachePreferL1));
	CUDA_CHECK(cudaFuncSetCacheConfig(fsi_forces_old, cudaFuncCachePreferL1));

    }
}

ComputeInteractionsRBC::ComputeInteractionsRBC(MPI_Comm _cartcomm):
nvertices(0), dualcells(XSIZE_SUBDOMAIN, YSIZE_SUBDOMAIN, ZSIZE_SUBDOMAIN)
{
    assert(XSIZE_SUBDOMAIN % 2 == 0 && YSIZE_SUBDOMAIN % 2 == 0 && ZSIZE_SUBDOMAIN % 2 == 0);
    assert(XSIZE_SUBDOMAIN >= 2 && YSIZE_SUBDOMAIN >= 2 && ZSIZE_SUBDOMAIN >= 2);

    if (rbcs)
    {
	CudaRBC::Extent host_extent;
	CudaRBC::setup(nvertices, host_extent);
    }

    MPI_CHECK( MPI_Comm_dup(_cartcomm, &cartcomm));

    MPI_CHECK( MPI_Comm_rank(cartcomm, &myrank));

    local_trunk = Logistic::KISS(1908 - myrank, 1409 + myrank, 290, 12968);

    MPI_CHECK( MPI_Comm_size(cartcomm, &nranks));

    MPI_CHECK( MPI_Cart_get(cartcomm, 3, dims, periods, coords) );

    for(int i = 0; i < 26; ++i)
    {
	int d[3] = { (i + 2) % 3 - 1, (i / 3 + 2) % 3 - 1, (i / 9 + 2) % 3 - 1 };

	recv_tags[i] = (2 - d[0]) % 3 + 3 * ((2 - d[1]) % 3 + 3 * ((2 - d[2]) % 3));

	int coordsneighbor[3];
	for(int c = 0; c < 3; ++c)
	    coordsneighbor[c] = coords[c] + d[c];

	MPI_CHECK( MPI_Cart_rank(cartcomm, coordsneighbor, dstranks + i) );
    }

    KernelsRBC::ParamsFSI params = {12.5 , gammadpd, sigmaf};

    CUDA_CHECK(cudaMemcpyToSymbol(KernelsRBC::params, &params, sizeof(KernelsRBC::ParamsFSI)));

    CUDA_CHECK(cudaEventCreate(&evextents, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreate(&evfsi, cudaEventDisableTiming));
}

void ComputeInteractionsRBC::_compute_extents(const Particle * const rbcs, const int nrbcs, cudaStream_t stream)
{
#if 1
    if (nrbcs)
	minmax_massimo(rbcs, nvertices, nrbcs, minextents.devptr, maxextents.devptr, stream);
#else
    for(int i = 0; i < nrbcs; ++i)
	CudaRBC::extent_nohost(stream, (float *)(rbcs + nvertices * i), extents.devptr + i);
#endif
}

void ComputeInteractionsRBC::extent(const Particle * const rbcs, const int nrbcs, cudaStream_t stream)
{
    NVTX_RANGE("RBC/extent", NVTX_C2);
    
    const int nsolute = nrbcs * nvertices;
   
    reordered_solute.resize(nsolute);
    reordering.resize(nsolute);
    lacc_solute.resize(nsolute);
    
    if (nrbcs > 0)
    {
	CUDA_CHECK(cudaMemsetAsync(lacc_solute.data, 0, sizeof(float) * 3 * nsolute, stream));	

	CUDA_CHECK(cudaMemcpyAsync(reordered_solute.data, rbcs, sizeof(Particle) * nrbcs * nvertices, cudaMemcpyDeviceToDevice, stream));

	dualcells.build(reordered_solute.data, nrbcs * nvertices, stream, reordering.data);
    }

    minextents.resize(nrbcs);
    maxextents.resize(nrbcs);

    _compute_extents(rbcs, nrbcs, stream);

    CUDA_CHECK(cudaEventRecord(evextents, stream));
}

void ComputeInteractionsRBC::count(const int nrbcs)
{
    NVTX_RANGE("RBC/count", NVTX_C3);

    CUDA_CHECK(cudaEventSynchronize(evextents));

    for(int i = 0; i < 26; ++i)
	haloreplica[i].clear();

    for(int i = 0; i < nrbcs; ++i)
    {
	float pmin[3] = { minextents.data[i].x, minextents.data[i].y, minextents.data[i].z };
	float pmax[3] = { maxextents.data[i].x, maxextents.data[i].y, maxextents.data[i].z };

	for(int code = 0; code < 26; ++code)
	{
	    const int L[3] = { XSIZE_SUBDOMAIN, YSIZE_SUBDOMAIN, ZSIZE_SUBDOMAIN };
	    const int d[3] = { (code + 2) % 3 - 1, (code / 3 + 2) % 3 - 1, (code / 9 + 2) % 3 - 1 };

	    bool interacting = true;

	    for(int c = 0; c < 3; ++c)
	    {
		const float range_start = max((float)(d[c] * L[c] - L[c]/2 - 1), pmin[c]);
		const float range_end = min((float)(d[c] * L[c] + L[c]/2 + 1), pmax[c]);

		interacting &= range_end > range_start;
	    }

	    if (interacting)
		haloreplica[code].push_back(i);
	}
    }

    for(int i = 0; i <26; ++i)
	MPI_CHECK(MPI_Irecv(recv_counts + i, 1, MPI_INTEGER, dstranks[i], recv_tags[i] + 2077, cartcomm, reqrecvcounts + i));


    for(int i = 0; i < 26; ++i)
    {
	send_counts[i] = haloreplica[i].size();
	MPI_CHECK(MPI_Isend(send_counts + i, 1, MPI_INTEGER, dstranks[i], i + 2077, cartcomm, reqsendcounts + i));
    }

    for(int i = 0; i < 26; ++i)
	local[i].setup(send_counts[i] * nvertices);
}

void ComputeInteractionsRBC::exchange_count()
{
    NVTX_RANGE("RBC/exchange-count", NVTX_C4);

    MPI_Status statuses[26];
    MPI_CHECK(MPI_Waitall(26, reqrecvcounts, statuses));
    MPI_CHECK(MPI_Waitall(26, reqsendcounts, statuses));

    for(int i = 0; i < 26; ++i)
	remote[i].setup(recv_counts[i] * nvertices);
}

void ComputeInteractionsRBC::pack_p(const Particle * const rbcs, cudaStream_t stream)
{
    NVTX_RANGE("RBC/pack", NVTX_C4);

#if 1
    {
	static std::vector<int> codes;
	static std::vector<const float *> src;
	static std::vector<float *> dst;

	codes.clear();
	src.clear();
	dst.clear();

	for(int i = 0; i < 26; ++i)
	    for(int j = 0; j < haloreplica[i].size(); ++j)
	    {
		codes.push_back(i);
		src.push_back((float *)(rbcs + nvertices * haloreplica[i][j]));
		dst.push_back((float *)(local[i].state.devptr + nvertices * j));
	    }

	KernelsRBC::shift_send_particles(stream, src.size(), nvertices, &src.front(), &codes.front(), &dst.front());

	CUDA_CHECK(cudaPeekAtLastError());
    }
#else
    for(int i = 0; i < 26; ++i)
    {
	for(int j = 0; j < haloreplica[i].size(); ++j)
	    KernelsRBC::shift_send_particles<<< (nvertices + 127) / 128, 128, 0, stream>>>
		(rbcs + nvertices * haloreplica[i][j], nvertices, i, local[i].state.devptr + nvertices * j);

	CUDA_CHECK(cudaPeekAtLastError());
    }
#endif

    CUDA_CHECK(cudaEventRecord(evfsi, stream));
}

void ComputeInteractionsRBC::post_p()
{
    NVTX_RANGE("RBC/post-p", NVTX_C5);

    CUDA_CHECK(cudaEventSynchronize(evfsi));

    for(int i = 0; i < 26; ++i)
	if (recv_counts[i] > 0)
	{
	    MPI_Request request;

	    MPI_CHECK(MPI_Irecv(remote[i].state.data, recv_counts[i] * nvertices, Particle::datatype(), dstranks[i],
				recv_tags[i] + 2011, cartcomm, &request));

	    reqrecvp.push_back(request);
	}

    for(int i = 0; i < 26; ++i)
	if (send_counts[i] > 0)
	{
	    MPI_Request request;

	    MPI_CHECK(MPI_Irecv(local[i].result.data, send_counts[i] * nvertices, Acceleration::datatype(), dstranks[i],
				recv_tags[i] + 2285, cartcomm, &request));

	    reqrecvacc.push_back(request);

	    MPI_CHECK(MPI_Isend(local[i].state.data, send_counts[i] * nvertices, Particle::datatype(), dstranks[i],
				i + 2011, cartcomm, &request));

	    reqsendp.push_back(request);
	}
}

void ComputeInteractionsRBC::internal_forces(const Particle * const rbcs, const int nrbcs, Acceleration * accrbc, cudaStream_t stream)
{
    CudaRBC::forces_nohost(stream, nrbcs, (float *)rbcs, (float *)accrbc);
}

void ComputeInteractionsRBC::fsi_bulk(const Particle * const solvent, const int nparticles, Acceleration * accsolvent,
				      const int * const cellsstart_solvent, const int * const cellscount_solvent,
				      const Particle * const rbcs, const int nrbcs, Acceleration * accrbc, cudaStream_t stream)
{
    NVTX_RANGE("RBC/fsi-bulk", NVTX_C6);

    const int nsolute = nrbcs * nvertices;
    const int nsolvent = nparticles;
    
    KernelsRBC::setup(solvent, nparticles, cellsstart_solvent, cellscount_solvent,
		      reordered_solute.data, nsolute, dualcells.start, dualcells.count);

    if (nrbcs > 0 && nparticles > 0)
    {
	const float seed = local_trunk.get_float();

#if 1
	const int3 vcells = make_int3(XSIZE_SUBDOMAIN, YSIZE_SUBDOMAIN, ZSIZE_SUBDOMAIN);

	KernelsRBC::fsi_forces<2, 2, 1, 8, 4><<<
	    dim3(vcells.x / 2, vcells.y / 2, vcells.z), dim3(32, 4), 0, stream>>>
	    (seed, (float *)lacc_solute.data, nsolute, (float *)accsolvent, nsolvent);

        KernelsRBC::merge_accelerations_scattered_float<true><<< (nrbcs * nvertices * 3 + 127) / 128, 128, 0, stream >>>(
	    reordering.data, lacc_solute.data, nrbcs * nvertices, accrbc);

#else
	KernelsRBC::fsi_forces<<< (nrbcs * nvertices + 127) / 128, 128, 0, stream >>>
	    (seed, accsolvent, nparticles, rbcs, nrbcs * nvertices, accrbc);
#endif
    }
}

void ComputeInteractionsRBC::fsi_halo(const Particle * const solvent, const int nparticles, Acceleration * accsolvent,
				      const int * const cellsstart_solvent, const int * const cellscount_solvent,
				      const Particle * const rbcs, const int nrbcs, Acceleration * accrbc, cudaStream_t stream)
{
    NVTX_RANGE("RBC/fsi-halo", NVTX_C7);

    KernelsRBC::setup(solvent, nparticles, cellsstart_solvent, cellscount_solvent,
		      NULL, 0, NULL, 0);
    _wait(reqrecvp);
    _wait(reqsendp);

#if 1
    {
	int nremote = 0;

	{
	    static int packstarts[27];

	    packstarts[0] = 0;
	    for(int i = 0, s = 0; i < 26; ++i)
		packstarts[i + 1] = (s += remote[i].state.size);

	    nremote = packstarts[26];

	    if (!is_mps_enabled)
		CUDA_CHECK(cudaMemcpyToSymbolAsync(KernelsRBC::packstarts, packstarts,
					       sizeof(packstarts), 0, cudaMemcpyHostToDevice, stream));
	    else
		CUDA_CHECK(cudaMemcpyToSymbol(KernelsRBC::packstarts, packstarts,
					       sizeof(packstarts), 0, cudaMemcpyHostToDevice));
	}

	{
	    static Particle * packstates[26];

	    for(int i = 0; i < 26; ++i)
		packstates[i] = remote[i].state.devptr;

	    if (!is_mps_enabled)
		CUDA_CHECK(cudaMemcpyToSymbolAsync(KernelsRBC::packstates, packstates,
					       sizeof(packstates), 0, cudaMemcpyHostToDevice, stream));
	    else
		CUDA_CHECK(cudaMemcpyToSymbol(KernelsRBC::packstates, packstates,
					       sizeof(packstates), 0, cudaMemcpyHostToDevice));
	}

	{
	    static Acceleration * packresults[26];

	    for(int i = 0; i < 26; ++i)
		packresults[i] = remote[i].result.devptr;

	    if (!is_mps_enabled)
		CUDA_CHECK(cudaMemcpyToSymbolAsync(KernelsRBC::packresults, packresults,
					       sizeof(packresults), 0, cudaMemcpyHostToDevice, stream));
	    else
		CUDA_CHECK(cudaMemcpyToSymbol(KernelsRBC::packresults, packresults,
					       sizeof(packresults), 0, cudaMemcpyHostToDevice));
	}

	if(nremote)
	    KernelsRBC::fsi_forces_all_nopref<128><<< (nremote + 127) / 128, 128, 0, stream>>>
		(local_trunk.get_float(), accsolvent, nparticles, nremote);

    }
#else
    for(int i = 0; i < 26; ++i)
    {
	const int count = remote[i].state.size;

	if (count > 0)
	    KernelsRBC::fsi_forces<<< (count + 127) / 128, 128, 0, stream >>>
		(local_trunk.get_float(), accsolvent, nparticles, remote[i].state.devptr, count, remote[i].result.devptr);
    }
#endif

    CUDA_CHECK(cudaEventRecord(evfsi));

    CUDA_CHECK(cudaPeekAtLastError());
}

void ComputeInteractionsRBC::post_a()
{
    NVTX_RANGE("RBC/send-results", NVTX_C1);

    CUDA_CHECK(cudaEventSynchronize(evfsi));

    _wait(reqsendacc);

    for(int i = 0; i < 26; ++i)
	if (recv_counts[i] > 0)
	{
	    MPI_Request request;

	    MPI_CHECK(MPI_Isend(remote[i].result.data, recv_counts[i] * nvertices, Acceleration::datatype(), dstranks[i],
				i + 2285, cartcomm, &request));

	    reqsendacc.push_back(request);
	}
}

void ComputeInteractionsRBC::merge_a(Acceleration * accrbc, cudaStream_t stream)
{
    NVTX_RANGE("RBC/merge", NVTX_C2);

    _wait(reqrecvacc);

#if 1
    {
	static std::vector<const float *> src;
	static std::vector<float *> dst;

	src.clear();
	dst.clear();

	for(int i = 0; i < 26; ++i)
	    for(int j = 0; j < haloreplica[i].size(); ++j)
	    {
		src.push_back((float *)(local[i].result.devptr + nvertices * j));
		dst.push_back((float *)(accrbc + nvertices * haloreplica[i][j]));
	    }

	KernelsRBC::merge_all_accel(stream, src.size(), nvertices, &src.front(), &dst.front());

	CUDA_CHECK(cudaPeekAtLastError());
    }
#else
    for(int i = 0; i < 26; ++i)
	for(int j = 0; j < haloreplica[i].size(); ++j)
	    KernelsRBC::merge_accelerations<<< (nvertices + 127) / 128, 128, 0, stream>>>(local[i].result.devptr + nvertices * j, nvertices,
											  accrbc + nvertices * haloreplica[i][j]);
#endif
}

ComputeInteractionsRBC::~ComputeInteractionsRBC()
{
    MPI_CHECK(MPI_Comm_free(&cartcomm));

    CUDA_CHECK(cudaEventDestroy(evextents));
    CUDA_CHECK(cudaEventDestroy(evfsi));

    KernelsRBC::dispose();
}
