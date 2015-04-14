/*
 *  wall-interactions.cu
 *  Part of CTC/mpi-dpd/
 *
 *  Created and authored by Diego Rossinelli on 2014-11-19.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */

#include <sys/stat.h>
#include <sys/types.h>

#include <cmath>

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/count.h>

#ifndef NO_VTK
#include <vtkImageData.h>
#include <vtkXMLImageDataWriter.h>
#endif

#include "io.h"
#include "halo-exchanger.h"
#include "wall-interactions.h"
#include "redistancing.h"

enum
{
    XSIZE_WALLCELLS = 2 * XMARGIN_WALL + XSIZE_SUBDOMAIN,
    YSIZE_WALLCELLS = 2 * YMARGIN_WALL + YSIZE_SUBDOMAIN,
    ZSIZE_WALLCELLS = 2 * ZMARGIN_WALL + ZSIZE_SUBDOMAIN,

    XTEXTURESIZE = 256,

    _YTEXTURESIZE =
    ((YSIZE_SUBDOMAIN + 2 * YMARGIN_WALL) * XTEXTURESIZE + XSIZE_SUBDOMAIN + 2 * XMARGIN_WALL - 1)
    / (XSIZE_SUBDOMAIN + 2 * XMARGIN_WALL),

    YTEXTURESIZE = 16 * ((_YTEXTURESIZE + 15) / 16),

    _ZTEXTURESIZE =
    ((ZSIZE_SUBDOMAIN + 2 * ZMARGIN_WALL) * XTEXTURESIZE + XSIZE_SUBDOMAIN + 2 * XMARGIN_WALL - 1)
    / (XSIZE_SUBDOMAIN + 2 * XMARGIN_WALL),

    ZTEXTURESIZE = 16 * ((_ZTEXTURESIZE + 15) / 16),

};

namespace SolidWallsKernel
{
    texture<float, 3, cudaReadModeElementType> texSDF;

    texture<float4, 1, cudaReadModeElementType> texWallParticles;
    texture<int, 1, cudaReadModeElementType> texWallCellStart, texWallCellCount;

    __global__ void interactions(const float2 * particles, const int * const cellsstart, const int * const cellscount,
				 const float seed, const float sigmaf, float * const axayaz);

    __global__ void interactions_old(const Particle * const particles, const int np, const int nsolid,
				     Acceleration * const acc, const float seed, const float sigmaf);
    void setup()
    {
	texSDF.normalized = 0;
	texSDF.filterMode = cudaFilterModePoint;
	texSDF.mipmapFilterMode = cudaFilterModePoint;
	texSDF.addressMode[0] = cudaAddressModeWrap;
	texSDF.addressMode[1] = cudaAddressModeWrap;
	texSDF.addressMode[2] = cudaAddressModeWrap;

	texWallParticles.channelDesc = cudaCreateChannelDesc<float4>();
	texWallParticles.filterMode = cudaFilterModePoint;
	texWallParticles.mipmapFilterMode = cudaFilterModePoint;
	texWallParticles.normalized = 0;

	texWallCellStart.channelDesc = cudaCreateChannelDesc<int>();
	texWallCellStart.filterMode = cudaFilterModePoint;
	texWallCellStart.mipmapFilterMode = cudaFilterModePoint;
	texWallCellStart.normalized = 0;

	texWallCellCount.channelDesc = cudaCreateChannelDesc<int>();
	texWallCellCount.filterMode = cudaFilterModePoint;
	texWallCellCount.mipmapFilterMode = cudaFilterModePoint;
	texWallCellCount.normalized = 0;

	CUDA_CHECK(cudaFuncSetCacheConfig(*interactions, cudaFuncCachePreferL1));
	CUDA_CHECK(cudaFuncSetCacheConfig(interactions_old, cudaFuncCachePreferL1));
    }

    __device__ float sdf(float x, float y, float z)
    {
	const int L[3] = { XSIZE_SUBDOMAIN, YSIZE_SUBDOMAIN, ZSIZE_SUBDOMAIN };
	const int MARGIN[3] = { XMARGIN_WALL, YMARGIN_WALL, ZMARGIN_WALL };
	const int TEXSIZES[3] = {XTEXTURESIZE, YTEXTURESIZE, ZTEXTURESIZE };

	float p[3] = {x, y, z};

	float texcoord[3], lambda[3];
	for(int c = 0; c < 3; ++c)
	{
	    const float t = TEXSIZES[c] * (p[c] + L[c] / 2 + MARGIN[c]) / (L[c] + 2 * MARGIN[c]);

	    lambda[c] = t - (int)t;
	    texcoord[c] = (int)t + 0.5;

	    assert(texcoord[c] >= 0 && texcoord[c] <= TEXSIZES[c]);
	}

	const float s000 = tex3D(texSDF, texcoord[0] + 0, texcoord[1] + 0, texcoord[2] + 0);
	const float s001 = tex3D(texSDF, texcoord[0] + 1, texcoord[1] + 0, texcoord[2] + 0);
	const float s010 = tex3D(texSDF, texcoord[0] + 0, texcoord[1] + 1, texcoord[2] + 0);
	const float s011 = tex3D(texSDF, texcoord[0] + 1, texcoord[1] + 1, texcoord[2] + 0);
	const float s100 = tex3D(texSDF, texcoord[0] + 0, texcoord[1] + 0, texcoord[2] + 1);
	const float s101 = tex3D(texSDF, texcoord[0] + 1, texcoord[1] + 0, texcoord[2] + 1);
	const float s110 = tex3D(texSDF, texcoord[0] + 0, texcoord[1] + 1, texcoord[2] + 1);
	const float s111 = tex3D(texSDF, texcoord[0] + 1, texcoord[1] + 1, texcoord[2] + 1);

	const float s00x = s000 * (1 - lambda[0]) + lambda[0] * s001;
	const float s01x = s010 * (1 - lambda[0]) + lambda[0] * s011;
	const float s10x = s100 * (1 - lambda[0]) + lambda[0] * s101;
	const float s11x = s110 * (1 - lambda[0]) + lambda[0] * s111;

	const float s0yx = s00x * (1 - lambda[1]) + lambda[1] * s01x;
	const float s1yx = s10x * (1 - lambda[1]) + lambda[1] * s11x;

	const float szyx = s0yx * (1 - lambda[2]) + lambda[2] * s1yx;

	return szyx;
    }

    __device__ float3 grad_sdf(float x, float y, float z)
    {
	const int L[3] = { XSIZE_SUBDOMAIN, YSIZE_SUBDOMAIN, ZSIZE_SUBDOMAIN };
	const int MARGIN[3] = { XMARGIN_WALL, YMARGIN_WALL, ZMARGIN_WALL };
	const int TEXSIZES[3] = {XTEXTURESIZE, YTEXTURESIZE, ZTEXTURESIZE };
	const float p[3] = {x, y, z};

	float tc[3];
	for(int c = 0; c < 3; ++c)
	{
	    tc[c] = TEXSIZES[c] * (p[c] + L[c] / 2 + MARGIN[c]) / (L[c] + 2 * MARGIN[c]);

	    if (!(tc[c] >= 0 && tc[c] <= TEXSIZES[c]))
	    {
		cuda_printf("oooooooooops wall-interactions: texture coordinate %f exceeds bounds [0, %d] for c %d\nincrease MARGIN or decrease timestep", TEXSIZES[c], tc[c], c);
	    }

	    assert(tc[c] >= 0 && tc[c] <= TEXSIZES[c]);
	}

	float factors[3];
	for(int c = 0; c < 3; ++c)
	    factors[c] = 1. / (2 * TEXSIZES[c]) * 1.f / (2 * MARGIN[c] + L[c]);

	float xmygrad = factors[0] * (tex3D(texSDF, tc[0] + 1, tc[1], tc[2]) - tex3D(texSDF, tc[0] - 1, tc[1], tc[2]));
	float ymygrad = factors[1] * (tex3D(texSDF, tc[0], tc[1] + 1, tc[2]) - tex3D(texSDF, tc[0], tc[1] - 1, tc[2]));
	float zmygrad = factors[2] * (tex3D(texSDF, tc[0], tc[1], tc[2] + 1) - tex3D(texSDF, tc[0], tc[1], tc[2] - 1));

	float mygradmag = sqrt(xmygrad * xmygrad + ymygrad * ymygrad + zmygrad * zmygrad);

	if (mygradmag > 1e-6)
	{
	    xmygrad /= mygradmag;
	    ymygrad /= mygradmag;
	    zmygrad /= mygradmag;
	}

	return make_float3(xmygrad, ymygrad, zmygrad);
    }

    __global__ void fill_keys(const Particle * const particles, const int n, int * const key)
    {
	assert(blockDim.x * gridDim.x >= n);

	const int pid = threadIdx.x + blockDim.x * blockIdx.x;

	if (pid >= n)
	    return;

	const Particle p = particles[pid];

	const float mysdf = sdf(p.x[0], p.x[1], p.x[2]);
	key[pid] = (int)(mysdf >= 0) + (int)(mysdf > 2);
    }

    __global__ void strip_solid4(Particle * const src, const int n, float4 * dst)
    {
	assert(blockDim.x * gridDim.x >= n);

	const int pid = threadIdx.x + blockDim.x * blockIdx.x;

	if (pid >= n)
	    return;

	Particle p = src[pid];

	dst[pid] = make_float4(p.x[0], p.x[1], p.x[2], 0);
    }

    __device__ bool handle_collision(float& x, float& y, float& z, float& u, float& v, float& w, const int rank, const double dt)
    {
	const float initial_sdf = sdf(x, y, z);

	if (initial_sdf < 0)
	    return false;

	const float xold = x - dt * u;
	const float yold = y - dt * v;
	const float zold = z - dt * w;

	if (sdf(xold, yold, zold) >= 0)
	{
	    //this is the worst case - it means that old position was bad already
	    //we need to rescue the particle, extracting it from the walls

	    const float3 mygrad = grad_sdf(x, y, z);
	    const float mysdf = sdf(x, y, z);

	    x -= mysdf * mygrad.x;
	    y -= mysdf * mygrad.y;
	    z -= mysdf * mygrad.z;

	    for(int l = 10; l >= 1; --l)
	    {
		if (sdf(x, y, z) < 0)
		{
		    u  = -u;
		    v  = -v;
		    w  = -w;

		    return false;
		}

		const float jump = 1.1f * mysdf / (1 << l);

		x -= jump * mygrad.x;
		y -= jump * mygrad.y;
		z -= jump * mygrad.z;
	    }

	    cuda_printf("RANK %d bounce collision failed OLD: %f %f %f, sdf %e \nNEW: %f %f %f sdf %e, gradient %f %f %f\n",
			rank,
			xold, yold, zold, sdf(xold, yold, zold),
			x, y, z, sdf(x, y, z), mygrad.x, mygrad.y, mygrad.z);

	    return false;
	}

	float subdt = 0;

	for(int i = 1; i < 8; ++i)
	{
	    const float tcandidate = subdt + dt / (1 << i);
	    const float xcandidate = xold + tcandidate * u;
	    const float ycandidate = yold + tcandidate * v;
	    const float zcandidate = zold + tcandidate * w;

	    if (sdf(xcandidate, ycandidate, zcandidate) < 0)
		subdt = tcandidate;
	}

	const float lambda = 2 * subdt - dt;

	x = xold + lambda * u;
	y = yold + lambda * v;
	z = zold + lambda * w;

	u  = -u;
	v  = -v;
	w  = -w;

	if (sdf(x, y, z) >= 0)
	{
	    x = xold;
	    y = yold;
	    z = zold;

	    assert(sdf(x, y, z) < 0);
	}

	return true;
    }

    __global__ void bounce(Particle * const particles, const int n, const int rank, const float dt)
    {
	assert(blockDim.x * gridDim.x >= n);

	const int pid = threadIdx.x + blockDim.x * blockIdx.x;

	if (pid >= n)
	    return;

	Particle p = particles[pid];

	const int L[3] = { XSIZE_SUBDOMAIN, YSIZE_SUBDOMAIN, ZSIZE_SUBDOMAIN };
	const int MARGIN[3] = { XMARGIN_WALL, YMARGIN_WALL, ZMARGIN_WALL };

	for(int c = 0; c < 3; ++c)
	{
	    if (!(abs(p.x[c]) <= L[c]/2 + MARGIN[c]))
		cuda_printf("bounce: ooooooooops component %d we have %f %f %f outside %d + %d\n", c, p.x[0], p.x[1], p.x[2], L[c]/2, MARGIN[c]);

	    assert(abs(p.x[c]) <= L[c]/2 + MARGIN[c]);
	}

	if (handle_collision(p.x[0], p.x[1], p.x[2], p.u[0], p.u[1], p.u[2], rank, dt))
	    particles[pid] = p;
    }

#define _XCPB_ 2
#define _YCPB_ 2
#define _ZCPB_ 1
#define CPB (_XCPB_ * _YCPB_ * _ZCPB_)

    __global__ __launch_bounds__(32 * CPB, 16)
	void interactions(const float2 * particles, const int * const cellsstart, const int * const cellscount,
			  const float seed, const float sigmaf, float * const axayaz)
    {
	const int COLS = 32;
	const int ROWS = 1;
	assert(warpSize == COLS * ROWS);
	assert(blockDim.x == warpSize && blockDim.y == CPB && blockDim.z == 1);
	assert(ROWS * 3 <= warpSize);

	const int tid = threadIdx.x;
	const int subtid = tid % COLS;
	const int slot = tid / COLS;
	const int wid = threadIdx.y;

	__shared__ int volatile starts[CPB][32], scan[CPB][32];

	const int xdestcid = blockIdx.x * _XCPB_ + ((threadIdx.y) % _XCPB_);
	const int ydestcid = blockIdx.y * _YCPB_ + ((threadIdx.y / _XCPB_) % _YCPB_);
	const int zdestcid = blockIdx.z * _ZCPB_ + ((threadIdx.y / (_XCPB_ * _YCPB_)) % _ZCPB_);

	int mycount = 0, myscan = 0;

	if (tid < 27)
	{
	    const int dx = (tid) % 3;
	    const int dy = ((tid / 3)) % 3;
	    const int dz = ((tid / 9)) % 3;

	    int xcid = XMARGIN_WALL + xdestcid + dx - 1;
	    int ycid = YMARGIN_WALL + ydestcid + dy - 1;
	    int zcid = ZMARGIN_WALL + zdestcid + dz - 1;

	    const bool valid_cid =
		xcid >= 0 && xcid < XSIZE_WALLCELLS &&
		ycid >= 0 && ycid < YSIZE_WALLCELLS &&
		zcid >= 0 && zcid < ZSIZE_WALLCELLS ;

	    xcid = min(XSIZE_WALLCELLS - 1, max(0, xcid));
	    ycid = min(YSIZE_WALLCELLS - 1, max(0, ycid));
	    zcid = min(ZSIZE_WALLCELLS - 1, max(0, zcid));

	    const int cid = max(0, xcid + XSIZE_WALLCELLS * (ycid + XSIZE_WALLCELLS * zcid));

	    starts[wid][tid] = tex1Dfetch(texWallCellStart, cid);

	    myscan = mycount = valid_cid * tex1Dfetch(texWallCellCount, cid);
	}

	for(int L = 1; L < 32; L <<= 1)
	    myscan += (tid >= L) * __shfl_up(myscan, L) ;

	if (tid < 28)
	    scan[wid][tid] = myscan - mycount;

	const int destcid = xdestcid + XSIZE_SUBDOMAIN * (ydestcid + YSIZE_SUBDOMAIN * zdestcid);
	const int dststart = cellsstart[destcid];
	const int ndst = cellscount[destcid];
	const int nsrc = scan[wid][27];

	for(int d = 0; d < ndst; d += ROWS)
	{
	    const int np1 = min(ndst - d, ROWS);

	    const int dpid = dststart + d + slot;
	    const int entry = 3 * dpid;
	    float2 dtmp0 = particles[entry + 0];
	    float2 dtmp1 = particles[entry + 1];
	    float2 dtmp2 = particles[entry + 2];

	    float xforce = 0, yforce = 0, zforce = 0;

	    for(int s = 0; s < nsrc; s += COLS)
	    {
		const int np2 = min(nsrc - s, COLS);

		const int pid = s + subtid;
		const int key9 = 9 * ((pid >= scan[wid][9]) + (pid >= scan[wid][18]));
		const int key3 = 3 * ((pid >= scan[wid][key9 + 3]) + (pid >= scan[wid][key9 + 6]));
		const int key = key9 + key3;

		const int spid = pid - scan[wid][key] + starts[wid][key];
		const float4 stmp0 = tex1Dfetch(texWallParticles, spid);

		{
		    const float xdiff = dtmp0.x - stmp0.x;
		    const float ydiff = dtmp0.y - stmp0.y;
		    const float zdiff = dtmp1.x - stmp0.z;

		    const float _xr = xdiff;
		    const float _yr = ydiff;
		    const float _zr = zdiff;

		    const float rij2 = _xr * _xr + _yr * _yr + _zr * _zr;
		    const float invrij = rsqrtf(rij2);
		    const float rij = rij2 * invrij;
		    const float argwr = max((float)0, 1 - rij);

		    const float wr = viscosity_function<-VISCOSITY_S_LEVEL>(argwr);

		    const float xr = _xr * invrij;
		    const float yr = _yr * invrij;
		    const float zr = _zr * invrij;

		    const float rdotv =
			xr * (dtmp1.y - 0) +
			yr * (dtmp2.x - 0) +
			zr * (dtmp2.y - 0);

		    const float myrandnr = Logistic::mean0var1(seed, min(spid, dpid), max(spid, dpid));

		    const float strength = aij * argwr - (gammadpd * wr * rdotv + sigmaf * myrandnr) * wr;
		    const bool valid = (slot < np1) && (subtid < np2);

		    if (valid)
		    {
			xforce += strength * xr;
			yforce += strength * yr;
			zforce += strength * zr;
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
		axayaz[c + 3 * dstpid] += fcontrib;
	}
    }


    __global__ void interactions_old(const Particle * const particles, const int np, const int nsolid,
				     Acceleration * const acc, const float seed, const float sigmaf)
    {
	assert(blockDim.x * gridDim.x >= np);

       	const int pid = threadIdx.x + blockDim.x * blockIdx.x;

	if (pid >= np)
	    return;

	Particle p = particles[pid];

	const int L[3] = { XSIZE_SUBDOMAIN, YSIZE_SUBDOMAIN, ZSIZE_SUBDOMAIN };
	const int MARGIN[3] = { XMARGIN_WALL, YMARGIN_WALL, ZMARGIN_WALL };

	int base[3];
	for(int c = 0; c < 3; ++c)
	{
	    assert(p.x[c] >= -L[c]/2 - MARGIN[c]);
	    assert(p.x[c] < L[c]/2 + MARGIN[c]);
	    base[c] = (int)(p.x[c] - (-L[c]/2 - MARGIN[c]));
	}

	const float xp = p.x[0], yp = p.x[1], zp = p.x[2];
	const float up = p.u[0], vp = p.u[1], wp = p.u[2];

	float xforce = 0, yforce = 0, zforce = 0;

	for(int code = 0; code < 27; ++code)
	{
	    const int xcid = base[0] + (code % 3) - 1;
	    const int ycid = base[1] + (code/3 % 3) - 1;
	    const int zcid = base[2] + (code/9 % 3) - 1;

	    if (xcid < 0 || xcid >= XSIZE_SUBDOMAIN + 2 * XMARGIN_WALL ||
		ycid < 0 || ycid >= YSIZE_SUBDOMAIN + 2 * YMARGIN_WALL ||
		zcid < 0 || zcid >= ZSIZE_SUBDOMAIN + 2 * ZMARGIN_WALL )
		continue;

	    const int cid = xcid +
		(XSIZE_SUBDOMAIN + 2 * XMARGIN_WALL) *
		(ycid + (YSIZE_SUBDOMAIN + 2 * YMARGIN_WALL) * zcid);

	    assert(cid >= 0 && cid < (XSIZE_SUBDOMAIN + 2 * XMARGIN_WALL) *
		   (YSIZE_SUBDOMAIN + 2 * YMARGIN_WALL) *
		   (ZSIZE_SUBDOMAIN + 2 * ZMARGIN_WALL));

	    const int start = tex1Dfetch(texWallCellStart, cid);
	    const int stop = start + tex1Dfetch(texWallCellCount, cid);

	    assert(start >= 0 && stop <= nsolid && start <= stop);

	    for(int s = start; s < stop; ++s)
	    {
		const float4 stmp0 = tex1Dfetch(texWallParticles, s);

		const float xq = stmp0.x;
		const float yq = stmp0.y;
		const float zq = stmp0.z;

	    	const float _xr = xp - xq;
		const float _yr = yp - yq;
		const float _zr = zp - zq;

		const float rij2 = _xr * _xr + _yr * _yr + _zr * _zr;

		const float invrij = rsqrtf(rij2);

		const float rij = rij2 * invrij;
		const float argwr = max((float)0, 1 - rij);
		const float wr = viscosity_function<-VISCOSITY_S_LEVEL>(argwr);

		const float xr = _xr * invrij;
		const float yr = _yr * invrij;
		const float zr = _zr * invrij;

		const float rdotv =
		    xr * (up - 0) +
		    yr * (vp - 0) +
		    zr * (wp - 0);

		const float myrandnr = Logistic::mean0var1(seed, pid, s);

		const float strength = aij * argwr + (- gammadpd * wr * rdotv + sigmaf * myrandnr) * wr;

		xforce += strength * xr;
		yforce += strength * yr;
		zforce += strength * zr;
	    }
	}

	acc[pid].a[0] += xforce;
	acc[pid].a[1] += yforce;
	acc[pid].a[2] += zforce;

	for(int c = 0; c < 3; ++c)
	    assert(!isnan(acc[pid].a[c]));
    }
}

template<int k>
struct Bspline
{
    template<int i>
    static float eval(float x)
	{
	    return
		(x - i) / (k - 1) * Bspline<k - 1>::template eval<i>(x) +
		(i + k - x) / (k - 1) * Bspline<k - 1>::template eval<i + 1>(x);
	}
};

template<>
struct Bspline<1>
{
    template <int i>
    static float eval(float x)
	{
	    return  (float)(i) <= x && x < (float)(i + 1);
	}
};

struct FieldSampler
{
    float * data, extent[3];
    int N[3];

    FieldSampler(const char * path, MPI_Comm comm, const bool verbose)
	{
	    NVTX_RANGE("WALL/load-file", NVTX_C3);

	    if (verbose)
		printf("reading header...\n");

	    char header[2048];

	    MPI_File fh;
	    MPI_CHECK( MPI_File_open(comm, path, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh) );

	    MPI_Status status;
	    MPI_CHECK( MPI_File_read_at_all(fh, 0, header, sizeof(header), MPI_CHAR, &status));

	    const int retval = sscanf(header, "%f %f %f\n%d %d %d\n", extent + 0, extent + 1, extent + 2, N + 0, N + 1, N + 2);

	    if(retval != 6)
	    {
		printf("ooops something went wrong in reading %s.\n", path);
		exit(EXIT_FAILURE);
	    }

	    if (verbose)
		printf("allocating data...\n");

	    const int nvoxels = N[0] * N[1] * N[2];

	    data = new float[nvoxels];

	    if(data == NULL)
	    {
		printf("ooops bad allocation %s.\n", path);
		exit(EXIT_FAILURE);
	    }

	    int header_size = 0;

	    for(int i = 0; i < sizeof(header); ++i)
		if (header[i] == '\n')
		{
		    if (header_size > 0)
		    {
			header_size = i + 1;
			break;
		    }
		    
		    header_size = i + 1;
		}

	    if (verbose)
		printf("reading binary data... from byte %d\n", header_size);

	    MPI_CHECK( MPI_File_read_at_all(fh, header_size, data, nvoxels, MPI_FLOAT, &status));

	    MPI_CHECK( MPI_File_close(&fh));
	}

    void sample(const float start[3], const float spacing[3], const int nsize[3],
		float * const output, const float amplitude_rescaling)
	{
	    NVTX_RANGE("WALL/sample", NVTX_C7);

	    Bspline<4> bsp;

	    for(int iz = 0; iz < nsize[2]; ++iz)
		for(int iy = 0; iy < nsize[1]; ++iy)
		    for(int ix = 0; ix < nsize[0]; ++ix)
		    {
			const float x[3] = {
			    start[0] + (ix  + 0.5f) * spacing[0] - 0.5f,
			    start[1] + (iy  + 0.5f) * spacing[1] - 0.5f,
			    start[2] + (iz  + 0.5f) * spacing[2] - 0.5f
			};

			int anchor[3];
			for(int c = 0; c < 3; ++c)
			    anchor[c] = (int)floor(x[c]);

			float w[3][4];
			for(int c = 0; c < 3; ++c)
			    for(int i = 0; i < 4; ++i)
				w[c][i] = bsp.eval<0>(x[c] - (anchor[c] - 1 + i) + 2);

			float tmp[4][4];
			for(int sz = 0; sz < 4; ++sz)
			    for(int sy = 0; sy < 4; ++sy)
			    {
				float s = 0;

				for(int sx = 0; sx < 4; ++sx)
				{
				    const int l[3] = {sx, sy, sz};

				    int g[3];
				    for(int c = 0; c < 3; ++c)
					g[c] = max(0, min(N[c] - 1, l[c] - 1 + anchor[c]));

				    s += w[0][sx] * data[g[0] + N[0] * (g[1] + N[1] * g[2])];
				}

				tmp[sz][sy] = s;
			    }

			float partial[4];
			for(int sz = 0; sz < 4; ++sz)
			{
			    float s = 0;

			    for(int sy = 0; sy < 4; ++sy)
				s += w[1][sy] * tmp[sz][sy];

			    partial[sz] = s;
			}

			float val = 0;
			for(int sz = 0; sz < 4; ++sz)
			    val += w[2][sz] * partial[sz];

			output[ix + nsize[0] * (iy + nsize[1] * iz)] = val * amplitude_rescaling;
		    }
	}

    ~FieldSampler()
	{
	    delete [] data;
	}
};

ComputeInteractionsWall::ComputeInteractionsWall(MPI_Comm cartcomm, Particle* const p, const int n, int& nsurvived,
						 ExpectedMessageSizes& new_sizes, const bool verbose):
    cartcomm(cartcomm), arrSDF(NULL), solid4(NULL), solid_size(0),
    cells(XSIZE_SUBDOMAIN + 2 * XMARGIN_WALL, YSIZE_SUBDOMAIN + 2 * YMARGIN_WALL, ZSIZE_SUBDOMAIN + 2 * ZMARGIN_WALL)
{
    MPI_CHECK( MPI_Comm_rank(cartcomm, &myrank));

    MPI_CHECK( MPI_Cart_get(cartcomm, 3, dims, periods, coords) );

    float * field = new float[ XTEXTURESIZE * YTEXTURESIZE * ZTEXTURESIZE];

    FieldSampler sampler("sdf.dat", cartcomm, verbose);

    const int L[3] = { XSIZE_SUBDOMAIN, YSIZE_SUBDOMAIN, ZSIZE_SUBDOMAIN };
    const int MARGIN[3] = { XMARGIN_WALL, YMARGIN_WALL, ZMARGIN_WALL };
    const int TEXTURESIZE[3] = { XTEXTURESIZE, YTEXTURESIZE, ZTEXTURESIZE };

#ifndef NDEBUG
    assert(fabs(dims[0] * XSIZE_SUBDOMAIN / (double) (dims[1] * YSIZE_SUBDOMAIN) - sampler.extent[0] / (double)sampler.extent[1]) < 1e-5);
    assert(fabs(dims[0] * XSIZE_SUBDOMAIN / (double) (dims[2] * ZSIZE_SUBDOMAIN) - sampler.extent[0] / (double)sampler.extent[2]) < 1e-5);
#endif

    if (myrank == 0)
	printf("sampling the geometry file...\n");

    {
       	float start[3], spacing[3];
	for(int c = 0; c < 3; ++c)
	{
	    start[c] = sampler.N[c] * (coords[c] * L[c] - MARGIN[c]) / (float)(dims[c] * L[c]) ;
	    spacing[c] =  sampler.N[c] * (L[c] + 2 * MARGIN[c]) / (float)(dims[c] * L[c]) / (float) TEXTURESIZE[c];
	}

	const float amplitude_rescaling = (XSIZE_SUBDOMAIN /*+ 2 * XMARGIN_WALL*/) / (sampler.extent[0] / dims[0]) ;

	sampler.sample(start, spacing, TEXTURESIZE, field, amplitude_rescaling);
    }

    if (myrank == 0)
	printf("redistancing the geometry field...\n");

    //extra redistancing because margin might exceed the domain
    {
	NVTX_RANGE("WALL/redistancing", NVTX_C4);

	const double dx =  (XSIZE_SUBDOMAIN + 2 * XMARGIN_WALL) / (double)XTEXTURESIZE;
	const double dy =  (YSIZE_SUBDOMAIN + 2 * YMARGIN_WALL) / (double)YTEXTURESIZE;
	const double dz =  (ZSIZE_SUBDOMAIN + 2 * ZMARGIN_WALL) / (double)ZTEXTURESIZE;

	redistancing(field, XTEXTURESIZE, YTEXTURESIZE, ZTEXTURESIZE, dx, dy, dz, XTEXTURESIZE * 2);
    }

#ifndef NO_VTK
    {
	if (myrank == 0)
	    printf("writing to VTK file..\n");

	vtkImageData * img = vtkImageData::New();

	img->SetExtent(0, XTEXTURESIZE-1, 0, YTEXTURESIZE-1, 0, ZTEXTURESIZE-1);
	img->SetDimensions(XTEXTURESIZE, YTEXTURESIZE, ZTEXTURESIZE);
	img->AllocateScalars(VTK_FLOAT, 1);

	const float dx = (XSIZE_SUBDOMAIN + 2 * XMARGIN_WALL) / (float)XTEXTURESIZE;
	const float dy = (YSIZE_SUBDOMAIN + 2 * YMARGIN_WALL) / (float)YTEXTURESIZE;
	const float dz = (ZSIZE_SUBDOMAIN + 2 * ZMARGIN_WALL) / (float)ZTEXTURESIZE;

	const float x0 = -XSIZE_SUBDOMAIN / 2 - XMARGIN_WALL;
	const float y0 = -YSIZE_SUBDOMAIN / 2 - YMARGIN_WALL;
	const float z0 = -ZSIZE_SUBDOMAIN / 2 - ZMARGIN_WALL;

	img->SetSpacing(dx, dy, dz);
	img->SetOrigin(x0, y0, z0);

	for(int iz=0; iz<ZTEXTURESIZE; iz++)
	    for(int iy=0; iy<YTEXTURESIZE; iy++)
		for(int ix=0; ix<XTEXTURESIZE; ix++)
		    img->SetScalarComponentFromFloat(ix, iy, iz, 0,  field[ix + XTEXTURESIZE * (iy + YTEXTURESIZE * iz)]);

	vtkXMLImageDataWriter * writer = vtkXMLImageDataWriter::New();
	char buf[1024];
	sprintf(buf, "redistancing-rank%d.vti", myrank);
	writer->SetFileName(buf);
	writer->SetInputData(img);
	writer->Write();

	writer->Delete();
	img->Delete();
    }
#endif

    if (myrank == 0)
	printf("estimating geometry-based message sizes...\n");

    {
	for(int dz = -1; dz <= 1; ++dz)
	    for(int dy = -1; dy <= 1; ++dy)
		for(int dx = -1; dx <= 1; ++dx)
		{
		    const int d[3] = { dx, dy, dz };
		    const int entry = (dx + 1) + 3 * ((dy + 1) + 3 * (dz + 1));

		    int local_start[3] = {
			d[0] + (d[0] == 1) * (XSIZE_SUBDOMAIN - 2),
			d[1] + (d[1] == 1) * (YSIZE_SUBDOMAIN - 2),
			d[2] + (d[2] == 1) * (ZSIZE_SUBDOMAIN - 2)
		    };

		    int local_extent[3] = {
			1 * (d[0] != 0 ? 2 : XSIZE_SUBDOMAIN),
			1 * (d[1] != 0 ? 2 : YSIZE_SUBDOMAIN),
			1 * (d[2] != 0 ? 2 : ZSIZE_SUBDOMAIN)
		    };

		    float start[3], spacing[3];
		    for(int c = 0; c < 3; ++c)
		    {
			start[c] = (coords[c] * L[c] + local_start[c]) / (float)(dims[c] * L[c]) * sampler.N[c];
			spacing[c] = sampler.N[c] / (float)(dims[c] * L[c]) ;
		    }

		    const int nextent = local_extent[0] * local_extent[1] * local_extent[2];
		    float * data = new float[nextent];

		    sampler.sample(start, spacing, local_extent, data, 1);

		    int s = 0;
		    for(int i = 0; i < nextent; ++i)
			s += (data[i] < 0);

		    delete [] data;
		    double avgsize = ceil(s * numberdensity / (double)pow(2, abs(d[0]) + abs(d[1]) + abs(d[2])));

		    new_sizes.msgsizes[entry] = (int)avgsize;

		}
    }

    if (hdf5field_dumps)
    {
	NVTX_RANGE("WALL/h5-dump", NVTX_C4);

	if (myrank == 0)
	    printf("H5 data dump of the geometry...\n");

	float * walldata = new float[XSIZE_SUBDOMAIN * YSIZE_SUBDOMAIN * ZSIZE_SUBDOMAIN];

	float start[3], spacing[3];
	for(int c = 0; c < 3; ++c)
	{
	    start[c] = coords[c] * L[c] / (float)(dims[c] * L[c]) * sampler.N[c];
	    spacing[c] = sampler.N[c] / (float)(dims[c] * L[c]) ;
	}

	int size[3] = {XSIZE_SUBDOMAIN, YSIZE_SUBDOMAIN, ZSIZE_SUBDOMAIN};

	const float amplitude_rescaling = L[0] / (sampler.extent[0] / dims[0]);
	sampler.sample(start, spacing, size, walldata, amplitude_rescaling);

	H5FieldDump dump(cartcomm);
	dump.dump_scalarfield(cartcomm, walldata, "wall");

	delete [] walldata;
    }

    CUDA_CHECK(cudaPeekAtLastError());

    cudaChannelFormatDesc fmt = cudaCreateChannelDesc<float>();
    CUDA_CHECK(cudaMalloc3DArray (&arrSDF, &fmt, make_cudaExtent(XTEXTURESIZE, YTEXTURESIZE, ZTEXTURESIZE)));

    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr   = make_cudaPitchedPtr((void *)field, XTEXTURESIZE * sizeof(float), XTEXTURESIZE, YTEXTURESIZE);
    copyParams.dstArray = arrSDF;
    copyParams.extent   = make_cudaExtent(XTEXTURESIZE, YTEXTURESIZE, ZTEXTURESIZE);
    copyParams.kind     = cudaMemcpyHostToDevice;
    CUDA_CHECK(cudaMemcpy3D(&copyParams));
    delete [] field;

    SolidWallsKernel::setup();

    CUDA_CHECK(cudaBindTextureToArray(SolidWallsKernel::texSDF, arrSDF, fmt));

    if (myrank == 0)
	printf("carving out wall particles...\n");

    thrust::device_vector<int> keys(n);

    SolidWallsKernel::fill_keys<<< (n + 127) / 128, 128 >>>(p, n, thrust::raw_pointer_cast(&keys[0]));
    CUDA_CHECK(cudaPeekAtLastError());

    thrust::sort_by_key(keys.begin(), keys.end(), thrust::device_ptr<Particle>(p));

    nsurvived = thrust::count(keys.begin(), keys.end(), 0);
    assert(nsurvived <= n);

    const int nbelt = thrust::count(keys.begin() + nsurvived, keys.end(), 1);

    thrust::device_vector<Particle> solid_local(thrust::device_ptr<Particle>(p + nsurvived), thrust::device_ptr<Particle>(p + nsurvived + nbelt));

    if (hdf5part_dumps)
    {
	const int n = solid_local.size();

	Particle * phost = new Particle[n];

	CUDA_CHECK(cudaMemcpy(phost, thrust::raw_pointer_cast(&solid_local[0]), sizeof(Particle) * n, cudaMemcpyDeviceToHost));

       	H5PartDump solid_dump("solid-walls.h5part", cartcomm, cartcomm);
	solid_dump.dump(phost, n);

	delete [] phost;
    }

    //can't use halo-exchanger class because of MARGIN
    //HaloExchanger halo(cartcomm, L, 666);
    //SimpleDeviceBuffer<Particle> solid_remote;
    //halo.exchange(thrust::raw_pointer_cast(&solid_local[0]), solid_local.size(), solid_remote);

    if (myrank == 0)
	printf("fetching remote wall particles in my proximity...\n");

    SimpleDeviceBuffer<Particle> solid_remote;

    {
	NVTX_RANGE("WALL/exchange-particles", NVTX_C3)

	    thrust::host_vector<Particle> local = solid_local;

	int dstranks[26], remsizes[26], recv_tags[26];
	for(int i = 0; i < 26; ++i)
	{
	    const int d[3] = { (i + 2) % 3 - 1, (i / 3 + 2) % 3 - 1, (i / 9 + 2) % 3 - 1 };

	    recv_tags[i] = (2 - d[0]) % 3 + 3 * ((2 - d[1]) % 3 + 3 * ((2 - d[2]) % 3));

	    int coordsneighbor[3];
	    for(int c = 0; c < 3; ++c)
		coordsneighbor[c] = coords[c] + d[c];

	    MPI_CHECK( MPI_Cart_rank(cartcomm, coordsneighbor, dstranks + i) );
	}

	//send local counts - receive remote counts
	{
	    for(int i = 0; i < 26; ++i)
		remsizes[i] = -1;

	    MPI_Request reqrecv[26];
	    for(int i = 0; i < 26; ++i)
		MPI_CHECK( MPI_Irecv(remsizes + i, 1, MPI_INTEGER, dstranks[i], 123 + recv_tags[i], cartcomm, reqrecv + i) );

	    const int localsize = local.size();

	    MPI_Request reqsend[26];
	    for(int i = 0; i < 26; ++i)
		MPI_CHECK( MPI_Isend(&localsize, 1, MPI_INTEGER, dstranks[i], 123 + i, cartcomm, reqsend + i) );

	    MPI_Status statuses[26];
	    MPI_CHECK( MPI_Waitall(26, reqrecv, statuses) );
	    MPI_CHECK( MPI_Waitall(26, reqsend, statuses) );

	    for(int i = 0; i < 26; ++i)
		assert(remsizes[i] >= 0);
	}

	std::vector<Particle> remote[26];

	//send local data - receive remote data
	{
	    for(int i = 0; i < 26; ++i)
		remote[i].resize(remsizes[i]);

	    MPI_Request reqrecv[26];
	    for(int i = 0; i < 26; ++i)
		MPI_CHECK( MPI_Irecv(remote[i].data(), remote[i].size() * 6, MPI_FLOAT, dstranks[i], 321 + recv_tags[i], cartcomm, reqrecv + i) );

	    MPI_Request reqsend[26];
	    for(int i = 0; i < 26; ++i)
		MPI_CHECK( MPI_Isend(local.data(), local.size() * 6, MPI_FLOAT, dstranks[i], 321 + i, cartcomm, reqsend + i) );

	    MPI_Status statuses[26];
	    MPI_CHECK( MPI_Waitall(26, reqrecv, statuses) );
	    MPI_CHECK( MPI_Waitall(26, reqsend, statuses) );
	}

	//select particles within my region [-L / 2 - MARGIN, +L / 2 + MARGIN]
	std::vector<Particle> selected;
	for(int i = 0; i < 26; ++i)
	{
	    const int d[3] = { (i + 2) % 3 - 1, (i / 3 + 2) % 3 - 1, (i / 9 + 2) % 3 - 1 };

	    for(int j = 0; j < remote[i].size(); ++j)
	    {
		Particle p = remote[i][j];

		for(int c = 0; c < 3; ++c)
		    p.x[c] += d[c] * L[c];

		bool inside = true;

		for(int c = 0; c < 3; ++c)
		    inside &= p.x[c] >= -L[c] / 2 - MARGIN[c] && p.x[c] < L[c] / 2 + MARGIN[c];

		if (inside)
		    selected.push_back(p);
	    }
	}

	solid_remote.resize(selected.size());
	CUDA_CHECK(cudaMemcpy(solid_remote.data, selected.data(), sizeof(Particle) * solid_remote.size, cudaMemcpyHostToDevice));
    }

    solid_size = solid_local.size() + solid_remote.size;

    Particle * solid;
    CUDA_CHECK(cudaMalloc(&solid, sizeof(Particle) * solid_size));
    CUDA_CHECK(cudaMemcpy(solid, thrust::raw_pointer_cast(&solid_local[0]), sizeof(Particle) * solid_local.size(), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(solid + solid_local.size(), solid_remote.data, sizeof(Particle) * solid_remote.size, cudaMemcpyDeviceToDevice));

    if (solid_size > 0)
	cells.build(solid, solid_size, 0);

    CUDA_CHECK(cudaMalloc(&solid4, sizeof(float4) * solid_size));

    if (myrank == 0)
	printf("consolidating wall particles...\n");

    if (solid_size > 0)
	SolidWallsKernel::strip_solid4<<< (solid_size + 127) / 128, 128>>>(solid, solid_size, solid4);

    CUDA_CHECK(cudaFree(solid));

    CUDA_CHECK(cudaPeekAtLastError());
}

void ComputeInteractionsWall::bounce(Particle * const p, const int n, cudaStream_t stream)
{
    NVTX_RANGE("WALL/bounce", NVTX_C3)

	if (n > 0)
	    SolidWallsKernel::bounce<<< (n + 127) / 128, 128, 0, stream>>>(p, n, myrank, dt);

    CUDA_CHECK(cudaPeekAtLastError());
}

void ComputeInteractionsWall::interactions(const Particle * const p, const int n, Acceleration * const acc,
					   const int * const cellsstart, const int * const cellscount, cudaStream_t stream)
{
    NVTX_RANGE("WALL/interactions", NVTX_C3)
	//cellsstart and cellscount IGNORED for now

	if (n > 0 && solid_size > 0)
	{
	    size_t textureoffset;
	    CUDA_CHECK(cudaBindTexture(&textureoffset, &SolidWallsKernel::texWallParticles, solid4,
				       &SolidWallsKernel::texWallParticles.channelDesc, sizeof(float4) * solid_size));
	    assert(textureoffset == 0);

	    CUDA_CHECK(cudaBindTexture(&textureoffset, &SolidWallsKernel::texWallCellStart, cells.start,
				       &SolidWallsKernel::texWallCellStart.channelDesc, sizeof(int) * cells.ncells));
	    assert(textureoffset == 0);

	    CUDA_CHECK(cudaBindTexture(&textureoffset, &SolidWallsKernel::texWallCellCount, cells.count,
				       &SolidWallsKernel::texWallCellCount.channelDesc, sizeof(int) * cells.ncells));
	    assert(textureoffset == 0);

#if 0
	    SolidWallsKernel::interactions<<<
		dim3(XSIZE_SUBDOMAIN / _XCPB_, YSIZE_SUBDOMAIN / _YCPB_, ZSIZE_SUBDOMAIN / _ZCPB_), dim3(32, CPB), 0, stream>>>(
		    (float2 *)p, cellsstart, cellscount, trunk.get_float(), sigmaf, &acc->a[0]);
#else
	    SolidWallsKernel::interactions_old<<< (n + 127) / 128, 128, 0, stream>>>
		(p, n, solid_size, acc, trunk.get_float(), sigmaf);
#endif

	    CUDA_CHECK(cudaUnbindTexture(SolidWallsKernel::texWallParticles));
	    CUDA_CHECK(cudaUnbindTexture(SolidWallsKernel::texWallCellStart));
	    CUDA_CHECK(cudaUnbindTexture(SolidWallsKernel::texWallCellCount));
	}

    CUDA_CHECK(cudaPeekAtLastError());
}

ComputeInteractionsWall::~ComputeInteractionsWall()
{
    CUDA_CHECK(cudaUnbindTexture(SolidWallsKernel::texSDF));
    CUDA_CHECK(cudaFreeArray(arrSDF));
}
