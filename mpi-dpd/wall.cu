/*
 *  wall.cu
 *  Part of uDeviceX/mpi-dpd/
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
#include "solvent-exchange.h"
#include "wall.h"
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

    template<bool computestresses>
    __global__ void interactions_3tpp(const float2 * const particles, const int np, const int nsolid,
				      float * const acc, const float seed, const float sigmaf, const float xvel, const float y0);

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

        CUDA_CHECK(cudaFuncSetCacheConfig(interactions_3tpp<true>, cudaFuncCachePreferL1));
	CUDA_CHECK(cudaFuncSetCacheConfig(interactions_3tpp<false>, cudaFuncCachePreferL1));
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

    __device__ float cheap_sdf(float x, float y, float z) //within the rescaled texel width error
    {
        const int L[3] = { XSIZE_SUBDOMAIN, YSIZE_SUBDOMAIN, ZSIZE_SUBDOMAIN };
        const int MARGIN[3] = { XMARGIN_WALL, YMARGIN_WALL, ZMARGIN_WALL };
        const int TEXSIZES[3] = {XTEXTURESIZE, YTEXTURESIZE, ZTEXTURESIZE };

        float p[3] = {x, y, z};

        float texcoord[3];
        for(int c = 0; c < 3; ++c)
            texcoord[c] = 0.5001f + (int)(TEXSIZES[c] * (p[c] + L[c] / 2 + MARGIN[c]) / (L[c] + 2 * MARGIN[c]));

        return tex3D(texSDF, texcoord[0], texcoord[1], texcoord[2]);
    }

    __device__ float3 ugrad_sdf(float x, float y, float z)
    {
        const int L[3] = { XSIZE_SUBDOMAIN, YSIZE_SUBDOMAIN, ZSIZE_SUBDOMAIN };
        const int MARGIN[3] = { XMARGIN_WALL, YMARGIN_WALL, ZMARGIN_WALL };
        const int TEXSIZES[3] = {XTEXTURESIZE, YTEXTURESIZE, ZTEXTURESIZE };
        const float p[3] = {x, y, z};

        float tc[3];
        for(int c = 0; c < 3; ++c)
            tc[c] = 0.5001f + (int)(TEXSIZES[c] * (p[c] + L[c] / 2 + MARGIN[c]) / (L[c] + 2 * MARGIN[c]));

        float factors[3];
        for(int c = 0; c < 3; ++c)
            factors[c] = TEXSIZES[c] / (2 * MARGIN[c] + L[c]);

        float myval = tex3D(texSDF, tc[0], tc[1], tc[2]);
        float xmygrad = factors[0] * (tex3D(texSDF, tc[0] + 1, tc[1], tc[2]) - myval);
        float ymygrad = factors[1] * (tex3D(texSDF, tc[0], tc[1] + 1, tc[2]) - myval);
        float zmygrad = factors[2] * (tex3D(texSDF, tc[0], tc[1], tc[2] + 1) - myval);

        return make_float3(xmygrad, ymygrad, zmygrad);
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

        float xmygrad = (tex3D(texSDF, tc[0] + 1, tc[1], tc[2]) - tex3D(texSDF, tc[0] - 1, tc[1], tc[2]));
        float ymygrad = (tex3D(texSDF, tc[0], tc[1] + 1, tc[2]) - tex3D(texSDF, tc[0], tc[1] - 1, tc[2]));
        float zmygrad = (tex3D(texSDF, tc[0], tc[1], tc[2] + 1) - tex3D(texSDF, tc[0], tc[1], tc[2] - 1));

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

    __device__ void handle_collision(const float currsdf, float& x, float& y, float& z, float& u, float& v, float& w, const int rank, const float dt)
    {
        assert(currsdf >= 0);

        const float xold = x - dt * u;
        const float yold = y - dt * v;
        const float zold = z - dt * w;

        if (sdf(xold, yold, zold) >= 0)
        {
            //this is the worst case - it means that old position was bad already
            //we need to search and rescue the particle

            cuda_printf("Warning rank %d sdf: %f (%.4f %.4f %.4f), from: sdf %f (%.4f %.4f %.4f)...   ",
                    rank, currsdf, x, y, z, sdf(xold, yold, zold), xold, yold, zold);

            const float3 mygrad = grad_sdf(x, y, z);
            const float mysdf = currsdf;

            x -= mysdf * mygrad.x;
            y -= mysdf * mygrad.y;
            z -= mysdf * mygrad.z;

            for(int l = 8; l >= 1; --l)
            {
                if (sdf(x, y, z) < 0)
                {
                    u  = -u;
                    v  = -v;
                    w  = -w;

                    cuda_printf("rescued in %d steps\n", 9-l);

                    return;
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

            return;
        }

        //newton raphson steps
        float subdt = dt;

        {
            const float3 mygrad = ugrad_sdf(x, y, z);
            const float DphiDt = max(1e-4f, mygrad.x * u + mygrad.y * v + mygrad.z * w);

            assert(DphiDt > 0);

            subdt = min(dt, max(0.f, subdt - currsdf / DphiDt * 1.02f));
        }

#if 1
        {
            const float3 xstar = make_float3(x + subdt * u, y + subdt * v, z + subdt * w);
            const float3 mygrad = ugrad_sdf(xstar.x, xstar.y, xstar.z);
            const float DphiDt = max(1e-4f, mygrad.x * u + mygrad.y * v + mygrad.z * w);

            assert(DphiDt > 0);

            subdt = min(dt, max(0.f, subdt - sdf(xstar.x, xstar.y, xstar.z) / DphiDt * 1.02f));
        }
#endif
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

        return;
    }

    __global__ __launch_bounds__(32 * 4, 12)
    void bounce(float2 * const particles, const int nparticles, const int rank, const float dt)
    {
        assert(blockDim.x * gridDim.x >= nparticles);

        const int pid = threadIdx.x + blockDim.x * blockIdx.x;

        if (pid >= nparticles)
            return;

        float2 data0 = particles[pid * 3];
        float2 data1 = particles[pid * 3 + 1];

#ifndef NDEBUG
        const int L[3] = { XSIZE_SUBDOMAIN, YSIZE_SUBDOMAIN, ZSIZE_SUBDOMAIN };
        const int MARGIN[3] = { XMARGIN_WALL, YMARGIN_WALL, ZMARGIN_WALL };
        const float x[3] = { data0.x, data0.y, data1.x } ;

        for(int c = 0; c < 3; ++c)
        {
            if (!(abs(x[c]) <= L[c]/2 + MARGIN[c]))
                cuda_printf("bounce: ooooooooops component %d we have %f %f %f outside %d + %d\n", c, x[0], x[1], x[2], L[c]/2, MARGIN[c]);

            assert(abs(x[c]) <= L[c]/2 + MARGIN[c]);
        }
#endif

        if (pid < nparticles)
        {
            const float mycheapsdf = cheap_sdf(data0.x, data0.y, data1.x);

            if (mycheapsdf >= -1.7320f * ((float)XSIZE_WALLCELLS / (float)XTEXTURESIZE))
            {
                const float currsdf = sdf(data0.x, data0.y, data1.x);

                float2 data2 = particles[pid * 3 + 2];

                if (currsdf >= 0)
                {
                    handle_collision(currsdf, data0.x, data0.y, data1.x, data1.y, data2.x, data2.y, rank, dt);

                    particles[3 * pid] = data0;
                    particles[3 * pid + 1] = data1;
                    particles[3 * pid + 2] = data2;
                }
            }
        }
    }

    struct StressInfo
    {
	float *sigma_xx, *sigma_xy, *sigma_xz, *sigma_yy, *sigma_yz, *sigma_zz;
    };

    __constant__ StressInfo stressinfo;


    template<bool computestresses >
    __global__ __launch_bounds__(128, 16) void interactions_3tpp(const float2 * const particles, const int np, const int nsolid,
								 float * const acc, const float seed, const float sigmaf,
								 const float xvelocity_wall, const float y0)
    {
        assert(blockDim.x * gridDim.x >= np * 3);

        const int gid = threadIdx.x + blockDim.x * blockIdx.x;
        const int pid = gid / 3;
        const int zplane = gid % 3;

        if (pid >= np)
            return;

        const float2 dst0 = particles[3 * pid + 0];
        const float2 dst1 = particles[3 * pid + 1];

        const float interacting_threshold = -1 - 1.7320f * ((float)XSIZE_WALLCELLS / (float)XTEXTURESIZE);

        if (cheap_sdf(dst0.x, dst0.y, dst1.x) <= interacting_threshold)
            return;

        const float2 dst2 = particles[3 * pid + 2];

#ifndef NDEBUG
        {
            const int L[3] = { XSIZE_SUBDOMAIN, YSIZE_SUBDOMAIN, ZSIZE_SUBDOMAIN };
            const int MARGIN[3] = { XMARGIN_WALL, YMARGIN_WALL, ZMARGIN_WALL };
            const float x[3] = { dst0.x, dst0.y, dst1.x};

            for(int c = 0; c < 3; ++c)
            {
                assert( x[c] >= -L[c]/2 - MARGIN[c]);
                assert( x[c] < L[c]/2 + MARGIN[c]);
            }
        }
#endif

        uint scan1, scan2, ncandidates, spidbase;
        int deltaspid1, deltaspid2;

        {
            const int xbase = (int)(dst0.x - (-XSIZE_SUBDOMAIN/2 - XMARGIN_WALL));
            const int ybase = (int)(dst0.y - (-YSIZE_SUBDOMAIN/2 - YMARGIN_WALL));
            const int zbase = (int)(dst1.x - (-ZSIZE_SUBDOMAIN/2 - ZMARGIN_WALL));

            enum
            {
                XCELLS = XSIZE_SUBDOMAIN + 2 * XMARGIN_WALL,
                YCELLS = YSIZE_SUBDOMAIN + 2 * YMARGIN_WALL,
                ZCELLS = ZSIZE_SUBDOMAIN + 2 * ZMARGIN_WALL,
                NCELLS = XCELLS * YCELLS * ZCELLS
            };

            assert (xbase > 0 && xbase < -1 + XCELLS &&
                    ybase > 0 && ybase < -1 + YCELLS &&
                    zbase > 0 && zbase < -1 + ZCELLS );

            const int cid0 = xbase - 1 + XCELLS * (ybase - 1 + YCELLS * (zbase - 1 + zplane));

            spidbase = tex1Dfetch(texWallCellStart, cid0);
            int count0 = tex1Dfetch(texWallCellStart, cid0 + 3) - spidbase;

            const int cid1 = cid0 + XCELLS;
            deltaspid1 = tex1Dfetch(texWallCellStart, cid1);
            const int count1 = tex1Dfetch(texWallCellStart, cid1 + 3) - deltaspid1;

            const int cid2 = cid0 + XCELLS * 2;
            deltaspid2 = tex1Dfetch(texWallCellStart, cid2);
            assert(cid2 + 3 <= NCELLS);
            const int count2 = cid2 + 3 == NCELLS ? nsolid : tex1Dfetch(texWallCellStart, cid2 + 3) - deltaspid2;

            scan1 = count0;
            scan2 = count0 + count1;
            ncandidates = scan2 + count2;

            deltaspid1 -= scan1;
            deltaspid2 -= scan2;
        }

        float xforce = 0, yforce = 0, zforce = 0;

#pragma unroll 2
        for(int i = 0; i < ncandidates; ++i)
        {
            const int m1 = (int)(i >= scan1);
            const int m2 = (int)(i >= scan2);
            const int spid = i + (m2 ? deltaspid2 : m1 ? deltaspid1 : spidbase);

            assert(spid >= 0 && spid < nsolid);

            const float4 stmp0 = tex1Dfetch(texWallParticles, spid);

            const float xq = stmp0.x;
            const float yq = stmp0.y;
            const float zq = stmp0.z;

            const float _xr = dst0.x - xq;
            const float _yr = dst0.y - yq;
            const float _zr = dst1.x - zq;

            const float rij2 = _xr * _xr + _yr * _yr + _zr * _zr;

            const float invrij = rsqrtf(rij2);

            const float rij = rij2 * invrij;
            const float argwr = max(0.f, 1.f - rij);
            const float wr = viscosity_function<-VISCOSITY_S_LEVEL>(argwr);

            const float xr = _xr * invrij;
            const float yr = _yr * invrij;
            const float zr = _zr * invrij;
	    
	    const float xvel = yq > y0 ? xvelocity_wall : 0;
	  	    
            const float rdotv =
                    xr * (dst1.y - xvel) +
                    yr * (dst2.x - 0) +
                    zr * (dst2.y - 0);

            const float myrandnr = Logistic::mean0var1(seed, pid, spid);

            const float strength = aij * argwr + (- gammadpd * wr * rdotv + sigmaf * myrandnr) * wr;

            xforce += strength * xr;
            yforce += strength * yr;
            zforce += strength * zr;

	    if (computestresses)
	    {
		atomicAdd(stressinfo.sigma_xx + pid, strength * xr * _xr);
		atomicAdd(stressinfo.sigma_xy + pid, strength * xr * _yr);
		atomicAdd(stressinfo.sigma_xz + pid, strength * xr * _zr);
		atomicAdd(stressinfo.sigma_yy + pid, strength * yr * _yr);
		atomicAdd(stressinfo.sigma_yz + pid, strength * yr * _zr);
		atomicAdd(stressinfo.sigma_zz + pid, strength * zr * _zr);
	    }
        }

        atomicAdd(acc + 3 * pid + 0, xforce);
        atomicAdd(acc + 3 * pid + 1, yforce);
        atomicAdd(acc + 3 * pid + 2, zforce);

        for(int c = 0; c < 3; ++c)
            assert(!isnan(acc[3 * pid + c]));
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

        static const size_t CHUNKSIZE = 1 << 25;
#if 1
        int rank;
        MPI_CHECK(MPI_Comm_rank(comm, &rank));

        if (rank==0)
        {
            char header[2048];

            FILE * fh = fopen(path, "rb");

            fread(header, 1, sizeof(header), fh);

            printf("root parsing header\n");
            const int retval = sscanf(header, "%f %f %f\n%d %d %d\n", extent + 0, extent + 1, extent + 2, N + 0, N + 1, N + 2);

            if(retval != 6)
            {
                printf("ooops something went wrong in reading %s.\n", path);
                exit(EXIT_FAILURE);
            }

            printf("broadcasting N\n");
            MPI_CHECK( MPI_Bcast( N, 3, MPI_INT, 0, comm ) );
            MPI_CHECK( MPI_Bcast(extent, 3, MPI_FLOAT, 0, comm ) );

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

            fseek(fh, header_size, SEEK_SET);
            fread(data, sizeof(float), nvoxels, fh);

            fclose(fh);

            printf("broadcasting data\n");

            for(size_t i = 0; i < nvoxels; i += CHUNKSIZE)
            {
                size_t s = (i + CHUNKSIZE <= nvoxels ) ? CHUNKSIZE : (nvoxels - i);
                MPI_CHECK( MPI_Bcast(data + i, s, MPI_FLOAT, 0, comm ) );
                printf("bum %d\n", i);
            }

        }
        else
        {
            MPI_CHECK( MPI_Bcast( N, 3, MPI_INT, 0, comm ) );
            MPI_CHECK( MPI_Bcast(extent, 3, MPI_FLOAT, 0, comm ) );
            const int nvoxels = N[0] * N[1] * N[2];

            data = new float[nvoxels];

            for(size_t i = 0; i < nvoxels; i += CHUNKSIZE)
            {
                size_t s = (i + CHUNKSIZE <= nvoxels ) ? CHUNKSIZE : (nvoxels - i);
                MPI_CHECK( MPI_Bcast(data + i, s, MPI_FLOAT, 0, comm ) );
            }

        }
#else
        char header[2048];

        MPI_File fh;
        MPI_CHECK( MPI_File_open(comm, path, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh) );

        MPI_Status status;
        MPI_CHECK( MPI_File_read_at(fh, 0, header, sizeof(header), MPI_CHAR, &status));

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

        MPI_CHECK( MPI_File_read_at(fh, header_size, data, nvoxels, MPI_FLOAT, &status));

        MPI_CHECK( MPI_File_close(&fh));
#endif
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
                                    g[c] = (l[c] - 1 + anchor[c] + N[c]) % N[c];

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

ComputeWall::ComputeWall(MPI_Comm cartcomm, Particle* const p, const int n, int& nsurvived,
        ExpectedMessageSizes& new_sizes, const float xvelocity):
    cartcomm(cartcomm), arrSDF(NULL), solid4(NULL), solid_size(0), xvelocity(xvelocity),
            cells(XSIZE_SUBDOMAIN + 2 * XMARGIN_WALL, YSIZE_SUBDOMAIN + 2 * YMARGIN_WALL, ZSIZE_SUBDOMAIN + 2 * ZMARGIN_WALL)
{
    MPI_CHECK( MPI_Comm_rank(cartcomm, &myrank));

    MPI_CHECK( MPI_Cart_get(cartcomm, 3, dims, periods, coords) );

    float * field = new float[ XTEXTURESIZE * YTEXTURESIZE * ZTEXTURESIZE];

    static const bool verbose = false;
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

void ComputeWall::bounce(Particle * const p, const int n, cudaStream_t stream)
{
    NVTX_RANGE("WALL/bounce", NVTX_C3)

	        if (n > 0)
	            SolidWallsKernel::bounce<<< (n + 127) / 128, 128, 0, stream>>>((float2 *)p, n, myrank, dt);

    CUDA_CHECK(cudaPeekAtLastError());
}

void ComputeWall::interactions(const Particle * const p, const int n, Acceleration * const acc,
        const int * const cellsstart, const int * const cellscount, cudaStream_t stream)
{
    NVTX_RANGE("WALL/interactions", NVTX_C3);

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

	const float y0 = (dims[1] - 1 - 2 * coords[1]) * YSIZE_SUBDOMAIN / 2;
	
	if (sigma_xx)
	{
	    SolidWallsKernel::StressInfo strinfo = { sigma_xx, sigma_xy, sigma_xz, sigma_yy, sigma_yz, sigma_zz };

	    CUDA_CHECK(cudaMemcpyToSymbolAsync(SolidWallsKernel::stressinfo, &strinfo, sizeof(strinfo), 0, cudaMemcpyHostToDevice, stream));

	    SolidWallsKernel::interactions_3tpp<true><<< (3 * n + 127) / 128, 128, 0, stream>>>
                ((float2 *)p, n, solid_size, (float *)acc, trunk.get_float(), sigmaf, xvelocity, y0);
	}
	else
	    SolidWallsKernel::interactions_3tpp<false><<< (3 * n + 127) / 128, 128, 0, stream>>>
                ((float2 *)p, n, solid_size, (float *)acc, trunk.get_float(), sigmaf, xvelocity, y0);


        CUDA_CHECK(cudaUnbindTexture(SolidWallsKernel::texWallParticles));
        CUDA_CHECK(cudaUnbindTexture(SolidWallsKernel::texWallCellStart));
        CUDA_CHECK(cudaUnbindTexture(SolidWallsKernel::texWallCellCount));
    }

    CUDA_CHECK(cudaPeekAtLastError());
}

ComputeWall::~ComputeWall()
{
    CUDA_CHECK(cudaUnbindTexture(SolidWallsKernel::texSDF));
    CUDA_CHECK(cudaFreeArray(arrSDF));
}
