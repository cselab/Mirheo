/*
 *  rbc-cuda.cu
 *  ctc local
 *
 *  Created by Dmitry Alexeev on Nov 3, 2014
 *  Copyright 2014 ETH Zurich. All rights reserved.
 *
 */


#include "rbc-cuda-faster.h"

#include "helper_math.h"
#include <cstdio>
#include <map>
#include <string>
#include <fstream>
#include <iostream>
#include <utility>
#include <cassert>
#include <cuda_runtime.h>
#include <vector>

#include <cuda-common.h>

#define WARPSIZE 32

#define CUDA_CHECK(ans) do { cudaAssert((ans), __FILE__, __LINE__); } while(0)
inline void cudaAssert(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);

        abort();
    }
}

using namespace std;

namespace CudaRBC
{

    int nvertices, npatches, degreemax, ntriangles;

    int *triangles;

    // Helper pointers
    int maxCells;
    __constant__ float *totA_V;
    float *host_av;

    // Original configuration
    float* orig_xyzuvw;

    map<cudaStream_t, float*> bufmap;
    __constant__ float A[4][4];

    texture<float2, cudaTextureType1D> texVertices;
    texture<int4,   cudaTextureType1D> texTriangles4;

    short2 *dpackedVertIds;
    int4   *dpackedTrIds;
    int4   *dpackedDihIds;
    short4 *dpackedReduceIds;

    int* triplets;

    Extent* dummy;

    vector<int> patchSize;

    void unitsSetup(float lmax, float p, float cq, float kb, float ka, float kv, float gammaC,
            float totArea0, float totVolume0, float lunit, float tunit, int ndens, bool prn);

    struct Particle
    {
        float x[3], u[3];
    };

    struct Acceleration
    {
        float a[3];
    };

    void resize(int ncells)
    {
        if (ncells > maxCells)
        {
            maxCells = 2*ncells;
            CUDA_CHECK( cudaFree(host_av) );
            CUDA_CHECK( cudaMalloc(&host_av, maxCells * 2 * sizeof(float)) );
            CUDA_CHECK( cudaMemcpyToSymbol(totA_V, &host_av,  sizeof(float*)) );

            delete[] dummy;
            dummy = new Extent[maxCells];
        }
    }
    void setup(int& nv, Extent& host_extent)
    {
        const float scale=1;

        const bool report = false;

        //        0.0945, 0.00141, 1.642599,
        //        1, 1.8, a, v, a/m.ntriang, 945, 0, 472.5,
        //        90, 30, sin(phi), cos(phi), 6.048

        FILE * f = fopen("../cuda-rbc/geom.dat", "r");

        assert(f);
        fscanf(f, "%d %d %d %d", &nvertices, &ntriangles, &npatches, &degreemax);

        patchSize.resize(npatches);
        for (int i=0; i<npatches; i++)
            fscanf(f, "%d", &patchSize[i]);

        vector<Particle> particles;
        for (int i=0; i<nvertices; i++)
        {
            Particle p = {0, 0, 0, 0, 0, 0};

            fscanf(f, "%e %e %e", p.x, p.x+1, p.x+2);

            for (int d=0; d<3; d++)
                p.x[i] *= scale;
            particles.push_back(p);
        }

        vector< int4 > htriangles(ntriangles);
        triplets = new int[3*ntriangles];
        for (int i=0; i<ntriangles; i++)
        {
            fscanf(f, "%d %d %d", &htriangles[i].x, &htriangles[i].y, &htriangles[i].z);
            triplets[3*i+0] = htriangles[i].x;
            triplets[3*i+1] = htriangles[i].y;
            triplets[3*i+2] = htriangles[i].z;
        }

        vector< short2 > packedVertIds(npatches * WARPSIZE);
        vector< int4 >   packedTrIds(npatches * WARPSIZE);
        vector< int4 >   packedDihIds(npatches * WARPSIZE);
        vector< short4 > packedReduceIds(npatches * WARPSIZE * 5);

        for (int i=0; i<npatches; i++)
        {
            for (int j=0; j<WARPSIZE; j++)
                fscanf(f, "%hd %hd", &packedVertIds[i*WARPSIZE + j].x, &packedVertIds[i*WARPSIZE + j].y);
        }

        for (int i=0; i<npatches; i++)
        {
            for (int j=0; j<WARPSIZE; j++)
                fscanf(f, "%d %d %d %d", &packedTrIds[i*WARPSIZE + j].x, &packedTrIds[i*WARPSIZE + j].y,
                        &packedTrIds[i*WARPSIZE + j].z, &packedTrIds[i*WARPSIZE + j].w);
        }

        for (int i=0; i<npatches; i++)
        {
            for (int j=0; j<WARPSIZE; j++)
                fscanf(f, "%d %d %d %d", &packedDihIds[i*WARPSIZE + j].x, &packedDihIds[i*WARPSIZE + j].y,
                        &packedDihIds[i*WARPSIZE + j].z, &packedDihIds[i*WARPSIZE + j].w);
        }

        for (int i=0; i<npatches * WARPSIZE * 5; i++)
        {
            fscanf(f, "%hd %hd %hd %hd", &packedReduceIds[i].x, &packedReduceIds[i].y, &packedReduceIds[i].z, &packedReduceIds[i].w);
        }

        float xmin[3] = { 1e10,  1e10,  1e10};
        float xmax[3] = {-1e10, -1e10, -1e10};

        for (int i=0; i<particles.size(); i++)
            for (int d=0; d<3; d++)
            {
                xmin[d] = min(xmin[d], particles[i].x[d]);
                xmax[d] = max(xmax[d], particles[i].x[d]);
            }

        float origin[3];
        for (int d=0; d<3; d++)
            origin[d] = 0.5 * (xmin[d] + xmax[d]);

        for (int i=0; i<particles.size(); i++)
            for (int d=0; d<3; d++)
                particles[i].x[d] -= origin[d];

        nv = particles.size();


        CUDA_CHECK( cudaMalloc(&orig_xyzuvw, nvertices  * 6 * sizeof(float)) );
        CUDA_CHECK( cudaMalloc(&triangles,   ntriangles * 4 * sizeof(int)) );

        CUDA_CHECK( cudaMemcpy(orig_xyzuvw, &particles[0],    nvertices  * 6 * sizeof(float), cudaMemcpyHostToDevice) );
        CUDA_CHECK( cudaMemcpy(triangles,   &htriangles[0],   ntriangles * 4 * sizeof(int),   cudaMemcpyHostToDevice) );

        //***************************************************************************************************
        CUDA_CHECK( cudaMalloc(&dpackedVertIds,   sizeof(short2)  * npatches * WARPSIZE) );
        CUDA_CHECK( cudaMalloc(&dpackedTrIds,     sizeof(int4)    * npatches * WARPSIZE) );
        CUDA_CHECK( cudaMalloc(&dpackedDihIds,    sizeof(int4)    * npatches * WARPSIZE) );
        CUDA_CHECK( cudaMalloc(&dpackedReduceIds, sizeof(short4)  * npatches * WARPSIZE * 5) );

        CUDA_CHECK( cudaMemcpy(dpackedVertIds,   &packedVertIds[0],   sizeof(short2)  * npatches * WARPSIZE, cudaMemcpyHostToDevice) );
        CUDA_CHECK( cudaMemcpy(dpackedTrIds,     &packedTrIds[0],     sizeof(int4)    * npatches * WARPSIZE, cudaMemcpyHostToDevice) );
        CUDA_CHECK( cudaMemcpy(dpackedDihIds,    &packedDihIds[0],    sizeof(int4)    * npatches * WARPSIZE, cudaMemcpyHostToDevice) );
        CUDA_CHECK( cudaMemcpy(dpackedReduceIds, &packedReduceIds[0], sizeof(short4)  * npatches * WARPSIZE * 5, cudaMemcpyHostToDevice) );
        //***************************************************************************************************


        dummy = new Extent[maxCells];
        host_extent.xmin = xmin[0] - origin[0];
        host_extent.ymin = xmin[1] - origin[1];
        host_extent.zmin = xmin[2] - origin[2];

        host_extent.xmax = xmax[0] - origin[0];
        host_extent.ymax = xmax[1] - origin[1];
        host_extent.zmax = xmax[2] - origin[2];

        maxCells = 5;
        CUDA_CHECK( cudaMalloc(&host_av, maxCells * 2 * sizeof(float)) );
        CUDA_CHECK( cudaMemcpyToSymbol(totA_V, &host_av,  sizeof(float*)) );

        // Texture setup
        texTriangles4.channelDesc = cudaCreateChannelDesc<int4>();
        texTriangles4.filterMode = cudaFilterModePoint;
        texTriangles4.mipmapFilterMode = cudaFilterModePoint;
        texTriangles4.normalized = 0;

        texVertices.channelDesc = cudaCreateChannelDesc<float2>();
        texVertices.filterMode = cudaFilterModePoint;
        texVertices.mipmapFilterMode = cudaFilterModePoint;
        texVertices.normalized = 0;

        size_t textureoffset;
        CUDA_CHECK( cudaBindTexture(&textureoffset, &texTriangles4, triangles, &texTriangles4.channelDesc, ntriangles * 4 * sizeof(int)) );
        assert(textureoffset == 0);

        unitsSetup(1.194170681, 0.003092250212, 20.49568481, 29.2254922344138, 10223.5137655706, 7710.76185113627, 10.14524310, 135, 94, 1e-6, 2.4295e-6, 4, false);
    }

    void unitsSetup(float lmax, float p, float cq, float kb, float ka, float kv, float gammaC,
            float totArea0, float totVolume0, float lunit, float tunit, int ndens, bool prn)
    {
        const float lrbc = 1.000000e-06;
        const float trbc = 3.009441e-03;
        //const float mrbc = 3.811958e-13;

        float ll = lunit / lrbc;
        float tt = tunit / trbc;

        float l0 = 0.5606098578 / ll;

        params.kbT = 0.0748 * 1239*1239 * pow(ll, -2.0) * pow(tt, 2.0);
        params.p = p / ll;
        params.lmax = lmax / ll;
        params.q = 1;
        params.Cq = cq * params.kbT * pow(ll, -2.0);
        params.totArea0 = totArea0 * pow(ll, -2.0);
        params.totVolume0 = totVolume0 * pow(ll, -3.0);
        params.ka =  params.kbT * ka / (l0*l0) / params.totArea0;
        params.kv =  params.kbT * kv / (l0*l0*l0) / params.totVolume0 / 6;
        params.gammaC = gammaC * 1239 * pow(tt, 1.0);
        params.gammaT = 3.0 * params.gammaC;


        float phi = 6.9722 / 180.0*M_PI; //float phi = 3.1 / 180.0*M_PI;
        params.sinTheta0 = sin(phi);
        params.cosTheta0 = cos(phi);
        params.kb = kb * params.kbT;

        //params.mass = 1.1 / 0.995 * params.totVolume0 * ndens / nvertices;

        params.ntriang = ntriangles;
        params.nvertices = nvertices;

        params.sint0kb = params.sinTheta0 * params.kb;
        params.cost0kb = params.cosTheta0 * params.kb;
        params.kbToverp = params.kbT / params.p;

        for (int i=0; i<npatches; i++)
            params.patchSize[i] = patchSize[i];

        CUDA_CHECK( cudaMemcpyToSymbol  (devParams, &params, sizeof(Params)) );


        if (prn)
        {
            printf("\n************* Parameters setup *************\n");
            printf("Started with <RBC space (DPD space)>:\n");
            printf("    DPD unit of time:  %e\n",   tunit);
            printf("    DPD unit of length:  %e\n\n", lunit);
            printf("\t Lmax    %12.5f  (%12.5f)\n", lmax,   params.lmax);
            printf("\t p       %12.5f  (%12.5f)\n", p,      params.p);
            printf("\t Cq      %12.5f  (%12.5f)\n", cq,     params.Cq);
            printf("\t kb      %12.5f  (%12.5f)\n", kb,     params.kb);
            printf("\t ka      %12.5f  (%12.5f)\n", ka,     params.ka * params.totArea0);
            printf("\t kv      %12.5f  (%12.5f)\n", kv,     params.kv * params.totVolume0 * 6);
            printf("\t gammaC  %12.5f  (%12.5f)\n\n", gammaC, params.gammaC);

            printf("\t kbT     %12e in dpd\n", params.kbT);
            //printf("\t mass    %12.5f in dpd\n", params.mass);
            printf("\t area    %12.5f  (%12.5f)\n", totArea0,  params.totArea0);
            printf("\t volume  %12.5f  (%12.5f)\n", totVolume0, params.totVolume0);
            printf("************* **************** *************\n\n");
        }
    }

    int get_nvertices()
    {
        return nvertices;
    }

    Params& get_params()
    {
        return params;
    }

    __device__ __inline__ float atomicMin(float *addr, float value)
    {
        float old = *addr, assumed;
        if(old <= value) return old;

        do
        {
            assumed = old;
            old = __int_as_float( atomicCAS((unsigned int*)addr, __float_as_int(assumed), __float_as_int(min(value, assumed))) );
        }while(old!=assumed);

        return old;
    }

    __device__ __inline__ float atomicMax(float *addr, float value)
    {
        float old = *addr, assumed;
        if(old >= value) return old;

        do
        {
            assumed = old;
            old = __int_as_float( atomicCAS((unsigned int*)addr, __float_as_int(assumed), __float_as_int(max(value, assumed))) );
        }while(old!=assumed);

        return old;
    }


    __global__ void extentKernel(const float* const __restrict__ xyzuvw, Extent* extent, int npart)
    {
        float3 loBound = make_float3( 1e10f,  1e10f,  1e10f);
        float3 hiBound = make_float3(-1e10f, -1e10f, -1e10f);
        const int cid = blockIdx.y;

        for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < npart; i += blockDim.x * gridDim.x)
        {
            const float* addr = xyzuvw + 6 * (devParams.nvertices*cid + i);
            float3 v = make_float3(addr[0], addr[1], addr[2]);

            loBound = fminf(loBound, v);
            hiBound = fmaxf(hiBound, v);
        }

        loBound = warpReduceMin(loBound);
        __syncthreads();
        hiBound = warpReduceMax(hiBound);

        if ((threadIdx.x & (warpSize - 1)) == 0)
        {
            atomicMin(&extent[cid].xmin, loBound.x);
            atomicMin(&extent[cid].ymin, loBound.y);
            atomicMin(&extent[cid].zmin, loBound.z);

            atomicMax(&extent[cid].xmax, hiBound.x);
            atomicMax(&extent[cid].ymax, hiBound.y);
            atomicMax(&extent[cid].zmax, hiBound.z);
        }
    }

    void extent_nohost(cudaStream_t stream, int ncells, const float * const xyzuvw, Extent * device_extent, int n)
    {
        if (ncells == 0) return;

        dim3 threads(32*3, 1);
        dim3 blocks( (nvertices + threads.x - 1) / threads.x, ncells );

        resize(ncells);

        for (int i=0; i<ncells; i++)
        {
            dummy[i].xmin = dummy[i].ymin = dummy[i].zmin = 1e10;
            dummy[i].xmax = dummy[i].ymax = dummy[i].zmax = -1e10;
        }

        CUDA_CHECK( cudaMemcpy(device_extent, dummy, ncells * sizeof(Extent), cudaMemcpyHostToDevice) );

        if (n == -1) n = nvertices;
        extentKernel<<<blocks, threads, 0, stream>>>(xyzuvw, device_extent, n);
        CUDA_CHECK( cudaPeekAtLastError() );
    }


    __global__ void transformKernel(float* xyzuvw, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n) return;

        float x = xyzuvw[6*i + 0];
        float y = xyzuvw[6*i + 1];
        float z = xyzuvw[6*i + 2];

        xyzuvw[6*i + 0] = A[0][0]*x + A[0][1]*y + A[0][2]*z + A[0][3];
        xyzuvw[6*i + 1] = A[1][0]*x + A[1][1]*y + A[1][2]*z + A[1][3];
        xyzuvw[6*i + 2] = A[2][0]*x + A[2][1]*y + A[2][2]*z + A[2][3];
    }

    void initialize(float *device_xyzuvw, const float (*transform)[4])
    {
        const int threads = 128;
        const int blocks  = (nvertices + threads - 1) / threads;

        CUDA_CHECK( cudaMemcpyToSymbol(A, transform, 16 * sizeof(float)) );
        CUDA_CHECK( cudaMemcpy(device_xyzuvw, orig_xyzuvw, 6*nvertices * sizeof(float), cudaMemcpyDeviceToDevice) );
        transformKernel<<<blocks, threads>>>(device_xyzuvw, nvertices);
    }

    __device__ __inline__ float3 tex2vec(int id)
    {
        float2 tmp0 = tex1Dfetch(texVertices, id+0);
        float2 tmp1 = tex1Dfetch(texVertices, id+1);
        return make_float3(tmp0.x, tmp0.y, tmp1.x);
    }

    __global__ void areaAndVolumeKernel()
    {
        float2 a_v = make_float2(0.0f, 0.0f);
        const int cid = blockIdx.y;

        for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < devParams.ntriang; i += blockDim.x * gridDim.x)
        {
            int4 ids = tex1Dfetch(texTriangles4, i);

            float3 v0( tex2vec(3*(ids.x+cid*devParams.nvertices)) );
            float3 v1( tex2vec(3*(ids.y+cid*devParams.nvertices)) );
            float3 v2( tex2vec(3*(ids.z+cid*devParams.nvertices)) );

            a_v.x += 0.5f * length(cross(v1 - v0, v2 - v0));
            a_v.y += 0.1666666667f * (- v0.z*v1.y*v2.x + v0.z*v1.x*v2.y + v0.y*v1.z*v2.x
                    - v0.x*v1.z*v2.y - v0.y*v1.x*v2.z + v0.x*v1.y*v2.z);
        }

        a_v = warpReduceSum(a_v);
        if ((threadIdx.x & (warpSize - 1)) == 0)
        {
            atomicAdd(&totA_V[2*cid+0], a_v.x);
            atomicAdd(&totA_V[2*cid+1], a_v.y);
        }
    }

    __global__ void comKernel(int ncells, const float* __restrict__ xyzuvw, float* coms)
    {
        float3 mycom = make_float3(0.0f, 0.0f, 0.0f);
        const int tid = threadIdx.x;
        const int cid = blockIdx.x;
        const int nvert = devParams.nvertices;

        if (cid >= ncells) return;

        float2 *xyz2 = (float2*)(xyzuvw + cid*6*nvert);
        for(int i = tid; i < nvert; i += blockDim.x)
        {
            float2 tmp1 = xyz2[i*3 + 0];
            float2 tmp2 = xyz2[i*3 + 1];

            mycom.x += tmp1.x;
            mycom.y += tmp1.y;
            mycom.z += tmp2.x;
        }

        mycom = warpReduceSum(mycom) / (float)nvert;

        if (tid == 0)
        {
            coms[cid*3+0] = mycom.x;
            coms[cid*3+1] = mycom.y;
            coms[cid*3+2] = mycom.z;
        }
        __syncthreads();
        if ((tid % warpSize) == 0 && tid > 0)
        {
            atomicAdd(coms + cid*3+0, mycom.x);
            atomicAdd(coms + cid*3+1, mycom.y);
            atomicAdd(coms + cid*3+2, mycom.z);
        }
    }
    __device__ __forceinline__ void trCoeffs(const int trid, const int pverts, const float area, const float3 *verts, float *trAlphas)
    {
        const int mask = 0x1ff;

        const int v1 = pverts & mask;
        const int v2 = (pverts >> 9)  & mask;
        const int v3 = (pverts >> 18) & mask;

        if (v1 < 2*WARPSIZE && v1 >= 0)
        {
            const float3 vv1 = verts[v1];
            const float3 ksi = cross(verts[v2] - vv1, verts[v3] - vv1);
            //trNormals[trid] = ksi;

            const float n_2 = 2.0f * rsqrt(dot(ksi, ksi));
            const float n_2to3 = n_2*n_2*n_2;
            trAlphas[trid] = 0.25f * (devParams.Cq * n_2to3 - devParams.ka * (area - devParams.totArea0) * n_2);
        }
    }
    __device__ inline double ddot(float3 a, float3 b)
    {
        return (double)a.x*b.x + (double)a.y*b.y + (double)a.z*b.z;
    }

    __device__ __forceinline__ void _dihCoeffs(const int dihid, const int pverts, const float3 *verts, float *dihcs)
    {
        const int mask = 0xff;

        const int v1 = pverts & mask;
        const int v2 = (pverts >> 8)  & mask;
        const int v3 = (pverts >> 16) & mask;
        const int v4 = (pverts >> 24) & mask;

        if (v1 < 2*WARPSIZE && v1 >= 0)
        {
            const float3 vv1 = verts[v1];
            const float3 vv2 = verts[v2];
            const float3 vv3 = verts[v3];
            const float3 vv4 = verts[v4];

            const float3 ksi   = cross(vv2 - vv1, vv3 - vv1);
            const float3 dzeta = cross(vv3 - vv4, vv2 - vv4);
            const float3 t_c0 = (vv1 + vv2 + vv3) * 0.3333333333f;
            const float3 t_c1 = (vv2 + vv3 + vv4) * 0.3333333333f;

            const double cosTheta = ddot(ksi, dzeta) * (double)rsqrt(ddot(ksi, ksi)) * (double)rsqrt(ddot(dzeta, dzeta));
            const double IsinThetaI = sqrt(fabs(1.0 - cosTheta*cosTheta));
            const double sinTheta = copysign(max(IsinThetaI, 0.001), dot(ksi - dzeta, t_c0 - t_c1));  // ">" because the normals look inside
            const float beta = devParams.kb * ((double)devParams.cosTheta0 - cosTheta * devParams.sinTheta0 / sinTheta);
            //printf("%f %f\n", cosTheta, beta);
            dihcs[dihid] = beta;
        }
    }


    __device__ __forceinline__ void dihCoeffs(const int dihid, const int pverts, const float3 *verts, float *dihcs)
    {
        const int mask = 0xff;

        const int v1 = pverts & mask;
        const int v2 = (pverts >> 8)  & mask;
        const int v3 = (pverts >> 16) & mask;
        const int v4 = (pverts >> 24) & mask;

        if (v1 < 2*WARPSIZE && v1 >= 0)
        {
            const float3 ksi   = cross(verts[v2] - verts[v1], verts[v3] - verts[v1]);
            const float3 dzeta = cross(verts[v3] - verts[v4], verts[v2] - verts[v4]);

            const float cosTheta = dot(ksi, dzeta) * rsqrt(dot(ksi, ksi)) * rsqrt(dot(dzeta, dzeta));
            const float IsinThetaI = sqrt(fabs(1.0 - cosTheta*cosTheta));
            const float sinTheta = copysign(max(IsinThetaI, 0.001), dot(ksi - dzeta, verts[v1] - verts[v4]));  // ">" because the normals look inside
            const float beta = devParams.cost0kb - cosTheta * devParams.sint0kb / sinTheta;
            //printf("%f %f\n", cosTheta, beta);
            dihcs[dihid] = beta;
        }
    }

    template <bool withAngle>
    __device__ __forceinline__ float3 consForce(const float3 myv, const float3 v1, const float3 v2, const float3 v3, const float3 vo,
            const int trid, const int dihid1, const int dihid2,
            const float beta, const float* trAlphas, const float* dihcs)
    {
        //       vo
        //     /   \ cur
        //   v1 --> v2 --> v3
        //     \   /
        //       V
        //      myv

        const float3 a12 = v1 - v2;
        const float3 a23 = v2 - v3;
        const float3 a2m = v2 - myv;

        const float3 ksi = cross(a2m, a12);
        const float3 dzeta1 = cross(a2m, a23);
        const float3 dzeta2 = cross(vo - v1, a12);

        const float overIksiI    = rsqrt(dot(ksi, ksi));
        const float overIdzeta1I = rsqrt(dot(dzeta1, dzeta1));
        const float overIdzeta2I = rsqrt(dot(dzeta2, dzeta2));

        const float cos1 = -dot(ksi, dzeta1) * overIksiI * overIdzeta1I;
        const float cos2 = dot(ksi, dzeta2) * overIksiI * overIdzeta2I;

        const float3 ckv = cross(ksi, a12);

        float3 f;
        f = withAngle ? ckv * trAlphas[trid] + cross(v2, v1) * beta : make_float3(0, 0, 0);

        //dihedral 1, myv = p2
        const float overIksiI2 = overIksiI * overIksiI;
        f += dihcs[dihid1] * ( cos1 * (ckv * overIksiI2 + cross(dzeta1, a23) * overIdzeta1I * overIdzeta1I) +
                (cross(ksi, a23) + cross(dzeta1, a12)) * overIksiI * overIdzeta1I );

        f += dihcs[dihid2] * ( cross(dzeta2, a12) * overIksiI * overIdzeta2I - cos2 * ckv * overIksiI2);

        float r = length(a2m);
        r = r < 0.0001f ? 0.0001f : r;
        const float xx = r/devParams.lmax;

        const float IbforceI = devParams.kbToverp * ( 0.25f/((1.0f-xx)*(1.0f-xx)) - 0.25f + xx ) / r;  // TODO: minus??
        f += IbforceI * a2m;
        return f;
    }

    __device__ __forceinline__ float3 viscForce(const float3 myv, const float3 myu, const int id,
            const float3* verts, const float3* vels)
    {
        if (id < 2*WARPSIZE)
        {
            const float3 du = vels[id] - myu;
            const float3 dr = myv - verts[id];

            //printf("%d   %f %f %f\n", id, dr.x, dr.y, dr.z);

            return du*devParams.gammaT + dr * devParams.gammaC*dot(du, dr) / dot(dr, dr);
        }
        return make_float3(0,0,0);
    }

    __global__ __launch_bounds__(128, 7)
    void fall_kernel(const int degreemax, const int npatches, const int nrbcs,
            Acceleration* const acc, const short2* __restrict__ packedVertIds, const int4* __restrict__ packedTrIds,
            const int4* __restrict__ packedDihIds, const short4* __restrict__ packedReduceIds)
    {
        const int globid = threadIdx.x + blockDim.x * blockIdx.x;
        const int idrbc  = globid / (npatches * WARPSIZE);
        const int locid  = globid % (npatches * WARPSIZE);
        const int warpid = threadIdx.x / WARPSIZE;
        const int thid   = threadIdx.x % WARPSIZE;

        if (idrbc >= nrbcs) return;

        const float totArea   =      totA_V[2*idrbc + 0];
        const float totVolume = fabs(totA_V[2*idrbc + 1]);

        extern __shared__ float shmem[];
        float* myshmem = shmem + 13 * WARPSIZE * warpid;
        float3* verts     = (float3*)(myshmem);
        float*  trAlphas  = (float*) (myshmem + (2*3)*WARPSIZE);
        float*  dihcs     = (float*) (myshmem + (3*3)*WARPSIZE);

        // Fetch vertices
        short2 vertInfo = packedVertIds[locid];
        const int width = devParams.patchSize[locid / WARPSIZE];
        const int v01   = vertInfo.x + idrbc * devParams.nvertices;
        const int v02   = vertInfo.y + idrbc * devParams.nvertices;

        //printf("uu %hd  %hd\n", vertInfo.x, vertInfo.y);

        if (vertInfo.x >=0 && vertInfo.x < devParams.nvertices)
        {
            const float2 tmp0 = tex1Dfetch(texVertices, v01 * 3 + 0);
            const float2 tmp1 = tex1Dfetch(texVertices, v01 * 3 + 1);
            verts[2*thid+0] = make_float3(tmp0.x, tmp0.y, tmp1.x);
        }

        if (vertInfo.y >=0 && vertInfo.y < devParams.nvertices)
        {
            const float2 tmp0 = tex1Dfetch(texVertices, v02 * 3 + 0);
            const float2 tmp1 = tex1Dfetch(texVertices, v02 * 3 + 1);
            verts[2*thid+1] = make_float3(tmp0.x, tmp0.y, tmp1.x);
        }

        // Compute per-triangle quantities

        const float beta = devParams.kv * (totVolume - devParams.totVolume0);
        const int4 ptrs = packedTrIds[locid];

        trCoeffs(3*thid+0, ptrs.y, totArea, verts, trAlphas);
        trCoeffs(3*thid+1, ptrs.z, totArea, verts, trAlphas);
        trCoeffs(3*thid+2, ptrs.w, totArea, verts, trAlphas);

        const int4 pdihs = packedDihIds[locid];

        dihCoeffs(4*thid+0, pdihs.x, verts, dihcs);
        dihCoeffs(4*thid+1, pdihs.y, verts, dihcs);
        dihCoeffs(4*thid+2, pdihs.z, verts, dihcs);
        dihCoeffs(4*thid+3, pdihs.w, verts, dihcs);

        // Sum up the forces

        float3 f = make_float3(0, 0, 0);
        if (thid < width)
        {
            const int mask8 = 0xff;
            const float3 myv = verts[2*thid];
            const short4 ptr0to7   = packedReduceIds[0 * WARPSIZE * npatches + locid];
            const short4 pv_1to6   = packedReduceIds[1 * WARPSIZE * npatches + locid];
            const short4 potv0to7  = packedReduceIds[2 * WARPSIZE * npatches + locid];
            const short4 pdih0to7  = packedReduceIds[3 * WARPSIZE * npatches + locid];
            const short4 pdih8to16 = packedReduceIds[4 * WARPSIZE * npatches + locid];
            int trid;
            float3 v, vprev, vnext, voth;

            trid  = ptr0to7.x & mask8;
            vprev = verts[pv_1to6.x & mask8];
            v     = verts[(pv_1to6.x >> 8) & mask8];
            vnext = verts[pv_1to6.y & mask8];
            voth  = verts[potv0to7.x & mask8];

            //printf("%d %d %d %d %d %d %d\n");

            f = consForce<true>(myv, vprev, v, vnext, voth, trid, pdih0to7.x & mask8, (pdih0to7.x >> 8) & mask8, beta, trAlphas, dihcs);

            trid  = (ptr0to7.x >> 8) & mask8;
            vprev = v;
            v = vnext;
            vnext = verts[(pv_1to6.y >> 8) & mask8];
            voth  = verts[(potv0to7.x >> 8) & mask8];

            f += consForce<true>(myv, vprev, v, vnext, voth, trid, pdih0to7.y & mask8, (pdih0to7.y >> 8) & mask8, beta, trAlphas, dihcs);

            trid  = ptr0to7.y & mask8;
            vprev = v;
            v = vnext;
            vnext = verts[(pv_1to6.z) & mask8];
            voth = verts[(potv0to7.y) & mask8];

            f += consForce<true>(myv, vprev, v, vnext, voth, trid, pdih0to7.z & mask8, (pdih0to7.z >> 8) & mask8, beta, trAlphas, dihcs);

            trid  = (ptr0to7.y >> 8) & mask8;
            vprev = v;
            v = vnext;
            vnext = verts[(pv_1to6.z >> 8) & mask8];
            voth  = verts[(potv0to7.y >> 8) & mask8];

            f += consForce<true>(myv, vprev, v, vnext, voth, trid, pdih0to7.w & mask8, (pdih0to7.w >> 8) & mask8, beta, trAlphas, dihcs);

            trid  = ptr0to7.z & mask8;
            vprev = v;
            v = vnext;
            vnext = ((pv_1to6.w & mask8) < 64) ? verts[pv_1to6.w & mask8] : verts[(pv_1to6.x >> 8) & mask8];
            voth  = verts[(potv0to7.z) & mask8];
            //printf("%f %f %f   %d\n", vnext.x, vnext.y, vnext.z, pv_1to6.w & mask8);


            f += consForce<true>(myv, vprev, v, vnext, voth, trid, pdih8to16.x & mask8, (pdih8to16.x >> 8) & mask8, beta, trAlphas, dihcs);

            trid = (ptr0to7.z >> 8) & mask8;
            if (trid >=0 && trid < 96)
            {
                vprev = v;
                v = vnext;
                vnext = ((pv_1to6.w >> 8) & mask8) < 64 ? verts[(pv_1to6.w >> 8) & mask8] : verts[(pv_1to6.x >> 8) & mask8];
                voth  = verts[(potv0to7.z >> 8) & mask8];

                f += consForce<true>(myv, vprev, v, vnext, voth, trid, pdih8to16.y & mask8, (pdih8to16.y >> 8) & mask8, beta, trAlphas, dihcs);

                trid = ptr0to7.w & mask8;
                if (trid >=0 && trid < 96)
                {
                    vprev = v;
                    v = vnext;
                    vnext = verts[(pv_1to6.x >> 8) & mask8];
                    voth  = verts[(potv0to7.w) & mask8];

                    f += consForce<true>(myv, vprev, v, vnext, voth, trid, pdih8to16.z & mask8, (pdih8to16.z >> 8) & mask8, beta, trAlphas, dihcs);
                }
            }
        }

        float3*  vels  = (float3*) (myshmem + (2*3)*WARPSIZE);

        if (vertInfo.x >=0 && vertInfo.x < devParams.nvertices)
        {
            const float2 tmp0 = tex1Dfetch(texVertices, v01 * 3 + 1);
            const float2 tmp1 = tex1Dfetch(texVertices, v01 * 3 + 2);
            vels[2*thid+0] = make_float3(tmp0.y, tmp1.x, tmp1.y);
        }

        if (vertInfo.y >=0 && vertInfo.y < devParams.nvertices)
        {
            const float2 tmp0 = tex1Dfetch(texVertices, v02 * 3 + 1);
            const float2 tmp1 = tex1Dfetch(texVertices, v02 * 3 + 2);
            vels[2*thid+1] = make_float3(tmp0.y, tmp1.x, tmp1.y);
        }


        // I'm not finished just yet! Dissipative force
        if (thid < width)
        {
            const int mask8 = 0xff;
            const float3 myv = verts[2*thid];
            const float3 myu = vels[2*thid];
            const short4 pv_1to6   = packedReduceIds[1 * WARPSIZE * npatches + locid];


            f += viscForce(myv, myu, (pv_1to6.x >> 8) & mask8, verts, vels);
            f += viscForce(myv, myu, (pv_1to6.y)      & mask8, verts, vels);
            f += viscForce(myv, myu, (pv_1to6.y >> 8) & mask8, verts, vels);
            f += viscForce(myv, myu, (pv_1to6.z)      & mask8, verts, vels);
            f += viscForce(myv, myu, (pv_1to6.z >> 8) & mask8, verts, vels);
            f += viscForce(myv, myu, (pv_1to6.w)      & mask8, verts, vels);
            f += viscForce(myv, myu, (pv_1to6.w >> 8) & mask8, verts, vels);
        }


        if (thid < width)
        {
            const int base = v01 - thid;

            myshmem[0 + 3*thid] = f.x;
            myshmem[1 + 3*thid] = f.y;
            myshmem[2 + 3*thid] = f.z;

            // Improve!

            acc[base + thid / 3].a[thid % 3] += myshmem[thid];
            acc[base + (thid +   width) / 3].a[(thid +   width) % 3] += myshmem[thid + width];
            acc[base + (thid + 2*width) / 3].a[(thid + 2*width) % 3] += myshmem[thid + 2*width];
        }
    }

    void getCom(cudaStream_t stream, int ncells, const float * const device_xyzuvw, float * const device_com)
    {
        const int warps = 4;

        resize(ncells);

        int threads = warps * 32;
        int blocks = ncells;

        comKernel<<<blocks, threads, 0, stream>>> (ncells, device_xyzuvw, device_com);
        CUDA_CHECK( cudaPeekAtLastError() );
    }
    void forces_nohost(cudaStream_t stream, int ncells, const float * const device_xyzuvw, float * const device_axayaz)
    {
        if (ncells == 0) return;

        resize(ncells);

        size_t textureoffset;
        CUDA_CHECK( cudaBindTexture(&textureoffset, &texVertices,  (float2 *)device_xyzuvw, &texVertices.channelDesc,  ncells * nvertices * 6 * sizeof(float)) );
        assert(textureoffset == 0);

        dim3 trThreads(256, 1);
        dim3 trBlocks( 1, ncells );

        CUDA_CHECK( cudaMemset(host_av, 0, ncells * 2 * sizeof(float)) );
        areaAndVolumeKernel<<<trBlocks, trThreads, 0, stream>>>();
        CUDA_CHECK( cudaPeekAtLastError() );

        //        float *temp = new float[ncells*2];
        //        gpuErrchk( cudaMemcpy(temp, host_av, ncells * 2 * sizeof(float), cudaMemcpyDeviceToHost) );
        //        for (int i=0; i<ncells; i++)
        //            printf("# %d:  Area:  %.4f,  volume  %.4f\n", i, temp[2*i], temp[2*i+1]);

        const int threads = 128;
        const int blocks = (npatches*ncells + (threads / 32) - 1)  / (threads / 32);
        const int shmemAll = threads * 13 * sizeof(float);

        fall_kernel<<<blocks, threads, shmemAll, stream>>>(degreemax, npatches, ncells, (Acceleration*)device_axayaz,
                dpackedVertIds, dpackedTrIds, dpackedDihIds, dpackedReduceIds);
        CUDA_CHECK( cudaPeekAtLastError() );
    }

    void get_triangle_indexing(int (*&host_triplets_ptr)[3], int& ntriangles)
    {
        host_triplets_ptr = (int(*)[3])triplets;
        ntriangles = CudaRBC::ntriangles;
    }

    float* get_orig_xyzuvw()
    {
        return orig_xyzuvw;
    }

}
