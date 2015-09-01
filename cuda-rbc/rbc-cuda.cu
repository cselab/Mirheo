/*
 *  rbc-cuda.cu
 *  ctc local
 *
 *  Created by Dmitry Alexeev on Nov 3, 2014
 *  Completely rewritten by Dmitry Alexeev and Diego Rossinelli since April 1, 2015.
 *
 */


#include "rbc-cuda.h"

#include <math_functions.h>
#include <cstdio>
#include <map>
#include <string>
#include <fstream>
#include <iostream>
#include <utility>
#include <cassert>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

#include "helper_math.h"

using namespace std;


#define CUDA_CHECK(ans) do { cudaAssert((ans), __FILE__, __LINE__); } while(0)
inline void cudaAssert(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);

        abort();
    }
}

namespace CudaRBC
{

    texture<float2, 1, cudaReadModeElementType> texVertices;
    texture<int, 1, cudaReadModeElementType> texAdjVert;
    texture<int, 1, cudaReadModeElementType> texAdjVert2;
    texture<int4,   cudaTextureType1D> texTriangles4;

    float* orig_xyzuvw;
    float* host_av;
    float* devtrs4;

    int *triplets;

    __constant__ float A[4][4];

    int maxCells;

    void unitsSetup(float lmax, float p, float cq, float kb, float ka, float kv, float gammaC,
            float totArea0, float totVolume0, float lunit, float tunit, int ndens, bool prn);

    void eat_until(FILE * f, string target)
    {
        while(!feof(f))
        {
            char buf[2048];
            fgets(buf, 2048, f);

            if (string(buf) == target)
            {
                fgets(buf, 2048, f);
                break;
            }
        }
    }

    vector<int> extract_neighbors(vector<int> adjVert, const int degreemax, const int v)
            {
        vector<int> myneighbors;
        for(int c = 0; c < degreemax; ++c)
        {
            const int val = adjVert[c + degreemax * v];
            if (val == -1)
                break;

            myneighbors.push_back(val);
        }

        return myneighbors;
            }

    void setup_support(const int * data, const int * data2, const int nentries)
    {
        texAdjVert.channelDesc = cudaCreateChannelDesc<int>();
        texAdjVert.filterMode = cudaFilterModePoint;
        texAdjVert.mipmapFilterMode = cudaFilterModePoint;
        texAdjVert.normalized = 0;

        size_t textureoffset;
        CUDA_CHECK(cudaBindTexture(&textureoffset, &texAdjVert, data,
                &texAdjVert.channelDesc, sizeof(int) * nentries));
        assert(textureoffset == 0);

        texAdjVert2.channelDesc = cudaCreateChannelDesc<int>();
        texAdjVert2.filterMode = cudaFilterModePoint;
        texAdjVert2.mipmapFilterMode = cudaFilterModePoint;
        texAdjVert2.normalized = 0;

        CUDA_CHECK(cudaBindTexture(&textureoffset, &texAdjVert2, data2,
                &texAdjVert.channelDesc, sizeof(int) * nentries));
        assert(textureoffset == 0);
    }

    struct Particle
    {
        float x[3], u[3];
    };

    template <int nvertices>
    __global__ __launch_bounds__(128, 12)
    void fall_kernel(const int nrbcs, float* const __restrict__ av, float * const acc);

    void setup(int& nvertices, Extent& host_extent)
    {
        const float scale=1;
        const bool report = false;

        FILE * f = fopen("../cuda-rbc/rbc.dat", "r");
        if (!f)
        {
            printf("Error in cuda-rbc: data file not found!\n");
            exit(1);
        }

        eat_until(f, "Atoms\n");

        vector<Particle> particles;
        while(!feof(f))
        {
            Particle p = {0, 0, 0, 0, 0, 0};
            int dummy[3];

            const int retval = fscanf(f, "%d %d %d %e %e %e\n", dummy + 0, dummy + 1, dummy + 2,
                    p.x, p.x+1, p.x+2);

            p.x[0] *= scale;
            p.x[1] *= scale;
            p.x[2] *= scale;

            if (retval != 6)
                break;

            particles.push_back(p);
        }

        eat_until(f, "Angles\n");

        vector< int3 > triangles;


        while(!feof(f))
        {
            int dummy[2];
            int3 tri;
            const int retval = fscanf(f, "%d %d %d %d %d\n", dummy + 0, dummy + 1,
                    &tri.x, &tri.y, &tri.z);

            //tri.x -= 1;      tri.y -= 1;      tri.z -= 1;

            if (retval != 5)
                break;

            triangles.push_back(tri);
        }
        fclose(f);

        triplets = new int[3*triangles.size()];
        int* trs4 = new int[4*triangles.size()];

        for (int i=0; i<triangles.size(); i++)
        {
            int3 tri = triangles[i];
            triplets[3*i + 0] = tri.x;
            triplets[3*i + 1] = tri.y;
            triplets[3*i + 2] = tri.z;

            trs4[4*i + 0] = tri.x;
            trs4[4*i + 1] = tri.y;
            trs4[4*i + 2] = tri.z;
            trs4[4*i + 3] = 0;
        }

        nvertices = particles.size();
        vector< map<int, int> > adjacentPairs(nvertices);

        for(int i = 0; i < triangles.size(); ++i)
        {
            const int tri[3] = {triangles[i].x, triangles[i].y, triangles[i].z};

            for(int d = 0; d < 3; ++d)
            {
                assert(tri[d] >= 0 && tri[d] < nvertices);

                adjacentPairs[tri[d]][tri[(d + 1) % 3]] = tri[(d + 2) % 3];
            }

        }

        vector<int> maxldeg;
        for(int i = 0; i < nvertices; ++i)
            maxldeg.push_back(adjacentPairs[i].size());

        const int degreemax = *max_element(maxldeg.begin(), maxldeg.end());
        assert(degreemax == 7);
        assert(nvertices == 498);

        vector<int> adjVert(nvertices * degreemax, -1);

        for(int v = 0; v < nvertices; ++v)
        {
            map<int, int> l = adjacentPairs[v];

            adjVert[0 + degreemax * v] = l.begin()->first;
            int last = adjVert[1 + degreemax * v] = l.begin()->second;

            for(int i = 2; i < l.size(); ++i)
            {
                assert(l.find(last) != l.end());

                int tmp = adjVert[i + degreemax * v] = l.find(last)->second;
                last = tmp;
            }
        }

        vector<int> adjVert2(degreemax * nvertices, -1);

        for(int v = 0; v < nvertices; ++v)
        {
            vector<int> myneighbors = extract_neighbors(adjVert, degreemax, v);

            for(int i = 0; i < myneighbors.size(); ++i)
            {
                vector<int> s1 = extract_neighbors(adjVert, degreemax, myneighbors[i]);
                sort(s1.begin(), s1.end());

                vector<int> s2 = extract_neighbors(adjVert, degreemax, myneighbors[(i + 1) % myneighbors.size()]);
                sort(s2.begin(), s2.end());

                vector<int> result(s1.size() + s2.size());

                const int nterms =  set_intersection(s1.begin(), s1.end(), s2.begin(), s2.end(),
                        result.begin()) - result.begin();

                assert(nterms == 2);

                const int myguy = result[0] == v;

                adjVert2[i + degreemax * v] = result[myguy];
            }
        }

        params.nvertices = nvertices;
        params.ntriangles = triangles.size();

        float* xyzuvw_host = new float[6*nvertices * sizeof(float)];
        for (int i=0; i<nvertices; i++)
        {
            xyzuvw_host[6*i+0] = particles[i].x[0];
            xyzuvw_host[6*i+1] = particles[i].x[1];
            xyzuvw_host[6*i+2] = particles[i].x[2];
            xyzuvw_host[6*i+3] = 0;
            xyzuvw_host[6*i+4] = 0;
            xyzuvw_host[6*i+5] = 0;
        }

        CUDA_CHECK( cudaMalloc(&orig_xyzuvw, nvertices * 6 * sizeof(float)) );
        CUDA_CHECK( cudaMemcpy(orig_xyzuvw, xyzuvw_host, nvertices * 6 * sizeof(float), cudaMemcpyHostToDevice) );
        delete[] xyzuvw_host;

        CUDA_CHECK( cudaMalloc(&devtrs4, params.ntriangles * 4 * sizeof(int)) );
        CUDA_CHECK( cudaMemcpy(devtrs4, trs4, params.ntriangles * 4 * sizeof(int), cudaMemcpyHostToDevice) );
        delete[] trs4;

        const int nentries = adjVert.size();

        int * ptr, * ptr2;
        CUDA_CHECK(cudaMalloc(&ptr, sizeof(int) * nentries));
        CUDA_CHECK(cudaMemcpy(ptr, &adjVert.front(), sizeof(int) * nentries, cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc(&ptr2, sizeof(int) * nentries));
        CUDA_CHECK(cudaMemcpy(ptr2, &adjVert2.front(), sizeof(int) * nentries, cudaMemcpyHostToDevice));

        setup_support(ptr, ptr2, nentries);

        texTriangles4.channelDesc = cudaCreateChannelDesc<int4>();
        texTriangles4.filterMode = cudaFilterModePoint;
        texTriangles4.mipmapFilterMode = cudaFilterModePoint;
        texTriangles4.normalized = 0;

        texVertices.channelDesc = cudaCreateChannelDesc<float2>();
        texVertices.filterMode = cudaFilterModePoint;
        texVertices.mipmapFilterMode = cudaFilterModePoint;
        texVertices.normalized = 0;

        size_t textureoffset;
        CUDA_CHECK( cudaBindTexture(&textureoffset, &texTriangles4, devtrs4, &texTriangles4.channelDesc, params.ntriangles * 4 * sizeof(int)) );
        assert(textureoffset == 0);

        maxCells = 0;
        CUDA_CHECK( cudaMalloc(&host_av, 1 * 2 * sizeof(float)) );

        unitsSetup(1.64, 0.001412, 19.0476, 35, 2500, 3500, 50, 135, 91, 1e-6, 2.4295e-6, 4, report);

        CUDA_CHECK( cudaFuncSetCacheConfig(fall_kernel<498>, cudaFuncCachePreferL1) );
    }

    void unitsSetup(float lmax, float p, float cq, float kb, float ka, float kv, float gammaC,
            float totArea0, float totVolume0, float lunit, float tunit, int ndens, bool prn)
    {
        const float lrbc = 1.000000e-06;
        const float trbc = 3.009441e-03;
        //const float mrbc = 3.811958e-13;

        float ll = lunit / lrbc;
        float tt = tunit / trbc;

        params.kbT = 580 * 250 * pow(ll, -2.0) * pow(tt, 2.0);
        params.p = p / ll;
        params.lmax = lmax / ll;
        params.q = 1;
        params.Cq = cq * params.kbT * pow(ll, -2.0);
        params.totArea0 = totArea0 * pow(ll, -2.0);
        params.totVolume0 = totVolume0 * pow(ll, -3.0);
        params.l0 = sqrt(params.totArea0 / (2.0*params.nvertices - 4.) * 4.0/sqrt(3.0));
        params.ka = ka * params.kbT / (params.totArea0 * params.l0 * params.l0);
        params.kv = kv * params.kbT / (6 * params.totVolume0 * powf(params.l0, 3));
        params.gammaC = gammaC * 580 * pow(tt, 1.0);
        params.gammaT = 3.0 * params.gammaC;

        float phi = 6.97 / 180.0*M_PI;
        params.sinTheta0 = sin(phi);
        params.cosTheta0 = cos(phi);
        params.kb = kb * params.kbT;

        params.kbToverp = params.kbT / params.p;
        params.sint0kb = params.sinTheta0 * params.kb;
        params.cost0kb = params.cosTheta0 * params.kb;
        CUDA_CHECK( cudaMemcpyToSymbol  (devParams, &params, sizeof(Params)) );

        if (prn)
        {
            printf("\n************* Parameters setup *************\n");
            printf("Started with <RBC space (DPD space)>:\n");
            printf("    DPD unit of time:  %e\n",   tunit);
            printf("    DPD unit of length:  %e\n\n", lunit);
            printf("\t Lmax    %12.5f  (%12.5f)\n", lmax,   params.lmax);
            printf("\t l0      %12.5f\n",           params.l0);
            printf("\t p       %12.5f  (%12.5f)\n", p,      params.p);
            printf("\t Cq      %12.5f  (%12.5f)\n", cq,     params.Cq);
            printf("\t kb      %12.5f  (%12.5f)\n", kb,     params.kb);
            printf("\t ka      %12.5f  (%12.5f)\n", ka,     params.ka);
            printf("\t kv      %12.5f  (%12.5f)\n", kv,     params.kv);
            printf("\t gammaC  %12.5f  (%12.5f)\n\n", gammaC, params.gammaC);

            printf("\t kbT     %12e in dpd\n", params.kbT);
            printf("\t area    %12.5f  (%12.5f)\n", totArea0,  params.totArea0);
            printf("\t volume  %12.5f  (%12.5f)\n", totVolume0, params.totVolume0);
            printf("************* **************** *************\n\n");
        }
    }

    int get_nvertices()
    {
        return params.nvertices;
    }

    Params& get_params()
    {
        return params;
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
        const int blocks  = (params.nvertices + threads - 1) / threads;

        CUDA_CHECK( cudaMemcpyToSymbol(A, transform, 16 * sizeof(float)) );
        CUDA_CHECK( cudaMemcpy(device_xyzuvw, orig_xyzuvw, 6*params.nvertices * sizeof(float), cudaMemcpyDeviceToDevice) );
        transformKernel<<<blocks, threads>>>(device_xyzuvw, params.nvertices);
        CUDA_CHECK( cudaPeekAtLastError() );
    }


    __device__ __forceinline__ float3 tex2vec(int id)
    {
        float2 tmp0 = tex1Dfetch(texVertices, id+0);
        float2 tmp1 = tex1Dfetch(texVertices, id+1);
        return make_float3(tmp0.x, tmp0.y, tmp1.x);
    }

    __device__ __forceinline__ float2 warpReduceSum(float2 val)
    {
        for (int offset = warpSize/2; offset > 0; offset /= 2)
        {
            val.x += __shfl_down(val.x, offset);
            val.y += __shfl_down(val.y, offset);
        }
        return val;
    }

    __global__ void areaAndVolumeKernel(float* totA_V)
    {
        float2 a_v = make_float2(0.0f, 0.0f);
        const int cid = blockIdx.y;

        for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < devParams.ntriangles; i += blockDim.x * gridDim.x)
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

    // **************************************************************************************************

    __device__ __forceinline__ float3 _fangle(const float3 v1, const float3 v2, const float3 v3,
            const float area, const float volume)
    {
        assert(devParams.q == 1);
        const float3 x21 = v2 - v1;
        const float3 x32 = v3 - v2;
        const float3 x31 = v3 - v1;

        const float3 normal = cross(x21, x31);

        const float n_2 = 2.0f * rsqrtf(dot(normal, normal));
        const float n_2to3 = n_2*n_2*n_2;
        const float coefArea = 0.25f * (devParams.Cq * n_2to3 -
                devParams.ka * (area - devParams.totArea0) * n_2);

        const float coeffVol = devParams.kv * (volume - devParams.totVolume0);
        const float3 addFArea = coefArea * cross(normal, x32);
        const float3 addFVolume = coeffVol * cross(v3, v2);

        float r = length(v2 - v1);
        r = r < 0.0001f ? 0.0001f : r;
        const float xx = r/devParams.lmax;
        const float IbforceI = devParams.kbToverp * ( 0.25f/((1.0f-xx)*(1.0f-xx)) - 0.25f + xx ) / r;

        return addFArea + addFVolume + IbforceI * x21;
    }

    __device__ __forceinline__ float3 _fvisc(const float3 v1, const float3 v2, const float3 u1, const float3 u2)
    {
        const float3 du = u2 - u1;
        const float3 dr = v1 - v2;

        return du*devParams.gammaT + dr * devParams.gammaC*dot(du, dr) / dot(dr, dr);
    }

    template <int nvertices>
    __device__
    float3 _fangle_device(const float2 tmp0, const float2 tmp1, float* av)
    {
        const int degreemax = 7;
        const int pid = (threadIdx.x + blockDim.x * blockIdx.x) / degreemax;
        const int lid = pid % nvertices;
        const int idrbc = pid / nvertices;
        const int offset = idrbc * nvertices * 3;
        const int neighid = (threadIdx.x + blockDim.x * blockIdx.x) % degreemax;

        const float2 tmp2 = tex1Dfetch(texVertices, pid * 3 + 2);
        const float3 v1 = make_float3(tmp0.x, tmp0.y, tmp1.x);
        const float3 u1 = make_float3(tmp1.y, tmp2.x, tmp2.y);

        const int idv2 = tex1Dfetch(texAdjVert, neighid + degreemax * lid);
        bool valid = idv2 != -1;

        int idv3 = tex1Dfetch(texAdjVert, ((neighid + 1) % degreemax) + degreemax * lid);

        if (idv3 == -1 && valid)
            idv3 = tex1Dfetch(texAdjVert, 0 + degreemax * lid);

        if (valid)
        {
            const float2 tmp0 = tex1Dfetch(texVertices, offset + idv2 * 3 + 0);
            const float2 tmp1 = tex1Dfetch(texVertices, offset + idv2 * 3 + 1);
            const float2 tmp2 = tex1Dfetch(texVertices, offset + idv2 * 3 + 2);
            const float2 tmp3 = tex1Dfetch(texVertices, offset + idv3 * 3 + 0);
            const float2 tmp4 = tex1Dfetch(texVertices, offset + idv3 * 3 + 1);

            const float3 v2 = make_float3(tmp0.x, tmp0.y, tmp1.x);
            const float3 u2 = make_float3(tmp1.y, tmp2.x, tmp2.y);
            const float3 v3 = make_float3(tmp3.x, tmp3.y, tmp4.x);

            float3 f = _fangle(v1, v2, v3, av[2*idrbc], av[2*idrbc+1]);
            f += _fvisc(v1, v2, u1, u2);
            return f;
        }
        return make_float3(-1.0e10f, -1.0e10f, -1.0e10f);
    }

    //======================================
    //======================================

    template<int update>
    __device__  __forceinline__  float3 _fdihedral(float3 v1, float3 v2, float3 v3, float3 v4)
    {
        const float3 ksi   = cross(v1 - v2, v1 - v3);
        const float3 dzeta = cross(v3 - v4, v2 - v4);

        const float overIksiI   = rsqrtf(dot(ksi, ksi));
        const float overIdzetaI = rsqrtf(dot(dzeta, dzeta));

        const float cosTheta = dot(ksi, dzeta) * overIksiI*overIdzetaI;
        const float IsinThetaI2 = 1.0f - cosTheta*cosTheta;
        const float sinTheta_1 = copysignf( rsqrtf(max(IsinThetaI2, 1.0e-6f)), dot(ksi - dzeta, v4 - v1) );  // ">" because the normals look inside
        const float beta = devParams.cost0kb - cosTheta * devParams.sint0kb * sinTheta_1;

        float b11 = -beta * cosTheta * overIksiI*overIksiI;
        float b12 = beta * overIksiI*overIdzetaI;
        float b22 = -beta * cosTheta * overIdzetaI*overIdzetaI;

        if (update == 1)
            return cross(ksi, v3 - v2)*b11 + cross(dzeta, v3 - v2)*b12;
        else if (update == 2)
            return cross(ksi, v1 - v3)*b11 + ( cross(ksi, v3 - v4) + cross(dzeta, v1 - v3) )*b12 + cross(dzeta, v3 - v4)*b22;
        else return make_float3(0, 0, 0);
    }


    template <int nvertices>
    __device__
    float3 _fdihedral_device(const float2 tmp0, const float2 tmp1)
    {
        const int degreemax = 7;
        const int pid = (threadIdx.x + blockDim.x * blockIdx.x) / degreemax;
        const int lid = pid % nvertices;
        const int offset = (pid / nvertices) * nvertices * 3;
        const int neighid = (threadIdx.x + blockDim.x * blockIdx.x) % degreemax;

        const float3 v0 = make_float3(tmp0.x, tmp0.y, tmp1.x);

        //       v4
        //     /   \
        //   v1 --> v2 --> v3
        //     \   /
        //       V
        //       v0

        // dihedrals: 0124, 0123

        int idv1, idv2, idv3, idv4;
        idv1 = tex1Dfetch(texAdjVert, neighid + degreemax * lid);
        const bool valid = idv1 != -1;

        idv2 = tex1Dfetch(texAdjVert, ((neighid + 1) % degreemax) +  degreemax * lid);

        if (idv2 == -1 && valid)
        {
            idv2 = tex1Dfetch(texAdjVert, 0 + degreemax * lid);
            idv3 = tex1Dfetch(texAdjVert, 1 + degreemax * lid);
        }
        else
        {
            idv3 = tex1Dfetch(texAdjVert, ((neighid + 2) % degreemax) +  degreemax * lid);
            if (idv3 == -1 && valid)
                idv3 = tex1Dfetch(texAdjVert, 0 + degreemax * lid);
        }

        idv4 = tex1Dfetch(texAdjVert2, neighid + degreemax * lid);

        if (valid)
        {
            const float2 tmp0 = tex1Dfetch(texVertices, offset + idv1 * 3 + 0);
            const float2 tmp1 = tex1Dfetch(texVertices, offset + idv1 * 3 + 1);
            const float2 tmp2 = tex1Dfetch(texVertices, offset + idv2 * 3 + 0);
            const float2 tmp3 = tex1Dfetch(texVertices, offset + idv2 * 3 + 1);
            const float2 tmp4 = tex1Dfetch(texVertices, offset + idv3 * 3 + 0);
            const float2 tmp5 = tex1Dfetch(texVertices, offset + idv3 * 3 + 1);
            const float2 tmp6 = tex1Dfetch(texVertices, offset + idv4 * 3 + 0);
            const float2 tmp7 = tex1Dfetch(texVertices, offset + idv4 * 3 + 1);

            const float3 v1 = make_float3(tmp0.x, tmp0.y, tmp1.x);
            const float3 v2 = make_float3(tmp2.x, tmp2.y, tmp3.x);
            const float3 v3 = make_float3(tmp4.x, tmp4.y, tmp5.x);
            const float3 v4 = make_float3(tmp6.x, tmp6.y, tmp7.x);

            return _fdihedral<1>(v0, v2, v1, v4) + _fdihedral<2>(v1, v0, v2, v3);
        }
        return make_float3(-1.0e10f, -1.0e10f, -1.0e10f);
    }

    template <int nvertices>
    __global__ __launch_bounds__(128, 12)
    void fall_kernel(const int nrbcs, float* const __restrict__ av, float * const acc)
    {
        const int degreemax = 7;
        const int pid = (threadIdx.x + blockDim.x * blockIdx.x) / degreemax;

        if (pid < nrbcs * nvertices)
        {
            const float2 tmp0 = tex1Dfetch(texVertices, pid * 3 + 0);
            const float2 tmp1 = tex1Dfetch(texVertices, pid * 3 + 1);

            float3 f = _fangle_device<nvertices>(tmp0, tmp1, av);
            f += _fdihedral_device<nvertices>(tmp0, tmp1);

            if (f.x > -1.0e9f)
            {
                atomicAdd(&acc[3*pid+0], f.x);
                atomicAdd(&acc[3*pid+1], f.y);
                atomicAdd(&acc[3*pid+2], f.z);
            }
        }
    }

    // **************************************************************************************************

    void forces_nohost(cudaStream_t stream, int ncells, const float * const device_xyzuvw, float * const device_axayaz)
    {
        if (ncells == 0) return;

        if (ncells > maxCells)
        {
            maxCells = 2*ncells;
            CUDA_CHECK( cudaFree(host_av) );
            CUDA_CHECK( cudaMalloc(&host_av, maxCells * 2 * sizeof(float)) );
        }

        size_t textureoffset;
        CUDA_CHECK( cudaBindTexture(&textureoffset, &texVertices, (float2 *)device_xyzuvw,
                &texVertices.channelDesc, ncells * params.nvertices * sizeof(float) * 6) );
        assert(textureoffset == 0);

        dim3 avThreads(256, 1);
        dim3 avBlocks( 1, ncells );

        CUDA_CHECK( cudaMemsetAsync(host_av, 0, ncells * 2 * sizeof(float), stream) );
        areaAndVolumeKernel<<<avBlocks, avThreads, 0, stream>>>(host_av);
        CUDA_CHECK( cudaPeekAtLastError() );

//        		float *temp = new float[ncells*2];
//        		CUDA_CHECK( cudaMemcpy(temp, host_av, ncells * 2 * sizeof(float), cudaMemcpyDeviceToHost) );
//        		for (int i=0; i<ncells; i++)
//        			printf("# %d:  Area:  %.4f,  volume  %.4f\n", i, temp[2*i], temp[2*i+1]);

        int threads = 128;
        int blocks  = (ncells*params.nvertices*7 + threads-1) / threads;

        fall_kernel<498><<<blocks, threads, 0, stream>>>(ncells, host_av, device_axayaz);
    }

    void get_triangle_indexing(int (*&host_triplets_ptr)[3], int& ntriangles)
    {
        host_triplets_ptr = (int(*)[3])triplets;
        ntriangles = params.ntriangles;
    }

    float* get_orig_xyzuvw()
    {
        return orig_xyzuvw;
    }

    void extent_nohost(cudaStream_t stream, int ncells, const float * const xyzuvw, Extent * device_extent, int n)
    { }


}
