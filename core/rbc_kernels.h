#pragma once

#include <core/rbc_vector.h>
#include <core/cuda_common.h>

__global__ void computeAreaAndVolume(const float4* coosvels, MembraneMesh mesh, int nRbcs, float* areas, float* volumes)
{
	float2 a_v = make_float2(0.0f, 0.0f);
	const int cid = blockIdx.y;

	for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < mesh.ntriangles; i += blockDim.x * gridDim.x)
	{
		int3 ids = mesh.triangles[i];

		float3 v0 = f4tof3( coosvels[ 2 * 3*(ids.x+cid*mesh.nvertices) ] );
		float3 v1 = f4tof3( coosvels[ 2 * 3*(ids.y+cid*mesh.nvertices) ] );
		float3 v2 = f4tof3( coosvels[ 2 * 3*(ids.z+cid*mesh.nvertices) ] );

		a_v.x += 0.5f * length(cross(v1 - v0, v2 - v0));
		a_v.y += 0.1666666667f * (- v0.z*v1.y*v2.x + v0.z*v1.x*v2.y + v0.y*v1.z*v2.x
				- v0.x*v1.z*v2.y - v0.y*v1.x*v2.z + v0.x*v1.y*v2.z);
	}

	a_v = warpReduce( a_v, [] (float2 a, float2 b) { return a+b; } );
	if ((threadIdx.x & (warpSize - 1)) == 0)
	{
		atomicAdd(areas   + cid, a_v.x);
		atomicAdd(volumes + cid, a_v.y);
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
