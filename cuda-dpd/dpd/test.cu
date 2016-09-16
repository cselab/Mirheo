
/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cassert>
#include <vector>

#include "cuda-dpd.h"
#include "../cell-lists.h"

#define CUDA_CHECK(ans) do { cudaAssert((ans), __FILE__, __LINE__); } while(0)
inline void cudaAssert(cudaError_t code, const char *file, int line)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);

		abort();
	}
}

struct Particle
{
	float x[3], u[3];
};

struct Acceleration
{
	float a[3];
};

//container for the gpu particles during the simulation
template<typename T>
struct SimpleDeviceBuffer
{
	int capacity, size;

	T * data;

	SimpleDeviceBuffer(int n = 0): capacity(0), size(0), data(NULL) { resize(n);}

	~SimpleDeviceBuffer()
	{
		if (data != NULL)
			CUDA_CHECK(cudaFree(data));

		data = NULL;
	}

	void dispose()
	{
		if (data != NULL)
			CUDA_CHECK(cudaFree(data));

		data = NULL;
	}

	void resize(const int n)
	{
		assert(n >= 0);

		size = n;

		if (capacity >= n)
			return;

		if (data != NULL)
			CUDA_CHECK(cudaFree(data));

		const int conservative_estimate = (int)ceil(1.1 * n);
		capacity = 128 * ((conservative_estimate + 129) / 128);

		CUDA_CHECK(cudaMalloc(&data, sizeof(T) * capacity));

#ifndef NDEBUG
		CUDA_CHECK(cudaMemset(data, 0, sizeof(T) * capacity));
#endif
	}

	void preserve_resize(const int n)
	{
		assert(n >= 0);

		T * old = data;

		const int oldsize = size;

		size = n;

		if (capacity >= n)
			return;

		const int conservative_estimate = (int)ceil(1.1 * n);
		capacity = 128 * ((conservative_estimate + 129) / 128);

		CUDA_CHECK(cudaMalloc(&data, sizeof(T) * capacity));

		if (old != NULL)
		{
			CUDA_CHECK(cudaMemcpy(data, old, sizeof(T) * oldsize, cudaMemcpyDeviceToDevice));
			CUDA_CHECK(cudaFree(old));
		}
	}
};

template<typename T>
struct PinnedHostBuffer
{
	int capacity, size;

	T * data, * devptr;

	PinnedHostBuffer(int n = 0): capacity(0), size(0), data(NULL), devptr(NULL) { resize(n);}

	~PinnedHostBuffer()
	{
		if (data != NULL)
			CUDA_CHECK(cudaFreeHost(data));

		data = NULL;
	}

	void resize(const int n)
	{
		assert(n >= 0);

		size = n;

		if (capacity >= n)
			return;

		if (data != NULL)
			CUDA_CHECK(cudaFreeHost(data));

		const int conservative_estimate = (int)ceil(1.1 * n);
		capacity = 128 * ((conservative_estimate + 129) / 128);

		CUDA_CHECK(cudaHostAlloc(&data, sizeof(T) * capacity, cudaHostAllocMapped));

		CUDA_CHECK(cudaHostGetDevicePointer(&devptr, data, 0));
	}

	void preserve_resize(const int n)
	{
		assert(n >= 0);

		T * old = data;

		const int oldsize = size;

		size = n;

		if (capacity >= n)
			return;

		const int conservative_estimate = (int)ceil(1.1 * n);
		capacity = 128 * ((conservative_estimate + 129) / 128);

		data = NULL;
		CUDA_CHECK(cudaHostAlloc(&data, sizeof(T) * capacity, cudaHostAllocMapped));

		if (old != NULL)
		{
			CUDA_CHECK(cudaMemcpy(data, old, sizeof(T) * oldsize, cudaMemcpyHostToHost));
			CUDA_CHECK(cudaFreeHost(old));
		}

		CUDA_CHECK(cudaHostGetDevicePointer(&devptr, data, 0));
	}
};


__global__ void make_texture( float4 * xyzouvwo, ushort4 * xyzo_half, const float * xyzuvw, const uint n )
{
	extern __shared__ volatile float  smem[];
	const uint warpid = threadIdx.x / 32;
	const uint lane = threadIdx.x % 32;

	const uint i =  (blockIdx.x * blockDim.x + threadIdx.x ) & 0xFFFFFFE0U;

	const float2 * base = ( float2* )( xyzuvw +  i * 6 );
#pragma unroll 3
	for( uint j = lane; j < 96; j += 32 ) {
		float2 u = base[j];
		// NVCC bug: no operator = between volatile float2 and float2
		asm volatile( "st.volatile.shared.v2.f32 [%0], {%1, %2};" : : "r"( ( warpid * 96 + j )*8 ), "f"( u.x ), "f"( u.y ) : "memory" );
	}
	// SMEM: XYZUVW XYZUVW ...
	uint pid = lane / 2;
	const uint x_or_v = ( lane % 2 ) * 3;
	xyzouvwo[ i * 2 + lane ] = make_float4( smem[ warpid * 192 + pid * 6 + x_or_v + 0 ],
			smem[ warpid * 192 + pid * 6 + x_or_v + 1 ],
			smem[ warpid * 192 + pid * 6 + x_or_v + 2 ], 0 );
	pid += 16;
	xyzouvwo[ i * 2 + lane + 32] = make_float4( smem[ warpid * 192 + pid * 6 + x_or_v + 0 ],
			smem[ warpid * 192 + pid * 6 + x_or_v + 1 ],
			smem[ warpid * 192 + pid * 6 + x_or_v + 2 ], 0 );

	xyzo_half[i + lane] = make_ushort4( __float2half_rn( smem[ warpid * 192 + lane * 6 + 0 ] ),
			__float2half_rn( smem[ warpid * 192 + lane * 6 + 1 ] ),
			__float2half_rn( smem[ warpid * 192 + lane * 6 + 2 ] ), 0 );
	// }
}

__global__ void make_texture1( float4 * xyzouvwo, ushort4 * xyzo_half, const float2 * __restrict xyzuvw, const uint n )
{
	const uint pid =  (blockIdx.x * blockDim.x + threadIdx.x);
	if (pid >= n) return;

	const float2 tmp0 = xyzuvw[pid*3 + 0];
	const float2 tmp1 = xyzuvw[pid*3 + 1];
	const float2 tmp2 = xyzuvw[pid*3 + 2];

	xyzo_half[pid] = make_ushort4( __float2half_rn(tmp0.x), __float2half_rn(tmp0.y), __float2half_rn(tmp1.x), 0 );

	xyzouvwo[2*pid+0] = make_float4(tmp0.x, tmp0.y, tmp1.x, 0 );
	xyzouvwo[2*pid+1] = make_float4(tmp1.y, tmp2.x, tmp2.y, 0 );
}

int main(int argc, char **argv)
{
	const int L = 48;
	const int ndens = 4;
	const int np = L*L*L*ndens;
	PinnedHostBuffer<Particle>   particles(np);
	SimpleDeviceBuffer<Particle> devp(np);
	SimpleDeviceBuffer<float4>   pWithO(np*2);
	SimpleDeviceBuffer<ushort4>  halfPwithO(np*2);

	SimpleDeviceBuffer<int> cellsstart(L*L*L + 1);
	SimpleDeviceBuffer<int> cellscount(L*L*L);

	PinnedHostBuffer<int> hcellsstart(L*L*L + 1);

	SimpleDeviceBuffer<Acceleration> acc(np);
	PinnedHostBuffer  <Acceleration> hacc(np);

	srand48(0);

	printf("initializing...\n");

	int c = 0;
	for (int i=0; i<L; i++)
		for (int j=0; j<L; j++)
			for (int k=0; k<L; k++)
				for (int p=0; p<ndens; p++)
				{
					particles.data[c].x[0] = i + drand48() - L/2;
					particles.data[c].x[1] = j + drand48() - L/2;
					particles.data[c].x[2] = k + drand48() - L/2;

					particles.data[c].u[0] = drand48() - 0.5;
					particles.data[c].u[1] = drand48() - 0.5;
					particles.data[c].u[2] = drand48() - 0.5;
					c++;
				}

	printf("building cells...\n");

	build_clists((float*)particles.devptr, np, 1.0, L, L, L, -L/2.0, -L/2.0, -L/2.0, nullptr, cellsstart.data, cellscount.data);
	CUDA_CHECK( cudaMemcpy(devp.data, particles.devptr, particles.size * sizeof(Particle), cudaMemcpyDeviceToDevice) );
	CUDA_CHECK( cudaMemcpy(hcellsstart.devptr, cellsstart.data, cellsstart.size * sizeof(int), cudaMemcpyDeviceToDevice) );

	//	for (int i = 0; i<np; i++)
	//		printf("  %f %f %f\n", particles.data[i].x[0], particles.data[i].x[1], particles.data[i].x[2]);
	//
	//
	//	for (int i = 0; i< L*L*L; i++)
	//		printf("start %d,  count %d\n", cellsstart.data[i], cellscount.data[i]);

	printf("making textures\n");

	//make_texture <<< (np + 1023) / 1024, 1024, 1024 * 6 * sizeof( float )>>>(pWithO.data, halfPwithO.data, (float *)particles.devptr, np );
	make_texture1  <<< (np + 1023) / 1024, 1024 >>>(pWithO.data, halfPwithO.data, (float2 *)particles.devptr, np );

	const float dt = 0.0025;
	const float kBT = 1.0;
	const float gammadpd = 20;
	const float sigma = sqrt(2 * gammadpd * kBT);
	const float sigmaf = 126.4911064;//sigma / sqrt(dt);
	const float aij = 50;

	for (int i=0; i<2; i++)
	{
		printf("running it # %d\n", i);
		forces_dpd_cuda_nohost((float*)devp.data, pWithO.data, halfPwithO.data, (float*)acc.data, np, cellsstart.data, cellscount.data, 1.0, L, L, L, aij, gammadpd, sigma, 1.0/sqrt(dt), 1, 0);
		CUDA_CHECK( cudaPeekAtLastError() );
	}

	CUDA_CHECK( cudaMemcpy(hacc.devptr, acc.data, acc.size * sizeof(Acceleration), cudaMemcpyDeviceToDevice) );
	cudaDeviceSynchronize();
	printf("finised, reducing acc\n");
	double a[3] = {};
	for (int i=0; i<np; i++)
	{
		for (int c=0; c<3; c++)
			a[c] += hacc.data[i].a[c];
	}
	printf("Reduced acc: %e %e %e\n\n", a[0], a[1], a[2]);


	printf("Checking (this is only a cubic domain, careful)......\n");
	std::vector<Acceleration> refAcc(acc.size);



	auto addForce = [&](int dstId, int srcId, Acceleration& a)
	{
		const float _xr = particles.data[dstId].x[0] - particles.data[srcId].x[0];
		const float _yr = particles.data[dstId].x[1] - particles.data[srcId].x[1];
		const float _zr = particles.data[dstId].x[2] - particles.data[srcId].x[2];

		const float rij2 = _xr * _xr + _yr * _yr + _zr * _zr;

		if (rij2 > 1.0f) return;
		//assert(rij2 < 1);

		const float invrij = 1.0f / sqrt(rij2);
		const float rij = rij2 * invrij;
		const float argwr = 1.0f - rij;
		const float wr = argwr;

		const float xr = _xr * invrij;
		const float yr = _yr * invrij;
		const float zr = _zr * invrij;

		const float rdotv =
				xr * (particles.data[dstId].u[0] - particles.data[srcId].u[0]) +
				yr * (particles.data[dstId].u[1] - particles.data[srcId].u[1]) +
				zr * (particles.data[dstId].u[2] - particles.data[srcId].u[2]);

		const float myrandnr = 0;//Logistic::mean0var1(1, min(srcId, dstId), max(srcId, dstId));

		const float strength = aij * argwr - (gammadpd * wr * rdotv + sigmaf * myrandnr) * wr;

		a.a[0] += strength * xr;
		a.a[1] += strength * yr;
		a.a[2] += strength * zr;
	};

#pragma omp parallel for collapse(3)
	for (int cx = 0; cx < L; cx++)
		for (int cy = 0; cy < L; cy++)
			for (int cz = 0; cz < L; cz++)
			{
				const int cid = (cz*L + cy)*L + cx;

				for (int dstId = hcellsstart.data[cid]; dstId < hcellsstart.data[cid+1]; dstId++)
				{
					Acceleration a {0,0,0};

					for (int dx = -1; dx <= 1; dx++)
						for (int dy = -1; dy <= 1; dy++)
							for (int dz = -1; dz <= 1; dz++)
							{
								const int srcCid = ( (cz+dz)*L + (cy+dy) ) * L + cx+dx;
								if (srcCid >= L*L*L || srcCid < 0) continue;

								for (int srcId = hcellsstart.data[srcCid]; srcId < hcellsstart.data[srcCid+1]; srcId++)
								{
									if (dstId != srcId)
										addForce(dstId, srcId, a);
								}
							}

					refAcc[dstId].a[0] = a.a[0];
					refAcc[dstId].a[1] = a.a[1];
					refAcc[dstId].a[2] = a.a[2];
				}
			}


	double l2 = 0, linf = -1;

	for (int i=0; i<np; i++)
	{
		double perr = -1;
		for (int c=0; c<3; c++)
		{
			const double err = fabs(refAcc[i].a[c] - hacc.data[i].a[c]);
			linf = max(linf, err);
			perr = max(perr, err);
			l2 += err * err;
		}

		if (argc > 1 && perr > 0.1)
		{
			printf("id %d,  %12f %12f %12f     ref %12f %12f %12f    diff   %12f %12f %12f\n", i,
				hacc.data[i].a[0], hacc.data[i].a[1], hacc.data[i].a[2],
				refAcc[i].a[0], refAcc[i].a[1], refAcc[i].a[2],
				hacc.data[i].a[0]-refAcc[i].a[0], hacc.data[i].a[1]-refAcc[i].a[1], hacc.data[i].a[2]-refAcc[i].a[2]);


			int dstId = i;

			int cx = floor(particles.data[dstId].x[0]) + L/2;
			int cy = floor(particles.data[dstId].x[1]) + L/2;
			int cz = floor(particles.data[dstId].x[2]) + L/2;

			printf(" [%f %f %f] --> %d %d %d\n", particles.data[dstId].x[0], particles.data[dstId].x[1], particles.data[dstId].x[2], cx, cy, cz);

			const int cid = (cz*L + cy)*L + cx;


			for (int dx = -1; dx <= 1; dx++)
				for (int dy = -1; dy <= 1; dy++)
					for (int dz = -1; dz <= 1; dz++)
					{
						const int srcCid = ( (cz+dz)*L + (cy+dy) ) * L + cx+dx;
						if (srcCid >= L*L*L || srcCid < 0) continue;

						for (int srcId = hcellsstart.data[srcCid]; srcId < hcellsstart.data[srcCid+1]; srcId++)
						{
							Acceleration a {hacc.data[i].a[0], hacc.data[i].a[1], hacc.data[i].a[2]};

							if (dstId != srcId)
								addForce(dstId, srcId, a);

							double nerr = fabs(refAcc[i].a[0] - a.a[0]) + fabs(refAcc[i].a[1] - a.a[1]) + fabs(refAcc[i].a[2] - a.a[2]);

							if (nerr < perr)
								printf("\t Possible correction %d (cell %d [%d %d %d]): %f   ==>   %f %f %f\n", srcId, srcCid, dx, dy, dz, nerr,
										a.a[0]-hacc.data[i].a[0], a.a[1]-hacc.data[i].a[1], a.a[2]-hacc.data[i].a[2]);
						}
					}
			printf("\n\n");
		}
	}


	l2 = sqrt(l2 / np);
	printf("L2   norm: %f\n", l2);
	printf("Linf norm: %f\n", linf);

	CUDA_CHECK( cudaPeekAtLastError() );
	return 0;
}
