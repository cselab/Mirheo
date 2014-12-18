/*
 *  main.cpp
 *  ctc local
 *
 *  Created by Dmitry Alexeev on Nov 5, 2014
 *  Copyright 2014 ETH Zurich. All rights reserved.
 *
 */

#include "timer.h"
#include "misc.h"
#include "cuda-common.h"
#include "vec3.h"

#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include "ctc-cuda.h"


using namespace std;

__global__ void _update_pos(real * const xyzuvw, const real f, const int n, const real L)
{
	const int tid = threadIdx.x + blockDim.x * blockIdx.x;

	if (tid < n)
	{
		for(int c = 0; c < 3; ++c)
		{
			const real xold = xyzuvw[c + 6 * tid];

			real xnew = xold + f * xyzuvw[3 + c + 6 * tid];
			xnew -= L * floor((xnew + 0.5 * L) / L);

			xyzuvw[c + 6 * tid] = xnew;
		}
	}
}

__global__ void _update_vel(real * const xyzuvw, const real * const axayaz, const real f, const int n)
{
	const int tid = threadIdx.x + blockDim.x * blockIdx.x;

	if (tid < n)
	{
		for(int c = 0; c < 3; ++c)
		{
			const real vold = xyzuvw[3 + c + 6 * tid];

			real vnew = vold + f * axayaz[c + 3 * tid];

			xyzuvw[3 + c + 6 * tid] = vnew;
		}
	}
}

__global__ void diagKernel(const float* const xyzuvw, int n, float * const diag)
{
	// assume diag[i] = 0;
	vec3 vtot(0.0f, 0.0f, 0.0f);
	float kbT = 0;

	for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
	{
		vec3 v(xyzuvw+6*i+3);
		vtot += v;
		kbT += 0.5*dot(v, v)/n;
	}

	vtot = warpReduceSum(vtot);
	__syncthreads();
	kbT = warpReduceSum(kbT);

	if ((threadIdx.x & (warpSize - 1)) == 0)
	{
		atomicAdd(&diag[0], vtot.x);
		atomicAdd(&diag[1], vtot.y);
		atomicAdd(&diag[2], vtot.z);
		atomicAdd(&diag[3], kbT);
	}
}

void vmd_xyz(const char * path, real* _xyzuvw, const int n, bool append)
{
	real* xyzuvw = new real[6*n];
	gpuErrchk( cudaMemcpy(xyzuvw, _xyzuvw, 6*n * sizeof(real), cudaMemcpyDeviceToHost) );

	FILE * f = fopen(path, append ? "a" : "w");

	if (f == NULL)
	{
		printf("I could not open the file <%s>\n", path);
		printf("Aborting now.\n");
		abort();
	}

	fprintf(f, "%d\n", n);
	fprintf(f, "mymolecule\n");

	for(int i = 0; i < n; ++i)
		fprintf(f, "1 %f %f %f\n",
				(real)xyzuvw[0 + 6 * i],
				(real)xyzuvw[1 + 6 * i],
				(real)xyzuvw[2 + 6 * i]);

	fclose(f);
	delete[] xyzuvw;

	printf("vmd_xyz: wrote to <%s>\n", path);
}

void vmd_xyz_3comp(const char * path, real* _xyz, const int n, bool append)
{
	real* xyz = new real[3*n];
	gpuErrchk( cudaMemcpy(xyz, _xyz, 3*n * sizeof(real), cudaMemcpyDeviceToHost) );

	FILE * f = fopen(path, append ? "a" : "w");

	if (f == NULL)
	{
		printf("I could not open the file <%s>\n", path);
		printf("Aborting now.\n");
		abort();
	}

	fprintf(f, "%d\n", n);
	fprintf(f, "mymolecule\n");

	for(int i = 0; i < n; ++i)
		fprintf(f, "%d %f %f %f\n", i,
				(real)xyz[0 + 3 * i],
				(real)xyz[1 + 3 * i],
				(real)xyz[2 + 3 * i]);

	fclose(f);

	printf("vmd_xyz: wrote to <%s>\n", path);
}

__global__ void setVels(real *xyzuvw, real vx, real vy, real vz, int n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;

	xyzuvw[6*i + 3] = vx;
	xyzuvw[6*i + 4] = vy;
	xyzuvw[6*i + 5] = vz;
}


class SimRBC
{
	int nparticles;
	const real L;
	real *xyzuvw, *fxfyfz;
	real dt;

	int ncells;

public:

	SimRBC(const real L, const real dt): L(L), dt(dt)
{
		CudaCTC::Extent extent;
		CudaCTC::setup(nparticles, extent, dt);
		printf("Original extent:  [%.3f  %.3f  %.3f] -- [%.3f  %.3f  %.3f]\n",
				extent.xmin, extent.ymin, extent.zmin,
				extent.xmax, extent.ymax, extent.zmax);

		ncells = 2;
		gpuErrchk( cudaMalloc(&xyzuvw, 6*ncells*nparticles*sizeof(real)) );
		gpuErrchk( cudaMalloc(&fxfyfz, 3*ncells*nparticles*sizeof(real)) );

		float A[4][4];
		memset(&A[0][0], 0, 16*sizeof(float));
		A[0][0] = A[1][1] = A[2][2] = A[3][3] = 1;
		CudaCTC::initialize(xyzuvw, A);

		A[1][3] = 10;
		CudaCTC::initialize(xyzuvw + 6*nparticles, A);
		setVels<<<(nparticles + 127) / 128, 128>>>(xyzuvw + 6*nparticles, 0, -0.2, 0, nparticles);

		printf("initialized\n");

		CudaCTC::Extent * devext;
		gpuErrchk( cudaMalloc(&devext, sizeof(CudaCTC::Extent)) );
		CudaCTC::extent_nohost(0, xyzuvw, devext);
		gpuErrchk( cudaMemcpy(&extent, devext, sizeof(CudaCTC::Extent), cudaMemcpyDeviceToHost) );
		printf("New extent (y+=2):  [%.3f  %.3f  %.3f] -- [%.3f  %.3f  %.3f]\n",
				extent.xmin, extent.ymin, extent.zmin,
				extent.xmax, extent.ymax, extent.zmax);

		int ntr;
		int (*trs)[3];
		CudaCTC::get_triangle_indexing(trs, ntr);

//		for (int i=0; i<ntr; i++)
//		{
//			printf("%3d: [%3d %3d %3d]  ", i, trs[i][0], trs[i][1], trs[i][2]);
//			if ((i+1)%6 == 0) printf("\n");
//		}
//		printf("\n");

}

	void _diag(FILE ** fs, const int nfs, real t)
	{
		static float *diag, *devDiag;
		static bool inited = false;

		if (!inited)
		{
			gpuErrchk( cudaMalloc    (&devDiag, 4*sizeof(float)) );
			gpuErrchk( cudaMallocHost(&diag,    4*sizeof(float)) );

			inited = true;
		}

		gpuErrchk( cudaMemset(devDiag, 0, 4*sizeof(float)) );

		diagKernel<<<(ncells*nparticles + 127) / 128, 128>>>(xyzuvw, ncells*nparticles, devDiag);

		gpuErrchk( cudaMemcpy(diag, devDiag, 4*sizeof(float), cudaMemcpyDeviceToHost) );

		for (int f=0; f<nfs; f++)
		{
			fprintf(fs[f], "%10.5f  %10.5f  %10.5f  %10.5f  %10.5f\n", t, diag[3], diag[0], diag[1], diag[2]);
			fflush(fs[f]);
		}
	}

	void _f()
	{
		gpuErrchk( cudaMemset(fxfyfz, 0, 3*ncells*nparticles * sizeof(real)) );
		CudaCTC::forces_nohost(0, xyzuvw, fxfyfz);
		CudaCTC::forces_nohost(0, xyzuvw + 6*nparticles, fxfyfz + 3*nparticles);
		//CudaCTC::interforce_nohost(0, xyzuvw, ncells, fxfyfz, 0);
	};

	void run(const real tend)
	{
		vmd_xyz("ic.xyz", xyzuvw, nparticles*ncells, false);

		FILE * fdiags[2] = {stdout, fopen("diag.txt", "w") };

		const size_t nt = (int)(tend / dt);

		_f();

		Timer tm;
		tm.start();
		for(int it = 0; it < nt; ++it)
		{
			//			if (it % 200 == 0)
			//			{
			//				real t = it * dt;
			//				_diag(fdiags, 2, t);
			//			}

			_update_vel<<<(ncells*nparticles + 127) / 128, 128>>>(xyzuvw, fxfyfz, dt * 0.5, ncells*nparticles);


			_update_pos<<<(ncells*nparticles + 127) / 128, 128>>>(xyzuvw, dt, ncells*nparticles, L);


			_f();

			_update_vel<<<(ncells*nparticles + 127) / 128, 128>>>(xyzuvw, fxfyfz, dt * 0.5, ncells*nparticles);

			if (it % 500 == 0)
			{
				_diag(fdiags, 2, it*dt);
				vmd_xyz("evolution.xyz", xyzuvw, nparticles*ncells, it > 0);
				//vmd_xyz_3comp("force.xyz", fxfyfz, nparticles, it > 0);
			}
		}

		printf("Avg time per step is is %f  ms\n", tm.elapsed() / 1e6 / nt);

		fclose(fdiags[1]);
	}
};

int main()
{
	printf("hello ctc-gpu test\n");

	real L = 100; //  /Volumes/Phenix/CTC/vanilla-rbc/evolution.xyz

	SimRBC sim(L, 0.001);

	sim.run(100);

	return 0;
}

