/*
 *  main-cuda.cpp
 *  ctc phenix
 *
 *  Created by Dmitry Alexeev on Nov 10, 2014
 *  Copyright 2014 ETH Zurich. All rights reserved.
 *
 */


/*
 *  main.cpp
 *  ctc local
 *
 *  Created by Dmitry Alexeev on Nov 5, 2014
 *  Copyright 2014 ETH Zurich. All rights reserved.
 *
 */

#include "rbc-cuda.h"
#include "timer.h"
#include "misc.h"
#include "cuda-common.h"


#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>


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

__global__ void _diag_kbt(const real * const xyzuvw, real * const diag, const int n)
{
	const int tid = threadIdx.x + blockDim.x * blockIdx.x;

	if (tid < n)
		diag[tid] =
				pow(xyzuvw[3 + 6 * tid], 2) +
				pow(xyzuvw[4 + 6 * tid], 2) +
				pow(xyzuvw[5 + 6 * tid], 2);
}

__global__ void _diag_p(const real * const xyzuvw, real * const diag, const int n, const int c)
{
	const int tid = threadIdx.x + blockDim.x * blockIdx.x;

	if (tid < n)
		diag[tid] = xyzuvw[3 + c + 6 * tid];
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

class SimRBC
{
	int nparticles;
	const real L;
	real *xyzuvw, *fxfyfz;

public:

	SimRBC(const real L): L(L)
{
		CudaRBC::Extent extent;
		CudaRBC::setup(nparticles, extent);
		printf("Original extent:  [%.3f  %.3f  %.3f] -- [%.3f  %.3f  %.3f]\n",
				extent.xmin, extent.ymin, extent.zmin,
				extent.xmax, extent.ymax, extent.zmax);

		gpuErrchk( cudaMalloc(&xyzuvw, 6*nparticles*sizeof(real)) );
		gpuErrchk( cudaMalloc(&fxfyfz, 3*nparticles*sizeof(real)) );

		float A[4][4];
		memset(&A[0][0], 0, 16*sizeof(float));
		A[0][0] = A[1][1] = A[2][2] = A[3][3] = 1;
		A[1][3] = 2;

		CudaRBC::initialize(xyzuvw, A);
		printf("initialized\n");

		CudaRBC::Extent * devext;
		gpuErrchk( cudaMalloc(&devext, sizeof(CudaRBC::Extent)) );
		CudaRBC::extent_nohost(0, xyzuvw, devext);
		gpuErrchk( cudaMemcpy(&extent, devext, sizeof(CudaRBC::Extent), cudaMemcpyDeviceToHost) );
		printf("New extent (y+=2):  [%.3f  %.3f  %.3f] -- [%.3f  %.3f  %.3f]\n",
				extent.xmin, extent.ymin, extent.zmin,
				extent.xmax, extent.ymax, extent.zmax);

		int ntr;
		int (*trs)[3];
		CudaRBC::get_triangle_indexing(trs, ntr);

		for (int i=0; i<ntr; i++)
		{
			printf("%3d: [%3d %3d %3d]  ", i, trs[i][0], trs[i][1], trs[i][2]);
			if ((i+1)%6 == 0) printf("\n");
		}
		printf("\n");

}

	void _diag(FILE ** fs, const int nfs, real t)
	{

	}

	void _f()
	{
		gpuErrchk( cudaMemset(fxfyfz, 0, 3*nparticles * sizeof(real)) );
		CudaRBC::forces_nohost(0, xyzuvw, fxfyfz);
	};

	void run(const real tend, const real dt)
	{
		vmd_xyz("ic.xyz", xyzuvw, nparticles, false);

		FILE * fdiags[2] = {stdout, fopen("diag.txt", "w") };

		const size_t nt = 1;//(int)(tend / dt);

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

			_update_vel<<<(nparticles + 127) / 128, 128>>>(xyzuvw, fxfyfz, dt * 0.5, nparticles);


			_update_pos<<<(nparticles + 127) / 128, 128>>>(xyzuvw, dt, nparticles, L);


			_f();

			_update_vel<<<(nparticles + 127) / 128, 128>>>(xyzuvw, fxfyfz, dt * 0.5, nparticles);

			if (it % 500 == 0)
			{
				vmd_xyz("evolution.xyz", xyzuvw, nparticles, it > 0);
				//vmd_xyz_3comp("force.xyz", fxfyfz, nparticles, it > 0);
			}
		}

		printf("Avg time per step is is %f  ms\n", tm.elapsed() / 1e6 / nt);

		fclose(fdiags[1]);
	}
};

int main()
{
	printf("hello rbc-gpu test\n");

	real L = 10; //  /Volumes/Phenix/CTC/vanilla-rbc/evolution.xyz

	SimRBC sim(L);

	sim.run(10, 0.001);

	return 0;
}

