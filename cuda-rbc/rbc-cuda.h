/*
 *  rbc.h
 *  ctc local
 *
 *  Created by Dmitry Alexeev on Nov 3, 2014
 *  Copyright 2014 ETH Zurich. All rights reserved.
 *
 */

#pragma once

#include <cuda_runtime.h>

using namespace std;

namespace CudaRBC
{
    struct Params
    {
        float kbT, p, lmax, q, Cq, totArea0, totVolume0,
        ka, kv, gammaT, gammaC,  sinTheta0, cosTheta0, kb, l0;
        float sint0kb, cost0kb, kbToverp;
        int  nvertices, ntriangles;
    };

	struct Extent
	{
		float xmin, ymin, zmin;
		float xmax, ymax, zmax;
	};

	static Params params;
	static __constant__ Params devParams;

	/* blocking, initializes params */
	void setup(int& nvertices, Extent& host_extent);

	int get_nvertices();

	/* A * (x, 1) */
	void initialize(float *device_xyzuvw, const float (*transform)[4]);

	/* non-synchronizing */
	void forces_nohost(cudaStream_t stream, int ncells, const float * const device_xyzuvw, float * const device_axayaz);

	/*non-synchronizing, extent not initialized */
	void extent_nohost(cudaStream_t stream, int ncells, const float * const xyzuvw, Extent * device_extent, int n = -1);

	/* get me a pointer to YOUR plain array - no allocation on my side */
	void get_triangle_indexing(int (*&host_triplets_ptr)[3], int& ntriangles);
};
