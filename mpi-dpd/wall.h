/*
 *  wall.h
 *  Part of uDeviceX/mpi-dpd/
 *
 *  Created and authored by Diego Rossinelli on 2014-11-19.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */

#pragma once

#include <mpi.h>

#include <../dpd-rng.h>
#include "common.h"

namespace SolidWallsKernel
{
    __global__ void fill_keys(const Particle * const particles, const int n, int * const key);
}

class ComputeWall
{
    MPI_Comm cartcomm;
    int myrank, dims[3], periods[3], coords[3];

    Logistic::KISS trunk;

    int solid_size;
    float4 * solid4;
    float * sigma_xx, * sigma_xy, * sigma_xz, * sigma_yy,
	* sigma_yz, * sigma_zz;

    cudaArray * arrSDF;

    CellLists cells;

public:

    ComputeWall(MPI_Comm cartcomm, Particle* const p, const int n, int& nsurvived, ExpectedMessageSizes& new_sizes, const bool verbose);

    ~ComputeWall();

    void bounce(Particle * const p, const int n, cudaStream_t stream);

    void set_stress_buffers(float * const stress_xx, float * const stress_xy, float * const stress_xz, float * const stress_yy,
			    float * const stress_yz, float * const stress_zz)
    {
	sigma_xx = stress_xx;
	sigma_xy = stress_xy;
	sigma_xz = stress_xz;
	sigma_yy = stress_yy;
	sigma_yz = stress_yz;
	sigma_zz = stress_zz;
    }

    void clr_stress_buffers()
    {
	sigma_xx = NULL;
	sigma_xy = NULL;
	sigma_xz = NULL;
	sigma_yy = NULL;
	sigma_yz = NULL;
	sigma_zz = NULL;
    }

    void interactions(const Particle * const p, const int n, Acceleration * const acc,
		      const int * const cellsstart, const int * const cellscount, cudaStream_t stream);
};
