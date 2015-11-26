/*
 *  dpd-interactions.h
 *  Part of uDeviceX/mpi-dpd/
 *
 *  Created and authored by Diego Rossinelli on 2014-11-14.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */

#pragma once

#include <utility>
#include <mpi.h>

#include <map>
#include <string>

#include <../dpd-rng.h>

#include "common.h"
#include "solvent-exchange.h"

//see the vanilla version of this code for details about how this class operates
class ComputeDPD : public SolventExchange
{
    Logistic::KISS local_trunk;
    Logistic::KISS interrank_trunks[26];

    float current_lseed, current_rseeds[26],
	* sigma_xx, * sigma_xy, * sigma_xz, * sigma_yy,
	* sigma_yz, * sigma_zz;

    bool interrank_masks[26];

public:

    ComputeDPD(MPI_Comm cartcomm);

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

    void remote_interactions(const Particle * const p, const int n, Acceleration * const a, cudaStream_t stream, cudaStream_t uploadstream);

    void local_interactions(const Particle * const xyzuvw, const float4 * const xyzouvwo, const ushort4 * const xyzo_half, const int n,
			    Acceleration * const a, const int * const cellsstart, const int * const cellscount, cudaStream_t stream);
};
