/*
 *  dpd-interactions.h
 *  Part of CTC/mpi-dpd/
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
#include "halo-exchanger.h"

//see the vanilla version of this code for details about how this class operates
class ComputeInteractionsDPD : public HaloExchanger
{           
    Logistic::KISS local_trunk;
    Logistic::KISS interrank_trunks[26];

    bool interrank_masks[26];
    
public:
    
    ComputeInteractionsDPD(MPI_Comm cartcomm);

    void remote_interactions(const Particle * const p, const int n, Acceleration * const a, cudaStream_t stream);

    void local_interactions(const float4 * const xyzouvwo, const ushort4 * const xyzo_half, const int n, Acceleration * const a,
			    const int * const cellsstart, const int * const cellscount, cudaStream_t stream);
};
