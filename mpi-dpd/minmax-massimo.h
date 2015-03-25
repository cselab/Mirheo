/*
 *  minmax-massimo.h
 *  Part of CTC/mpi-dpd/
 *
 *  Created and authored by Massimo Bernaschi on 2015-03-24.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */

#pragma once

#include "common.h"

void minmax_massimo(const Particle * const particles, int nparticles_per_body, int nbodies, 
		    float3 * minextents, float3 * maxextents, cudaStream_t stream);
