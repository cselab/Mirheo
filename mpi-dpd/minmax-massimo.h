#pragma once

#include "common.h"

void minmax_massimo(const Particle * const particles, int nparticles_per_body, int nbodies, 
		    float3 * minextents, float3 * maxextents, cudaStream_t stream);
