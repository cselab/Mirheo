#pragma once

#include "containers.h"

void computeInternalDPD(ParticleVector& pv, cudaStream_t stream);
void computeHaloDPD(ParticleVector& pv, cudaStream_t stream);
