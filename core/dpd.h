#pragma once

#include "containers.h"
#include "celllist.h"

void computeInternalDPD(ParticleVector& pv, CellList& cl, cudaStream_t stream);
void computeHaloDPD(ParticleVector& pv, CellList& cl, cudaStream_t stream);
