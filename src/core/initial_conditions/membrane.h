#pragma once

#include "interface.h"

#include <core/datatypes.h>

#include <mpi.h>
#include <string>
#include <vector>
#include <vector_types.h>

class ParticleVector;

/**
 * Initialize membranes.
 */
class MembraneIC : public InitialConditions
{
public:
    MembraneIC(const std::vector<ComQ>& com_q, real globalScale = 1.0f);
    ~MembraneIC();

    void exec(const MPI_Comm& comm, ParticleVector *pv, cudaStream_t stream) override;

private:
    std::vector<ComQ> com_q;
    real globalScale;
};
