#pragma once

#include "interface.h"

#include <mirheo/core/datatypes.h>
#include <mirheo/core/domain.h>

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
    MembraneIC(const std::vector<ComQ>& com_q, real globalScale = 1.0_r);
    ~MembraneIC();

    void exec(const MPI_Comm& comm, ParticleVector *pv, cudaStream_t stream) override;

protected:
    std::vector<int> createMap(DomainInfo domain) const;
    
private:
    std::vector<ComQ> com_q;
    real globalScale;
};
