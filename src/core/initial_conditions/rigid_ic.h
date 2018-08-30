#pragma once

#include "interface.h"

#include <core/utils/pytypes.h>

class RigidIC : public InitialConditions
{
private:
    PyContainer coords;
    ICvector com_q;

public:
    RigidIC(ICvector com_q, std::string xyzfname);
    RigidIC(ICvector com_q, const PyContainer& coords);

    void exec(const MPI_Comm& comm, ParticleVector* pv, DomainInfo domain, cudaStream_t stream) override;

    ~RigidIC();
};
