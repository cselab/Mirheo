#pragma once

#include "interface.h"

#include <core/utils/pytypes.h>

class RigidIC : public InitialConditions
{
private:
    std::string xyzfname;
    ICvector com_q;

public:
    RigidIC(ICvector com_q, std::string xyzfname);

    void exec(const MPI_Comm& comm, ParticleVector* pv, DomainInfo domain, cudaStream_t stream) override;

    ~RigidIC();
};
