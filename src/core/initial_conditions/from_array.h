#pragma once

#include <core/utils/pytypes.h>
#include "interface.h"

class FromArrayIC : public InitialConditions
{
private:
    PyTypes::VectorOfFloat3 pos, vel;

public:
    FromArrayIC(const PyTypes::VectorOfFloat3 &pos, const PyTypes::VectorOfFloat3 &vel);

    void exec(const MPI_Comm& comm, ParticleVector *pv, cudaStream_t stream) override;
};

