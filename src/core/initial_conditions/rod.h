#pragma once

#include "interface.h"

#include <core/utils/pytypes.h>

#include <functional>
#include <string>

/**
 * Initialize rods.
 */
class RodIC : public InitialConditions
{
public:

    using MappingFunc3D = std::function<PyTypes::float3(float)>;
    using MappingFunc1D = std::function<float(float)>;
    
    RodIC(PyTypes::VectorOfFloat7 com_q, MappingFunc3D centerLine, MappingFunc1D torsion);
    ~RodIC();
    
    void exec(const MPI_Comm& comm, ParticleVector *pv, cudaStream_t stream) override;    

private:
    MappingFunc3D centerLine;
    MappingFunc1D torsion;
    PyTypes::VectorOfFloat7 com_q;
};
