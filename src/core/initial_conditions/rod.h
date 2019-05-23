#pragma once

#include "interface.h"

#include <core/utils/pytypes.h>

#include <functional>
#include <limits>
#include <string>

/**
 * Initialize rods.
 */
class RodIC : public InitialConditions
{
public:
    constexpr static float Default = std::numeric_limits<float>::infinity();
    static constexpr PyTypes::float3 DefaultFrame {Default, Default, Default};
    
    using MappingFunc3D = std::function<PyTypes::float3(float)>;
    using MappingFunc1D = std::function<float(float)>;
    
    RodIC(PyTypes::VectorOfFloat7 com_q, MappingFunc3D centerLine, MappingFunc1D torsion, float a,
          PyTypes::float3 initialMaterialFrame = DefaultFrame);
    ~RodIC();
    
    void exec(const MPI_Comm& comm, ParticleVector *pv, cudaStream_t stream) override;    

private:
    PyTypes::VectorOfFloat7 com_q;
    MappingFunc3D centerLine;
    MappingFunc1D torsion;
    float3 initialMaterialFrame;
    float a;
};
