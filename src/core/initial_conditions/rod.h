#pragma once

#include "interface.h"

#include <core/datatypes.h>

#include <functional>
#include <string>
#include <vector>
#include <vector_types.h>

/**
 * Initialize rods.
 */
class RodIC : public InitialConditions
{
public:
    static const float Default;
    static const float3 DefaultFrame;
    
    using MappingFunc3D = std::function<float3(float)>;
    using MappingFunc1D = std::function<float(float)>;
    
    RodIC(const std::vector<ComQ>& com_q, MappingFunc3D centerLine, MappingFunc1D torsion, float a,
          float3 initialMaterialFrame = DefaultFrame);
    ~RodIC();
    
    void exec(const MPI_Comm& comm, ParticleVector *pv, cudaStream_t stream) override;    

private:
    std::vector<ComQ> com_q;
    MappingFunc3D centerLine;
    MappingFunc1D torsion;
    float3 initialMaterialFrame;
    float a;
};
