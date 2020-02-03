#pragma once

#include "interface.h"

#include <mirheo/core/datatypes.h>

#include <functional>
#include <string>
#include <vector>
#include <vector_types.h>

namespace mirheo
{

/**
 * Initialize rods.
 */
class RodIC : public InitialConditions
{
public:
    static const real Default;
    static const real3 DefaultFrame;
    
    using MappingFunc3D = std::function<real3(real)>;
    using MappingFunc1D = std::function<real(real)>;
    
    RodIC(const std::vector<ComQ>& com_q, MappingFunc3D centerLine, MappingFunc1D torsion, real a,
          real3 initialMaterialFrame = DefaultFrame);
    ~RodIC();
    
    void exec(const MPI_Comm& comm, ParticleVector *pv, cudaStream_t stream) override;    

private:
    std::vector<ComQ> comQ_;
    MappingFunc3D centerLine_;
    MappingFunc1D torsion_;
    real3 initialMaterialFrame_;
    real a_;
};

} // namespace mirheo
