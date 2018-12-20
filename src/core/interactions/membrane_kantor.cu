#include <core/pvs/membrane_vector.h>
#include <core/pvs/views/ov.h>
#include <core/utils/cuda_common.h>
#include <core/utils/kernel_launch.h>

#include "membrane_kantor.h"
#include "membrane/bending_kantor.h"

InteractionMembraneKantor::InteractionMembraneKantor(std::string name, const YmrState *state,
                                                     MembraneParameters parameters,
                                                     KantorBendingParameters bendingParameters,
                                                     bool stressFree, float growUntil) :
    InteractionMembrane(name, state, parameters, stressFree, growUntil),
    bendingParameters(bendingParameters)
{}


InteractionMembraneKantor::~InteractionMembraneKantor() = default;

static bendingKantor::GPU_BendingParams setKantorBendingParams(float scale, KantorBendingParameters& p)
{
    bendingKantor::GPU_BendingParams devP;
    
    devP.cost0kb = cos(p.theta / 180.0 * M_PI) * p.kb * scale*scale;
    devP.sint0kb = sin(p.theta / 180.0 * M_PI) * p.kb * scale*scale;

    return devP;
}

void InteractionMembraneKantor::bendingForces(float scale, MembraneVector *ov, MembraneMeshView mesh, cudaStream_t stream)
{
    OVview view(ov, ov->local());

    const int nthreads = 128;
    const int blocks = getNblocks(view.size, nthreads);

    auto devParams = setKantorBendingParams(scale, bendingParameters);

    SAFE_KERNEL_LAUNCH(
            bendingKantor::computeBendingForces,
            blocks, nthreads, 0, stream,
            view, mesh, devParams );    
}
