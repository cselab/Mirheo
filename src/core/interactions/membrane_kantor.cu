#include <core/pvs/membrane_vector.h>
#include <core/pvs/views/ov.h>
#include <core/utils/cuda_common.h>
#include <core/utils/kernel_launch.h>

#include "membrane_kantor.h"
#include "membrane/bending_kantor.h"

InteractionMembraneKantor::InteractionMembraneKantor(const YmrState *state, std::string name,
                                                     MembraneParameters parameters,
                                                     KantorBendingParameters bendingParameters,
                                                     bool stressFree, float growUntil) :
    InteractionMembrane(state, name, parameters, stressFree, growUntil),
    bendingParameters(bendingParameters)
{}


InteractionMembraneKantor::~InteractionMembraneKantor() = default;

static BendingKantorKernels::GPU_BendingParams setKantorBendingParams(float scale, KantorBendingParameters& p)
{
    BendingKantorKernels::GPU_BendingParams devP;
    
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
            BendingKantorKernels::computeBendingForces,
            blocks, nthreads, 0, stream,
            view, mesh, devParams );    
}
