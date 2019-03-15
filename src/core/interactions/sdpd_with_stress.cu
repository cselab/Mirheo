#include "sdpd_with_stress.h"

#include "pairwise_with_stress.impl.h"
#include "pairwise_interactions/sdpd.h"

#include <core/utils/make_unique.h>
#include <core/pvs/particle_vector.h>

BasicInteractionSDPDWithStress::BasicInteractionSDPDWithStress(const YmrState *state, std::string name, float rc,
                                                               float viscosity, float kBT, float stressPeriod) :
    BasicInteractionSDPD(state, name, rc, viscosity, kBT)
{}

BasicInteractionSDPDWithStress::~BasicInteractionSDPDWithStress() = default;



template <class PressureEOS, class DensityKernel>
InteractionSDPDWithStress<PressureEOS, DensityKernel>::
InteractionSDPDWithStress(const YmrState *state, std::string name, float rc,
                          PressureEOS pressure, DensityKernel densityKernel,
                          float viscosity, float kBT, float stressPeriod) :
    BasicInteractionSDPDWithStress(state, name, rc, viscosity, kBT, stressPeriod)
{
    using pairwiseType = PairwiseSDPD<PressureEOS, DensityKernel>;
    pairwiseType sdpd(rc, pressure, densityKernel, viscosity, kBT, state->dt);
    impl = std::make_unique<InteractionPair_withStress<pairwiseType>> (state, name, rc, stressPeriod, sdpd);
}
    
    
template <class PressureEOS, class DensityKernel>
InteractionSDPDWithStress<PressureEOS, DensityKernel>::
~InteractionSDPDWithStress() = default;

template class InteractionSDPDWithStress<LinearPressureEOS, WendlandC2DensityKernel>;
template class InteractionSDPDWithStress<QuasiIncompressiblePressureEOS, WendlandC2DensityKernel>;
