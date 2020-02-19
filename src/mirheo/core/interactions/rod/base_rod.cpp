#include "base_rod.h"

#include <mirheo/core/pvs/particle_vector.h>
#include <mirheo/core/logger.h>

namespace mirheo
{

BaseRodInteraction::BaseRodInteraction(const MirState *state, const std::string& name) :
    Interaction(state, name)
{}

BaseRodInteraction::~BaseRodInteraction() = default;
    
void BaseRodInteraction::halo(ParticleVector *pv1,
                              __UNUSED ParticleVector *pv2,
                              __UNUSED CellList *cl1,
                              __UNUSED CellList *cl2,
                              __UNUSED cudaStream_t stream)
{
    debug("Not computing internal rod forces between local and halo rods of '%s'", pv1->getCName());
}

bool BaseRodInteraction::isSelfObjectInteraction() const
{
    return true;
}


} // namespace mirheo
