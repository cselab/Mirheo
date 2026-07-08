// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "base_triplewise.h"

#include <mirheo/core/celllist.h>
#include <mirheo/core/logger.h>

namespace mirheo
{

BaseTriplewiseInteraction::BaseTriplewiseInteraction(const MirState *state, const std::string& name, real rc) :
    Interaction(state, name),
    rc_(rc)
{}

BaseTriplewiseInteraction::~BaseTriplewiseInteraction() = default;

void BaseTriplewiseInteraction::local(__UNUSED ParticleVector *pv1, __UNUSED ParticleVector *pv2,
                                      __UNUSED CellList *cl1, __UNUSED CellList *cl2,
                                      __UNUSED cudaStream_t stream)
{
    die("triplewise interaction '%s' must be invoked with three particle vectors", getCName());
}

void BaseTriplewiseInteraction::halo(__UNUSED ParticleVector *pv1, __UNUSED ParticleVector *pv2,
                                     __UNUSED CellList *cl1, __UNUSED CellList *cl2,
                                     __UNUSED cudaStream_t stream)
{
    die("triplewise interaction '%s' must be invoked with three particle vectors", getCName());
}

std::optional<real> BaseTriplewiseInteraction::getCutoffRadius() const
{
    // local-halo-halo particles have a reach of 2*rc, see base_triplewise.h
    return 2 * rc_;
}

BaseTriplewiseInteraction::CellListPair::CellListPair(
        ParticleVector *pv, real rc, const CellList *ref) :
    refinedLocal(pv, rc, ref->localDomainSize, ParticleVectorLocality::Local),
    halo(pv, rc, ref->localDomainSize + make_real3(4 * rc),  // 2*rc on each side
         ParticleVectorLocality::Halo)
{}

BaseTriplewiseInteraction::CellListPair *BaseTriplewiseInteraction::_getOrCreateCellLists(
        ParticleVector *pv, const CellList *refCL)
{
    const auto it = cellLists_.find(pv);
    if (it != cellLists_.end())
        return &it->second;
    const auto newIt = cellLists_.emplace(
            std::piecewise_construct,
            std::make_tuple(pv),                    // ParticleVector *
            std::make_tuple(pv, rc_, refCL)).first; // CellListPair
    return &newIt->second;
}

} // namespace mirheo
