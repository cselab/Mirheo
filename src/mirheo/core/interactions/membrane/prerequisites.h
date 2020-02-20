#pragma once

#include "force_kernels/parameters.h"

#include <mirheo/core/utils/macros.h>

#include <cuda_runtime.h>

namespace mirheo
{
class MembraneVector;

/** \brief Set prerequisites on a MembraneVector for a given energy term.
    \tparam EnergyParams The energy term type
    \param [in] params energy parameters
    \param [in,out] mv The MembraneVector that will get prerequisites.
 */
template <class EnergyParams>
void setPrerequisitesPerEnergy(__UNUSED const EnergyParams& params,
                               __UNUSED MembraneVector *mv)
{}

/// \overload
void setPrerequisitesPerEnergy(const JuelicherBendingParameters& params, MembraneVector *mv);


/** \brief Compute prerequired quantities needed by a specific interaction for a MembraneVector
    \tparam EnergyParams The energy term type
    \param [in] params energy parameters
    \param [in,out] mv The MembraneVector that will store the precomputed quantities.
 */
template <class EnergyParams>
void precomputeQuantitiesPerEnergy(__UNUSED const EnergyParams&,
                                   __UNUSED MembraneVector *mv1,
                                   __UNUSED cudaStream_t stream)
{}

/// \overload
void precomputeQuantitiesPerEnergy(const JuelicherBendingParameters&, MembraneVector *pv, cudaStream_t stream);

} // namespace mirheo
