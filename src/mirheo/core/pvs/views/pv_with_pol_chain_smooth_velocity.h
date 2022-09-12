// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include "pv_with_pol_chain.h"

namespace mirheo {

/** \brief A View with additional polymeric chain vectors info
 */
struct PVviewWithPolChainVectorAndSmoothVelocity : public PVviewWithPolChainVector
{
    /** \brief Construct a PVviewWithPolChainVectorAndSmoothVelocity
        \param [in] pv The ParticleVector that the view represents
        \param [in] lpv The LocalParticleVector that the view represents

        \rst
        .. warning::
            The pv must hold polymeric chain vectors and their time derivative per particle channel.
        \endrst
        \rst
        .. warning::
            The pv must hold a smoothed velocity channel.
        \endrst
     */
    PVviewWithPolChainVectorAndSmoothVelocity(ParticleVector *pv, LocalParticleVector *lpv);

    real4 *smoothVel {nullptr}; ///< smoothed velocity
};

} // namespace mirheo
