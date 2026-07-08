// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include "pv.h"

namespace mirheo {

/** \brief A View with additional polymeric chain vectors info
 */
struct PVviewWithPolChainVector : public PVview
{
    /** \brief Construct a PVviewWithPolChainVector
        \param [in] pv The ParticleVector that the view represents
        \param [in] lpv The LocalParticleVector that the view represents

        \rst
        .. warning::
            The pv must hold polymeric chain vectors and their time derivative per particle channel.
        \endrst
     */
    PVviewWithPolChainVector(ParticleVector *pv, LocalParticleVector *lpv);

    real3 *Q {nullptr}; ///< polymeric chain vector end-to-end
    real3 *dQdt {nullptr}; ///< time derivative of Q
};

} // namespace mirheo
