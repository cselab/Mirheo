// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include "pv.h"

#include <mirheo/core/pvs/particle_vector.h>

namespace mirheo
{

/** \brief A View with additional stress info
    \tparam BasicView The pv view to extend with stresses
 */
template <typename BasicView>
struct PVviewWithStresses : public BasicView
{
#ifndef DOXYGEN_SHOULD_SKIP_THIS
    using PVType = typename BasicView::PVType;  ///< Particle Vector compatible type
    using LPVType = typename BasicView::LPVType;  ///< Local Particle Vector compatible type
#endif // DOXYGEN_SHOULD_SKIP_THIS

    /** \brief Construct a PVviewWithStresses
        \param [in] pv The ParticleVector that the view represents
        \param [in] lpv The LocalParticleVector that the view represents

        \rst
        .. warning::
            The pv must hold a stress per particle channel.
        \endrst
     */
    PVviewWithStresses(PVType *pv, LPVType *lpv) :
        BasicView(pv, lpv)
    {
        this->stresses = lpv->dataPerParticle.template getData<Stress>(channel_names::stresses)->devPtr();
    }

    Stress *stresses {nullptr}; ///< stresses per particle
};

} // namespace mirheo
