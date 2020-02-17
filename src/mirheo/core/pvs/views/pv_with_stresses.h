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
    /** \brief Construct a \c PVviewWithStresses
        \param [in] pv The \c ParticleVector that the view represents
        \param [in] lpv The \c LocalParticleVector that the view represents

        \rst
        .. warning::
            The pv must hold a stress per particle channel.
        \endrst
     */
    PVviewWithStresses(ParticleVector *pv, LocalParticleVector *lpv) :
        BasicView(pv, lpv)
    {
        stresses = lpv->dataPerParticle.getData<Stress>(ChannelNames::stresses)->devPtr();            
    }

    Stress *stresses {nullptr}; ///< stresses per particle
};

} // namespace mirheo
