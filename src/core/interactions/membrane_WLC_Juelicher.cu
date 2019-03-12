#include "membrane_WLC_Juelicher.h"
#include "membrane.impl.h"
#include "membrane/common.h"
#include "membrane/dihedral/juelicher.h"
#include "membrane/triangle/wlc.h"

#include <core/utils/make_unique.h>

InteractionMembraneWLCJuelicher::InteractionMembraneWLCJuelicher(const YmrState *state, std::string name,
                                                                 CommonMembraneParameters parameters,
                                                                 WLCParameters wlcParams,
                                                                 JuelicherBendingParameters juelicherParams,
                                                                 bool stressFree, float growUntil) :
    InteractionMembraneJuelicher(state, name)
{
    if (stressFree)
        impl = std::make_unique<InteractionMembraneImpl<TriangleWLCForce<StressFreeState::Active>, DihedralJuelicher>>
            (state, name, parameters, wlcParams, juelicherParams, growUntil);
    else
        impl = std::make_unique<InteractionMembraneImpl<TriangleWLCForce<StressFreeState::Inactive>, DihedralJuelicher>>
            (state, name, parameters, wlcParams, juelicherParams, growUntil);
}

InteractionMembraneWLCJuelicher::~InteractionMembraneWLCJuelicher() = default;
