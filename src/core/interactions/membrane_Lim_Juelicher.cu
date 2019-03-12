#include "membrane_Lim_Juelicher.h"
#include "membrane.impl.h"
#include "membrane/common.h"
#include "membrane/dihedral/juelicher.h"
#include "membrane/triangle/lim.h"

#include <core/utils/make_unique.h>

InteractionMembraneLimJuelicher::InteractionMembraneLimJuelicher(const YmrState *state, std::string name,
                                                                 CommonMembraneParameters parameters,
                                                                 LimParameters limParams,
                                                                 JuelicherBendingParameters juelicherParams,
                                                                 bool stressFree, float growUntil) :
    InteractionMembraneJuelicher(state, name)
{
    if (stressFree)
        impl = std::make_unique<InteractionMembraneImpl<TriangleLimForce<StressFreeState::Active>, DihedralJuelicher>>
            (state, name, parameters, limParams, juelicherParams, growUntil);
    else
        impl = std::make_unique<InteractionMembraneImpl<TriangleLimForce<StressFreeState::Inactive>, DihedralJuelicher>>
            (state, name, parameters, limParams, juelicherParams, growUntil);
}

InteractionMembraneLimJuelicher::~InteractionMembraneLimJuelicher() = default;
