#include "membrane_WLC_Kantor.h"
#include "membrane.impl.h"
#include "membrane/dihedral/kantor.h"
#include "membrane/triangle/wlc.h"

#include <core/utils/make_unique.h>

InteractionMembraneWLCKantor::InteractionMembraneWLCKantor(const YmrState *state, std::string name,
                                                           MembraneParameters parameters,
                                                           WLCParameters wlcParams,
                                                           KantorBendingParameters kantorParams,
                                                           bool stressFree, float growUntil) :
    InteractionMembrane(state, name)
{
    if (stressFree)
        impl = std::make_unique<InteractionMembraneImpl<TriangleWLCForce<StressFreeState::Active>, DihedralKantor>>
            (state, name, parameters, wlcParams, kantorParams, growUntil);
    else
        impl = std::make_unique<InteractionMembraneImpl<TriangleWLCForce<StressFreeState::Inactive>, DihedralKantor>>
            (state, name, parameters, wlcParams, kantorParams, growUntil);

}


InteractionMembraneWLCKantor::~InteractionMembraneWLCKantor() = default;

