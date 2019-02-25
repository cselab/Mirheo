#include "membrane_WLC_Kantor.h"
#include "membrane.impl.h"
#include "membrane/dihedral/kantor.h"
#include "membrane/triangle/wlc.h"

#include <core/utils/make_unique.h>

InteractionMembraneWLCKantor::InteractionMembraneWLCKantor(const YmrState *state, std::string name,
                                                           MembraneParameters parameters, KantorBendingParameters kantorParams,
                                                           bool stressFree, float growUntil) :
    InteractionMembrane(state, name)
{
    // TODO
    WLCParameters wlc;
    wlc.x0       = parameters.x0;
    wlc.ks       = parameters.ks;
    wlc.mpow     = parameters.mpow;
    wlc.kd       = parameters.kd;
    wlc.totArea0 = parameters.totArea0;

    if (stressFree)
        impl = std::make_unique<InteractionMembraneImpl<TriangleWLCForce<StressFreeState::Active>, DihedralKantor>>
            (state, name, parameters, wlc, kantorParams, growUntil);
    else
        impl = std::make_unique<InteractionMembraneImpl<TriangleWLCForce<StressFreeState::Inactive>, DihedralKantor>>
            (state, name, parameters, wlc, kantorParams, growUntil);

}

InteractionMembraneWLCKantor::~InteractionMembraneWLCKantor() = default;

