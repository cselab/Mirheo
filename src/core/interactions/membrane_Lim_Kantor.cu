#include "membrane_Lim_Kantor.h"
#include "membrane.impl.h"
#include "membrane/dihedral/kantor.h"
#include "membrane/triangle/lim.h"

#include <core/utils/make_unique.h>

InteractionMembraneLimKantor::InteractionMembraneLimKantor(const YmrState *state, std::string name,
                                                           MembraneParameters parameters,
                                                           LimParameters limParams,
                                                           KantorBendingParameters kantorParams,
                                                           bool stressFree, float growUntil) :
    InteractionMembrane(state, name)
{
    if (stressFree)
        impl = std::make_unique<InteractionMembraneImpl<TriangleLimForce<StressFreeState::Active>, DihedralKantor>>
            (state, name, parameters, limParams, kantorParams, growUntil);
    else
        impl = std::make_unique<InteractionMembraneImpl<TriangleLimForce<StressFreeState::Inactive>, DihedralKantor>>
            (state, name, parameters, limParams, kantorParams, growUntil);

}

InteractionMembraneLimKantor::~InteractionMembraneLimKantor() = default;

