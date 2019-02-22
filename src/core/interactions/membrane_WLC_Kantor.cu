#include "membrane_WLC_Kantor.h"
#include "membrane.impl.h"
#include "membrane/dihedral/kantor.h"

#include <core/utils/make_unique.h>

InteractionMembraneWLCKantor::InteractionMembraneWLCKantor(const YmrState *state, std::string name,
                                                           MembraneParameters parameters, KantorBendingParameters kantorParams,
                                                           bool stressFree, float growUntil) :
    InteractionMembrane(state, name)
{
    impl = std::make_unique<InteractionMembraneImpl<DihedralKantor>>(state, name, parameters, kantorParams, stressFree, growUntil);
}

InteractionMembraneWLCKantor::~InteractionMembraneWLCKantor() = default;

