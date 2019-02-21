#include "membrane_WLC_Kantor.h"
#include "membrane.impl.h"
#include "membrane/dihedral/kantor.h"

#include <core/utils/make_unique.h>

MembraneWLCKantor::MembraneWLCKantor(const YmrState *state, std::string name,
                                     MembraneParameters parameters, KantorBendingParameters kantorParams,
                                     bool stressFree, float growUntil) :
    InteractionMembraneNew(state, name)
{
    impl = std::make_unique<InteractionMembrane<DihedralKantor>>(state, name, parameters, kantorParams, stressFree, growUntil);
}

MembraneWLCKantor::~MembraneWLCKantor() = default;

