#include "rod.h"
#include "rod.impl.h"

template <int Nstates>
auto instantiateImpl(const MirState *state, std::string name, RodParameters parameters, VarSpinParams varSpinParams, bool saveEnergies)
{
    std::unique_ptr<Interaction> impl;

    mpark::visit([&](auto spinParams)
    {
        using SpinParamsType = decltype(spinParams);
        
        impl = std::make_unique<InteractionRodImpl<Nstates, SpinParamsType>>
            (state, name, parameters, spinParams, saveEnergies);
    }, varSpinParams);

    return impl;
}

InteractionRod::InteractionRod(const MirState *state, std::string name, RodParameters parameters,
                               VarSpinParams varSpinParams, bool saveEnergies) :
    Interaction(state, name, /*rc*/ 1.f)
{
    int nstates = parameters.kappaEq.size();

    if (mpark::holds_alternative<StatesParametersNone>(varSpinParams))
    {
        if (nstates != 1)
            die("only one state supported for state_update = 'none' (while creating %s)", name.c_str());

        impl = std::make_unique<InteractionRodImpl<1, StatesParametersNone>>
            (state, name, parameters, mpark::get<StatesParametersNone>(varSpinParams), saveEnergies);
    }
    else
    {
        if (nstates <= 1)
            warn("using only one state for state_update != 'none' (while creating %s)", name.c_str());
        
#define CHECK_IMPLEMENT(Nstates) do {                                   \
            if (nstates == Nstates) {                                   \
                impl = instantiateImpl<Nstates>                         \
                    (state, name, parameters, varSpinParams, saveEnergies); \
                debug("Create interaction rod with %d states", Nstates); \
                return;                                                 \
            } } while(0)
        
        CHECK_IMPLEMENT(2); // 2 polymorphic states
        CHECK_IMPLEMENT(11); // bbacterial flagella have up to 11 states

        die("'%s' : number of states %d is not implemented", name.c_str(), nstates);
    }
}

InteractionRod::~InteractionRod() = default;

void InteractionRod::setPrerequisites(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2)
{
    if (pv1 != pv2)
        die("Internal rod forces can't be computed between two different particle vectors");

    auto rv = dynamic_cast<RodVector*>(pv1);
    if (rv == nullptr)
        die("Internal rod forces can only be computed with a RodVector");

    impl->setPrerequisites(pv1, pv2, cl1, cl2);
}

void InteractionRod::local(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2, cudaStream_t stream)
{
    if (impl.get() == nullptr)
        die("%s needs a concrete implementation, none was provided", name.c_str());

    impl->local(pv1, pv2, cl1, cl2, stream);
}

void InteractionRod::halo(ParticleVector *pv1,
                          __UNUSED ParticleVector *pv2,
                          __UNUSED CellList *cl1,
                          __UNUSED CellList *cl2,
                          __UNUSED cudaStream_t stream)
{
    debug("Not computing internal rod forces between local and halo rods of '%s'", pv1->name.c_str());
}

bool InteractionRod::isSelfObjectInteraction() const
{
    return true;
}
