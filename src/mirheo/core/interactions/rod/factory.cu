#include "factory.h"
#include "rod.h"

namespace mirheo
{

template <int Nstates>
std::shared_ptr<BaseRodInteraction>
instantiateImpl(const MirState *state, const std::string& name, RodParameters parameters,
                VarSpinParams varSpinParams, bool saveEnergies)
{
    std::shared_ptr<BaseRodInteraction> impl;

    mpark::visit([&](auto spinParams)
    {
        using SpinParamsType = decltype(spinParams);

        impl = std::make_shared<RodInteraction<Nstates, SpinParamsType>>
            (state, name, parameters, spinParams, saveEnergies);
    }, varSpinParams);

    return impl;
}


std::shared_ptr<BaseRodInteraction>
createInteractionRod(const MirState *state, const std::string& name,
                     RodParameters params, VarSpinParams spinParams, bool saveEnergies)
{
    std::shared_ptr<BaseRodInteraction> impl;
    const int nstates = params.kappaEq.size();

    if (mpark::holds_alternative<StatesParametersNone>(spinParams))
    {
        if (nstates != 1)
            die("only one state supported for state_update = 'none' (while creating %s)", name.c_str());

        impl = std::make_shared<RodInteraction<1, StatesParametersNone>>
            (state, name, params, mpark::get<StatesParametersNone>(spinParams), saveEnergies);
    }
    else
    {
        if (nstates <= 1)
            warn("using only one state for state_update != 'none' (while creating %s)", name.c_str());

#define CHECK_IMPLEMENT(Nstates) do {                                   \
            if (nstates == Nstates) {                                   \
                impl = instantiateImpl<Nstates>                         \
                    (state, name, params, spinParams, saveEnergies); \
                debug("Create interaction rod with %d states", Nstates); \
                return impl;                                                 \
            } } while(0)

        CHECK_IMPLEMENT(2); // 2 polymorphic states
        CHECK_IMPLEMENT(11); // bbacterial flagella have up to 11 states

        die("'%s' : number of states %d is not implemented", name.c_str(), nstates);
    }

    return impl;
}

} // namespace mirheo
