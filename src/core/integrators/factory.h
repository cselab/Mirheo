#pragma once

#include <memory>

#include <core/utils/make_unique.h>
#include <core/utils/pytypes.h>

#include "const_omega.h"
#include "forcing_terms/const_dp.h"
#include "forcing_terms/none.h"
#include "forcing_terms/periodic_poiseuille.h"
#include "oscillate.h"
#include "rigid_vv.h"
#include "sub_step_membrane.h"
#include "translate.h"
#include "vv.h"

namespace IntegratorFactory
{
    static std::shared_ptr<IntegratorVV<Forcing_None>>
    createVV(const YmrState *state, std::string name)
    {
        Forcing_None forcing;
        return std::make_shared<IntegratorVV<Forcing_None>> (state, name, forcing);
    }

    static std::shared_ptr<IntegratorVV<Forcing_ConstDP>>
    createVV_constDP(const YmrState *state, std::string name, PyTypes::float3 extraForce)
    {
        Forcing_ConstDP forcing(make_float3(extraForce));
        return std::make_shared<IntegratorVV<Forcing_ConstDP>> (state, name, forcing);
    }

    static std::shared_ptr<IntegratorVV<Forcing_PeriodicPoiseuille>>
    createVV_PeriodicPoiseuille(const YmrState *state, std::string name, float force, std::string direction)
    {
        Forcing_PeriodicPoiseuille::Direction dir;
        if      (direction == "x") dir = Forcing_PeriodicPoiseuille::Direction::x;
        else if (direction == "y") dir = Forcing_PeriodicPoiseuille::Direction::y;
        else if (direction == "z") dir = Forcing_PeriodicPoiseuille::Direction::z;
        else die("Direction can only be 'x' or 'y' or 'z'");
        
        Forcing_PeriodicPoiseuille forcing(force, dir);
        return std::make_shared<IntegratorVV<Forcing_PeriodicPoiseuille>> (state, name, forcing);
    }

    static std::shared_ptr<IntegratorConstOmega>
    createConstOmega(const YmrState *state, std::string name, PyTypes::float3 center, PyTypes::float3 omega)
    {
        return std::make_shared<IntegratorConstOmega> (state, name, make_float3(center), make_float3(omega));
    }

    static std::shared_ptr<IntegratorTranslate>
    createTranslate(const YmrState *state, std::string name, PyTypes::float3 velocity)
    {
        return std::make_shared<IntegratorTranslate> (state, name, make_float3(velocity));
    }

    static std::shared_ptr<IntegratorOscillate>
    createOscillating(const YmrState *state, std::string name, PyTypes::float3 velocity, float period)
    {
        return std::make_shared<IntegratorOscillate> (state, name, make_float3(velocity), period);
    }

    static std::shared_ptr<IntegratorVVRigid>
    createRigidVV(const YmrState *state, std::string name)
    {
        return std::make_shared<IntegratorVVRigid> (state, name);
    }

    static std::shared_ptr<IntegratorSubStepMembrane>
    createSubStepMembrane(const YmrState *state, std::string name, int substeps, Interaction *fastForces)
    {
        return std::make_shared<IntegratorSubStepMembrane> (state, name, substeps, fastForces);
    }    
};
