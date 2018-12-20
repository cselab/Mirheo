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
    createVV(std::string name, float dt)
    {
        Forcing_None forcing;
        return std::make_shared<IntegratorVV<Forcing_None>> (name, dt, forcing);
    }

    static std::shared_ptr<IntegratorVV<Forcing_ConstDP>>
    createVV_constDP(std::string name, float dt, PyTypes::float3 extraForce)
    {
        Forcing_ConstDP forcing(make_float3(extraForce));
        return std::make_shared<IntegratorVV<Forcing_ConstDP>> (name, dt, forcing);
    }

    static std::shared_ptr<IntegratorVV<Forcing_PeriodicPoiseuille>>
    createVV_PeriodicPoiseuille(std::string name, float dt, float force, std::string direction)
    {
        Forcing_PeriodicPoiseuille::Direction dir;
        if      (direction == "x") dir = Forcing_PeriodicPoiseuille::Direction::x;
        else if (direction == "y") dir = Forcing_PeriodicPoiseuille::Direction::y;
        else if (direction == "z") dir = Forcing_PeriodicPoiseuille::Direction::z;
        else die("Direction can only be 'x' or 'y' or 'z'");
        
        Forcing_PeriodicPoiseuille forcing(force, dir);
        return std::make_shared<IntegratorVV<Forcing_PeriodicPoiseuille>> (name, dt, forcing);
    }

    static std::shared_ptr<IntegratorConstOmega>
    createConstOmega(std::string name, float dt, PyTypes::float3 center, PyTypes::float3 omega)
    {
        return std::make_shared<IntegratorConstOmega> (name, dt, make_float3(center), make_float3(omega));
    }

    static std::shared_ptr<IntegratorTranslate>
    createTranslate(std::string name, float dt, PyTypes::float3 velocity)
    {
        return std::make_shared<IntegratorTranslate> (name, dt, make_float3(velocity));
    }

    static std::shared_ptr<IntegratorOscillate>
    createOscillating(std::string name, float dt, PyTypes::float3 velocity, float period)
    {
        return std::make_shared<IntegratorOscillate> (name, dt, make_float3(velocity), period);
    }

    static std::shared_ptr<IntegratorVVRigid>
    createRigidVV(std::string name, float dt)
    {
        return std::make_shared<IntegratorVVRigid> (name, dt);
    }

    static std::shared_ptr<IntegratorSubStepMembrane>
    createSubStepMembrane(std::string name, float dt, int substeps, Interaction *fastForces)
    {
        return std::make_shared<IntegratorSubStepMembrane> (name, dt, substeps, fastForces);
    }    
};
