#pragma once

#include <memory>

#include <core/utils/make_unique.h>
#include <core/utils/pytypes.h>

#include "const_omega.h"
#include "forcing_terms/const_dp.h"
#include "forcing_terms/none."
#include "forcing_terms/periodic_poiseuille.h"
#include "oscillate.h"
#include "rigid_vv.h"
#include "sub_step_membrane.h"
#include "translate.h"
#include "vv.h"

namespace IntegratorFactory
{
    static IntegratorVV<Forcing_None>* createVV(std::string name, float dt)
    {
        Forcing_None forcing;
        return  new IntegratorVV<Forcing_None> (name, dt, forcing);
    }

    static IntegratorVV<Forcing_ConstDP>* createVV_constDP(std::string name, float dt, PyTypes::float3 extraForce)
    {
        Forcing_ConstDP forcing(make_float3(extraForce));
        return  new IntegratorVV<Forcing_ConstDP>(name, dt, forcing);
    }

    static IntegratorVV<Forcing_PeriodicPoiseuille>*
        createVV_PeriodicPoiseuille(std::string name, float dt, float force, std::string direction)
    {
        Forcing_PeriodicPoiseuille::Direction dir;
        if      (direction == "x") dir = Forcing_PeriodicPoiseuille::Direction::x;
        else if (direction == "y") dir = Forcing_PeriodicPoiseuille::Direction::y;
        else if (direction == "z") dir = Forcing_PeriodicPoiseuille::Direction::z;
        else die("Direction can only be 'x' or 'y' or 'z'");
        
        Forcing_PeriodicPoiseuille forcing(force, dir);
        return  new IntegratorVV<Forcing_PeriodicPoiseuille>(name, dt, forcing);
    }

    static IntegratorConstOmega* createConstOmega(std::string name, float dt, PyTypes::float3 center, PyTypes::float3 omega)
    {
        return  new IntegratorConstOmega(name, dt, make_float3(center), make_float3(omega));
    }

    static IntegratorTranslate* createTranslate(std::string name, float dt, PyTypes::float3 velocity)
    {
        return  new IntegratorTranslate(name, dt, make_float3(velocity));
    }

    static IntegratorOscillate* createOscillating(std::string name, float dt, PyTypes::float3 velocity, float period)
    {
        return  new IntegratorOscillate(name, dt, make_float3(velocity), period);
    }

    static IntegratorVVRigid* createRigidVV(std::string name, float dt)
    {
        return  new IntegratorVVRigid(name, dt);
    }

    static IntegratorSubStepMembrane* createSubStepMembrane(std::string name, float dt, int substeps, Interaction *fastForces)
    {
        return new IntegratorSubStepMembrane(name, dt, substeps, fastForces);
    }    
};
