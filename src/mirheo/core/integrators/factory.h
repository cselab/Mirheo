#pragma once

#include "const_omega.h"
#include "forcing_terms/const_dp.h"
#include "forcing_terms/none.h"
#include "forcing_terms/periodic_poiseuille.h"
#include "oscillate.h"
#include "rigid_vv.h"
#include "sub_step.h"
#include "translate.h"
#include "vv.h"

#include <vector_types.h>

#include <memory>

namespace IntegratorFactory
{
inline std::shared_ptr<IntegratorVV<Forcing_None>>
createVV(const MirState *state, const std::string& name)
{
    Forcing_None forcing;
    return std::make_shared<IntegratorVV<Forcing_None>> (state, name, forcing);
}

inline std::shared_ptr<IntegratorVV<Forcing_ConstDP>>
createVV_constDP(const MirState *state, const std::string& name, real3 extraForce)
{
    Forcing_ConstDP forcing(extraForce);
    return std::make_shared<IntegratorVV<Forcing_ConstDP>> (state, name, forcing);
}

inline std::shared_ptr<IntegratorVV<Forcing_PeriodicPoiseuille>>
createVV_PeriodicPoiseuille(const MirState *state, const std::string& name, real force, std::string direction)
{
    Forcing_PeriodicPoiseuille::Direction dir;
    if      (direction == "x") dir = Forcing_PeriodicPoiseuille::Direction::x;
    else if (direction == "y") dir = Forcing_PeriodicPoiseuille::Direction::y;
    else if (direction == "z") dir = Forcing_PeriodicPoiseuille::Direction::z;
    else die("Direction can only be 'x' or 'y' or 'z'");
        
    Forcing_PeriodicPoiseuille forcing(force, dir);
    return std::make_shared<IntegratorVV<Forcing_PeriodicPoiseuille>> (state, name, forcing);
}

inline std::shared_ptr<IntegratorConstOmega>
createConstOmega(const MirState *state, const std::string& name, real3 center, real3 omega)
{
    return std::make_shared<IntegratorConstOmega> (state, name, center, omega);
}

inline std::shared_ptr<IntegratorTranslate>
createTranslate(const MirState *state, const std::string& name, real3 velocity)
{
    return std::make_shared<IntegratorTranslate> (state, name, velocity);
}

inline std::shared_ptr<IntegratorOscillate>
createOscillating(const MirState *state, const std::string& name, real3 velocity, real period)
{
    return std::make_shared<IntegratorOscillate> (state, name, velocity, period);
}

inline std::shared_ptr<IntegratorVVRigid>
createRigidVV(const MirState *state, const std::string& name)
{
    return std::make_shared<IntegratorVVRigid> (state, name);
}

inline std::shared_ptr<IntegratorSubStep>
createSubStep(const MirState *state, const std::string& name, int substeps,
              const std::vector<Interaction*>& fastForces)
{
    return std::make_shared<IntegratorSubStep> (state, name, substeps, fastForces);
}    
} // namespace IntegratorFactory
