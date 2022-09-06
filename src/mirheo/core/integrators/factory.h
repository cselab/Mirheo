// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include "const_omega.h"
#include "forcing_terms/const_dp.h"
#include "forcing_terms/none.h"
#include "forcing_terms/periodic_poiseuille.h"
#include "minimize.h"
#include "oscillate.h"
#include "rbc_shardlow.h"
#include "rigid_vv.h"
#include "shear.h"
#include "shear_pol_chain.h"
#include "sub_step.h"
#include "translate.h"
#include "vv.h"
#include "vv_pol_chain.h"

#include <vector_types.h>

#include <memory>

namespace mirheo {
namespace integrator_factory {

inline std::shared_ptr<IntegratorMinimize>
createMinimize(const MirState *state, const std::string& name, real maxDisplacement)
{
    return std::make_shared<IntegratorMinimize> (state, name, maxDisplacement);
}

inline std::shared_ptr<IntegratorVV<ForcingTermNone>>
createVV(const MirState *state, const std::string& name)
{
    ForcingTermNone forcing;
    return std::make_shared<IntegratorVV<ForcingTermNone>> (state, name, forcing);
}

inline std::shared_ptr<IntegratorVV<ForcingTermConstDP>>
createVV_constDP(const MirState *state, const std::string& name, real3 extraForce)
{
    ForcingTermConstDP forcing(extraForce);
    return std::make_shared<IntegratorVV<ForcingTermConstDP>> (state, name, forcing);
}

inline std::shared_ptr<IntegratorVV<ForcingTermPeriodicPoiseuille>>
createVV_PeriodicPoiseuille(const MirState *state, const std::string& name, real force, std::string direction)
{
    ForcingTermPeriodicPoiseuille::Direction dir;
    if      (direction == "x") dir = ForcingTermPeriodicPoiseuille::Direction::x;
    else if (direction == "y") dir = ForcingTermPeriodicPoiseuille::Direction::y;
    else if (direction == "z") dir = ForcingTermPeriodicPoiseuille::Direction::z;
    else die("Direction can only be 'x' or 'y' or 'z'");

    ForcingTermPeriodicPoiseuille forcing(force, dir);
    return std::make_shared<IntegratorVV<ForcingTermPeriodicPoiseuille>> (state, name, forcing);
}

inline std::shared_ptr<IntegratorVVPolChain>
createVVPolChain(const MirState *state, const std::string& name)
{
    return std::make_shared<IntegratorVVPolChain> (state, name);
}


inline std::shared_ptr<IntegratorConstOmega>
createConstOmega(const MirState *state, const std::string& name, real3 center, real3 omega)
{
    return std::make_shared<IntegratorConstOmega> (state, name, center, omega);
}

inline std::shared_ptr<IntegratorShear>
createShear(const MirState *state, const std::string& name, std::array<real,9> shear, real3 origin)
{
    return std::make_shared<IntegratorShear> (state, name, shear, origin);
}

inline std::shared_ptr<IntegratorShearPolChain>
createShearPolChain(const MirState *state, const std::string& name, std::array<real,9> shear, real3 origin)
{
    return std::make_shared<IntegratorShearPolChain> (state, name, shear, origin);
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

inline std::shared_ptr<IntegratorSubStepShardlowSweep>
createSubstepShardlowSweep(const MirState *state, const std::string& name, int substeps,
                           BaseMembraneInteraction* fastForces, real gammaC, real kBT, int nsweeps)
{
    return std::make_shared<IntegratorSubStepShardlowSweep> (state, name, substeps, fastForces, gammaC, kBT, nsweeps);
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

} // namespace integrator_factory

} // namespace mirheo
