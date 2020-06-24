// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "factory.h"
#include <mirheo/core/pvs/particle_vector.h>
#include <mirheo/core/pvs/object_vector.h>
#include <mirheo/core/pvs/membrane_vector.h>
#include <mirheo/core/utils/config.h>

namespace mirheo
{

namespace particle_vector_factory
{

std::shared_ptr<ParticleVector> loadParticleVector(
        const MirState *state, Loader& loader, const ConfigObject& config)
{
    const std::string& type = config["__type"];
    if (type == "ParticleVector")
        return std::make_shared<ParticleVector>(state, loader, config);
    if (type == "MembraneVector")
        return std::make_shared<MembraneVector>(state, loader, config);
    die("Unrecognized or unimplemented particle vector type \"%s\".", type.c_str());
}

} // namespace particle_vector_factory
} // namespace mirheo
