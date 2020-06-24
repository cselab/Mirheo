// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <memory>
#include <string>

namespace mirheo
{

class ConfigObject;
class MirState;
class Mesh;
class Loader;

/** \brief Mesh factory. Instantiate the correct mesh type depending on the snapshot parameters.
    \param [in] state The global state of the system.
    \param [in] loader The \c Loader object. Provides load context and unserialization functions.
    \param [in] config The parameters of the integrator.
 */
std::shared_ptr<Mesh> loadMesh(
        const MirState *state, Loader& loader, const ConfigObject& config);

} // namespace mirheo
