// Copyright 2021 ETH Zurich. All Rights Reserved.
#pragma once

#include "interface.h"

#include <memory>

namespace mirheo
{

namespace object_belonging_checker_factory
{

/** \brief Construct an ObjectBelongingChecker from its snapshot.
    \param [in] state The global state of the system.
    \param [in] loader The \c Loader object. Provides load context and unserialization functions.
    \param [in] config The parameters of the particle vector.
 */
std::shared_ptr<ObjectBelongingChecker> loadChecker(
        const MirState *state, Loader& loader, const ConfigObject& config);

} // namespace object_belonging_checker_factory
} // namespace mirheo
