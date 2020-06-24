// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <string>

namespace mirheo
{

/// holds information of the current Mirheo version
struct Version
{
    static const std::string mir_version; ///< The current version of Mirheo
    static const std::string git_SHA1;    ///< The current git SHA1
};

} // namespace mirheo
