// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include "keep_all.h"
#include "keep_by_type_id.h"

#include <variant>

namespace mirheo
{

/// variant that contains all possible filters
using VarMembraneFilter = std::variant<FilterKeepAll,
                                       FilterKeepByTypeId>;

} // namespace mirheo
