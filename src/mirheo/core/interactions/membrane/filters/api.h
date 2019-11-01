#pragma once

#include "keep_all.h"
#include "keep_by_type_id.h"

#include <extern/variant/include/mpark/variant.hpp>

namespace mirheo
{

using VarMembraneFilter = mpark::variant<FilterKeepAll,
                                         FilterKeepByTypeId>;

} // namespace mirheo
