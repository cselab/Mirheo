// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "compile_options.h"

namespace mirheo
{

constexpr bool CompileOptions::useDouble;

const CompileOptions compile_options{
#ifdef MIRHEO_MEMBRANE_FORCES_DOUBLE
    true,
#else
    false,
#endif
#ifdef MIRHEO_ROD_FORCES_DOUBLE
    true,
#else
    false,
#endif
#ifdef MIRHEO_USE_NVTX
    true,
#else
    false,
#endif
};

} // namespace mirheo
