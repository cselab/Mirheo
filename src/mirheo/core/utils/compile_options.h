#pragma once

#include <string>

namespace mirheo
{

/// Store all compile options as a string representation
struct CompileOptions
{
#ifdef USE_NVTX
    static constexpr bool useNvtx = true;
#else
    static constexpr bool useNvtx = false;
#endif

#ifdef MIRHEO_DOUBLE_PRECISION
    static constexpr bool useDouble = true;
#else
    static constexpr bool useDouble = false;
#endif

#ifdef MEMBRANE_FORCES_DOUBLE
    static constexpr bool membraneDouble = true;
#else
    static constexpr bool membraneDouble = false;
#endif

#ifdef ROD_FORCES_DOUBLE
    static constexpr bool rodDouble = true;
#else
    static constexpr bool rodDouble = false;
#endif
};

/// a xmacro that lists all compile options
#define MIRHEO_COMPILE_OPT_TABLE(OP)            \
    OP(useNvtx)                                 \
    OP(useDouble)                               \
    OP(membraneDouble)                          \
    OP(rodDouble)

} //namespace mirheo
