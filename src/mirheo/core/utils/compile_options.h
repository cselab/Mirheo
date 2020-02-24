#pragma once

#include <string>

namespace mirheo
{

/// Store all compile options as a string representation
struct CompileOptions
{
    static const std::string useNvtx;
    static const std::string useDouble;
    static const std::string membraneDouble;
    static const std::string rodDouble;
};

/// a xmacro that lists all compile options
#define MIRHEO_COMPILE_OPT_TABLE(OP)            \
    OP(useNvtx)                                 \
    OP(useDouble)                               \
    OP(membraneDouble)                          \
    OP(rodDouble)

} //namespace mirheo
