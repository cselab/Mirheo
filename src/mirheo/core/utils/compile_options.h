#pragma once

namespace mirheo
{

/// Stores all relevant compilation flags.
struct CompileOptions
{
    // Public flags (propagate to all user codes).
#ifdef MIRHEO_DOUBLE_PRECISION
    static constexpr bool useDouble = true;
#else
    static constexpr bool useDouble = false;
#endif

    // Core-private flags. Cannot be constexpr. If changing the field or their
    // order, don't forget to update the .cpp file!
    bool membraneDouble;
    bool rodDouble;
    bool useNvtx;
};

extern const CompileOptions compile_options;

/// a xmacro that lists all compile options
#define MIRHEO_COMPILE_OPT_TABLE(OP)            \
    OP(useNvtx)                                 \
    OP(useDouble)                               \
    OP(membraneDouble)                          \
    OP(rodDouble)

} // namespace mirheo
