#include "compile_options.h"

namespace mirheo {

const std::string CompileOptions::useNvtx =
#ifdef USE_NVTX
        "ON"
#else
        "OFF"
#endif
    ;

const std::string CompileOptions::useDouble =
#ifdef MIRHEO_DOUBLE_PRECISION
        "ON"
#else
        "OFF"
#endif
        ;

const std::string CompileOptions::membraneDouble =
#ifdef MEMBRANE_FORCES_DOUBLE
        "ON"
#else
        "OFF"
#endif
        ;

const std::string CompileOptions::rodDouble =
#ifdef ROD_FORCES_DOUBLE
        "ON"
#else
        "OFF"
#endif
        ;

} //namespace mirheo
