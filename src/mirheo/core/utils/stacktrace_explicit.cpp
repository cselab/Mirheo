// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "stacktrace_explicit.h"

#include <iostream>

#ifdef MIRHEO_ENABLE_STACKTRACE
#define BACKWARD_HAS_BFD 1
#include <extern/backward-cpp/backward.hpp>
#endif // MIRHEO_ENABLE_STACKTRACE

namespace mirheo {
namespace stacktrace {

void registerSignals()
{
#ifdef MIRHEO_ENABLE_STACKTRACE
    // This will load most default signals that trigger stack trace
    // (destroying the object will not unload anything)
    backward::SignalHandling sh;
#endif // MIRHEO_ENABLE_STACKTRACE
}

void getStacktrace(std::ostream& stream, size_t traceCntMax)
{
#ifdef MIRHEO_ENABLE_STACKTRACE
    using namespace backward;

    StackTrace st;
    st.load_here(traceCntMax);
    Printer p;
    p.object = true;
    p.color_mode = ColorMode::automatic;
    p.address = true;
    p.print(st, stream);
#else
    stream << "Stacktrace is disabled. Was called with traceCntMax=" << traceCntMax << ".\n";
#endif // MIRHEO_ENABLE_STACKTRACE
}

} // namespace stacktrace
} // namespace mirheo
