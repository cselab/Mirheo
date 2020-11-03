// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "stacktrace_explicit.h"

#define BACKWARD_HAS_BFD 1
#include <extern/backward-cpp/backward.hpp>

namespace mirheo {
namespace stacktrace {

void registerSignals()
{
    // This will load most default signals that trigger stack trace
    // (destroying the object will not unload anything)
    backward::SignalHandling sh;
}

void getStacktrace(std::ostream& stream, size_t traceCntMax)
{
    using namespace backward;

    StackTrace st;
    st.load_here(traceCntMax);
    Printer p;
    p.object = true;
    p.color_mode = ColorMode::automatic;
    p.address = true;
    p.print(st, stream);
}

} // namespace stacktrace
} // namespace mirheo
