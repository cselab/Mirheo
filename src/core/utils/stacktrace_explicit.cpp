#include "stacktrace_explicit.h"

#define BACKWARD_HAS_BFD 1
#include <extern/backward-cpp/backward.hpp>

void pretty_stacktrace(std::ostream& stream)
{
    using namespace backward;

    StackTrace st;
    st.load_here(40);
    Printer p;
    p.object = true;
    p.color_mode = ColorMode::automatic;
    p.address = true;
    p.print(st, stream);
}
