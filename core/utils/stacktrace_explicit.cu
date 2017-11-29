#include "stacktrace_explicit.h"
#include "stacktrace.h"

void pretty_stacktrace(std::ostream& stream)
{
	using namespace backward;

	StackTrace st;
	st.load_here(32);
	Printer p;
	p.object = true;
	p.color_mode = ColorMode::automatic;
	p.address = true;
	p.print(st, stream);
}
