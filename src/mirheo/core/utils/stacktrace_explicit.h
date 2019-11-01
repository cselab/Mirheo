#pragma once

#include <iosfwd>

namespace mirheo
{

void pretty_stacktrace(std::ostream& stream);
void register_signals();

} // namespace mirheo
