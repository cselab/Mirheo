#pragma once

#include <iosfwd>

void pretty_stacktrace(std::ostream& stream);
void register_signals();
