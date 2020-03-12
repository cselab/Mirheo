#pragma once

#include <iosfwd>

namespace mirheo {
namespace stacktrace {

void registerSignals();
void getStacktrace(std::ostream& stream);

} // namespace stacktrace
} // namespace mirheo
