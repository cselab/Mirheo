#pragma once

#include <string>

namespace mirheo
{

std::string strprintf [[gnu::format(printf, 1, 2)]] (const char *fmt, ...);

} // namespace mirheo
