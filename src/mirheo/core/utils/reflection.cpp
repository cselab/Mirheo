#include "reflection.h"

#include <cstdarg>

namespace mirheo
{

std::string constructTypeName(const char *base, int N, ...)
{
    std::string out = base;
    out += '<';
    // https://en.cppreference.com/w/c/variadic/va_arg
    va_list args;
    va_start(args, N);
    for (int i = 0; i < N; ++i) {
        if (i > 0)
            out += ", ";
        out += va_arg(args, const char*);
    }
    va_end(args);
    out += '>';
    return out;
}

} // namespace mirheo
