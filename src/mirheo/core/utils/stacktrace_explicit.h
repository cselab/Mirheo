// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <iosfwd>

namespace mirheo {
namespace stacktrace {

/** \brief Register all signal handlings that trigger a stacktrace.

    Should be called as soon as possible in the program, e.g. in logger initialization
*/
void registerSignals();

/** \brief Print the current stacktrace in a stream
    \param [out] stream The stream in which to dump the stack trace
    \param [in] traceCntMax The maximum number of traces to print
 */
void getStacktrace(std::ostream& stream, size_t traceCntMax = 100);

} // namespace stacktrace
} // namespace mirheo
