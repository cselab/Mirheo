// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <string>

namespace mirheo
{

/** \brief Create a std::string from a printf-style formatting
    \param [in] fmt The c-style string that represents the format of the string, followed by extra arguments (see printf docs)
    \return the std::string corresponding to the given format evaluated with the passed extra arguments
 */
std::string strprintf [[gnu::format(printf, 1, 2)]] (const char *fmt, ...);

} // namespace mirheo
