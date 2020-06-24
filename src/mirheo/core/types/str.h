#pragma once

#include "type_list.h"

#include <string>

namespace mirheo
{

/** \brief transform the input in readable string
    \param val The value to transform to string
    \return the string representation of \p val (e.g. "42" if val=42)
 */
std::string printToStr(int val);
std::string printToStr(int64_t val); ///< \overload

std::string printToStr(float val);  ///< \overload
std::string printToStr(float2 val); ///< \overload
std::string printToStr(float3 val); ///< \overload
std::string printToStr(float4 val); ///< \overload

std::string printToStr(double val);  ///< \overload
std::string printToStr(double2 val); ///< \overload
std::string printToStr(double3 val); ///< \overload
std::string printToStr(double4 val); ///< \overload

std::string printToStr(Stress val); ///< \overload
std::string printToStr(RigidMotion val); ///< \overload
std::string printToStr(COMandExtent val); ///< \overload
std::string printToStr(Force val); ///< \overload

} // namespace mirheo
