#pragma once

#include "type_list.h"

#include <string>

namespace mirheo
{

std::string printToStr(int val);
std::string printToStr(int64_t val);

std::string printToStr(float val);
std::string printToStr(float2 val);
std::string printToStr(float3 val);
std::string printToStr(float4 val);

std::string printToStr(double val);
std::string printToStr(double2 val);
std::string printToStr(double3 val);
std::string printToStr(double4 val);

std::string printToStr(Stress val);
std::string printToStr(RigidMotion val);
std::string printToStr(COMandExtent val);
std::string printToStr(Force val);

} // namespace mirheo
