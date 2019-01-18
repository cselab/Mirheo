#pragma once

#include <core/field/interface.h>

class StationaryWall_SDF : public Field
{
public:
    StationaryWall_SDF(std::string sdfFileName, float3 sdfH);
    StationaryWall_SDF(StationaryWall_SDF&&);
};
