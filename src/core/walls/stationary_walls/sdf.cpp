#include "sdf.h"

StationaryWall_SDF::StationaryWall_SDF(std::string sdfFileName, float3 sdfH) :
    Field(sdfFileName, sdfH)
{}

StationaryWall_SDF::StationaryWall_SDF(StationaryWall_SDF&&) = default;
