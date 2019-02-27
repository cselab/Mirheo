#pragma once

#include <string>

namespace ChannelNames
{

// per particle fields
static const std::string forces      = "forces";
static const std::string stresses    = "stresses";
static const std::string densities   = "densities";
static const std::string oldParts    = "old_particles";

// per object fields
static const std::string motions     = "motions";
static const std::string oldMotions  = "old_motions";
static const std::string comExtents  = "com_extents";
static const std::string areaVolumes = "area_volumes";
    
// per object, specific to Juelicher bending + ADE    
static const std::string areas          = "areas";
static const std::string meanCurvatures = "meanCurvatures";
static const std::string lenThetaTot    = "lenThetaTot";
    
    
// per entity (particles or objects
static const std::string globalIds   = "ids";

} // namespace ChannelNames
