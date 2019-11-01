#pragma once

#include <string>

namespace mirheo
{

/**
 * Channel names used in several places of the program
 * names starting with "__" are internal channel names 
 */
namespace ChannelNames
{

// per particle fields
static const std::string positions     = "__positions";
static const std::string velocities    = "__velocities";
static const std::string forces        = "__forces";
static const std::string stresses      = "stresses";
static const std::string densities     = "densities";
static const std::string oldPositions  = "old_positions";

// per object fields
static const std::string motions     = "motions";
static const std::string oldMotions  = "old_motions";
static const std::string comExtents  = "com_extents";
static const std::string areaVolumes = "area_volumes";

static const std::string membraneTypeId = "membrane_type_id";
    
// per object, specific to Juelicher bending + ADE    
static const std::string areas          = "areas";
static const std::string meanCurvatures = "meanCurvatures";
static const std::string lenThetaTot    = "lenThetaTot";

// per bisegment data
static const std::string polyStates    = "states";
static const std::string energies      = "energies";
static const std::string rodKappa      = "biseg_kappa";
static const std::string rodTau_l      = "biseg_tau_l";

    
// per entity (particles or objects)
static const std::string globalIds   = "ids";


namespace XDMF
{
static const std::string position = "position";
static const std::string velocity = "velocity";
static const std::string ids      = "ids";
namespace Motions
{
static const std::string quaternion = "quaternion";
static const std::string velocity   = "velocity";
static const std::string omega      = "omega";
static const std::string force      = "force";
static const std::string torque     = "torque";
} // namespace Motions
} // namespace XDMF
} // namespace ChannelNames


enum class CheckpointIdAdvanceMode
{
    PingPong,
    Incremental
};

struct CheckpointInfo
{
    CheckpointInfo(int every = 0, const std::string& folder = "restart/",
                   CheckpointIdAdvanceMode mode = CheckpointIdAdvanceMode::PingPong);

    int every;
    std::string folder;
    CheckpointIdAdvanceMode mode;
};

// tag used to stop the postprocess side to stop
constexpr int stoppingTag = 424242;
constexpr int stoppingMsg = -1;

// tag used to tell the postprocess side to dump checkpoint 
constexpr int checkpointTag = 434343;

} // namespace mirheo
