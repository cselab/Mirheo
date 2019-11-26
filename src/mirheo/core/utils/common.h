#pragma once

#include <vector>
#include <string>

namespace mirheo
{

/**
 * Channel names used in several places of the program
 */
namespace ChannelNames
{

// per entity fields (particles or objects)
extern const std::string globalIds;

// per particle fields
extern const std::string positions;
extern const std::string velocities;
extern const std::string forces;
extern const std::string stresses;
extern const std::string densities;
extern const std::string oldPositions;

// per object fields
extern const std::string motions;
extern const std::string oldMotions;
extern const std::string comExtents;
extern const std::string areaVolumes;

extern const std::string membraneTypeId;
    
// per object, specific to Juelicher bending + ADE    
extern const std::string areas;
extern const std::string meanCurvatures;
extern const std::string lenThetaTot;

// per bisegment data
extern const std::string polyStates;
extern const std::string energies;
extern const std::string rodKappa;
extern const std::string rodTau_l;

extern const std::vector<std::string> reservedParticleFields;
extern const std::vector<std::string> reservedObjectFields;
extern const std::vector<std::string> reservedBisegmentFields;

bool isReserved    (const std::string& name, const std::vector<std::string>& reservedNames);
void failIfReserved(const std::string& name, const std::vector<std::string>& reservedNames);

// channel names used in the xdmf format
namespace XDMF
{
extern const std::string position;
extern const std::string velocity;
extern const std::string ids;
namespace Motions
{
extern const std::string quaternion;
extern const std::string velocity;
extern const std::string omega;
extern const std::string force;
extern const std::string torque;
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

    bool needDump() const;

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
