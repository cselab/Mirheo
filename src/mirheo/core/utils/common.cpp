#include "common.h"

#include <mirheo/core/logger.h>
#include <mirheo/core/utils/config.h>

#include <algorithm>
#include <cassert>

namespace mirheo
{

namespace channel_names
{

const std::string globalIds   = "ids";

const std::string positions     = "positions";
const std::string velocities    = "velocities";
const std::string forces        = "__forces"; // forces are special, as they are not available di

const std::string stresses      = "stresses";
const std::string densities     = "densities";
const std::string oldPositions  = "old_positions";

const std::string motions     = "motions";
const std::string oldMotions  = "old_motions";
const std::string comExtents  = "com_extents";
const std::string areaVolumes = "area_volumes";

const std::string membraneTypeId = "membrane_type_id";
    
const std::string areas          = "areas";
const std::string meanCurvatures = "mean_curvatures";
const std::string lenThetaTot    = "len_theta_tot";

const std::string polyStates    = "states";
const std::string energies      = "energies";
const std::string rodKappa      = "biseg_kappa";
const std::string rodTau_l      = "biseg_tau_l";


const std::vector<std::string> reservedParticleFields =
    {globalIds, positions, velocities, forces, stresses, densities, oldPositions};

const std::vector<std::string> reservedObjectFields =
    {globalIds, motions, oldMotions, comExtents, areaVolumes, membraneTypeId,
     areas, meanCurvatures, lenThetaTot};

const std::vector<std::string> reservedBisegmentFields =
    {polyStates, energies, rodKappa, rodTau_l};

bool isReserved(const std::string& name, const std::vector<std::string>& reservedNames)
{
    auto it = std::find(reservedNames.begin(), reservedNames.end(), name);
    return it != reservedNames.end();
}

void failIfReserved(const std::string& name, const std::vector<std::string>& reservedNames)
{
    if (channel_names::isReserved(name, reservedNames))
        die("Channel '%s' is reserved. Please choose another name", name.c_str());
}

namespace XDMF
{
const std::string position = "positions";
const std::string velocity = "velocities";
const std::string ids      = "ids";
namespace motions
{
const std::string quaternion = "quaternions";
const std::string velocity   = "velocities";
const std::string omega      = "omegas";
const std::string force      = "forces";
const std::string torque     = "torques";
} // namespace motions
} // namespace XDMF


} // namespace channel_names

CheckpointInfo::CheckpointInfo(int every_, const std::string& folder_,
                               CheckpointIdAdvanceMode mode_,
                               CheckpointMechanism mechanism_) :
    every(every_),
    folder(folder_),
    mode(mode_),
    mechanism(mechanism_)
{}

bool CheckpointInfo::needDump() const
{
    return every != 0;
}

ConfigValue TypeLoadSave<CheckpointInfo>::save(Saver& saver, const CheckpointInfo& info)
{
    return ConfigValue::Object{
        {"__type",    saver("CheckpointInfo")},
        {"every",     saver(info.every)},
        {"folder",    saver(info.folder)},
        {"mode",      saver(info.mode)},
        {"mechanism", saver(info.mechanism)},
    };
}

CheckpointInfo TypeLoadSave<CheckpointInfo>::load(Loader&, const ConfigValue& config)
{
    assert(config["__type"] == "CheckpointInfo");
    return CheckpointInfo{config["every"], config["folder"], config["mode"],
                          config["mechanism"]};
}

} // namespace mirheo
