#include "common.h"

#include <mirheo/core/logger.h>

#include <algorithm>

namespace mirheo
{

namespace ChannelNames
{

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
    if (ChannelNames::isReserved(name, reservedNames))
        die("Channel '%s' is reserved. Please choose another name");
}

} // namespace ChannelNames

CheckpointInfo::CheckpointInfo(int every, const std::string& folder,
                               CheckpointIdAdvanceMode mode) :
    every(every),
    folder(folder),
    mode(mode)
{}

} // namespace mirheo
