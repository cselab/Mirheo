#pragma once

#include <core/pvs/data_manager.h>
#include <core/xdmf/xdmf.h>

#include <set>
#include <string>
#include <vector>

namespace XDMF {struct Channel;}

namespace CheckpointHelpers
{

std::vector<XDMF::Channel> extractShiftPersistentData(const DomainInfo& domain,
                                                      const DataManager& extraData,
                                                      const std::set<std::string>& blackList={});

} // namespace CheckpointHelpers
