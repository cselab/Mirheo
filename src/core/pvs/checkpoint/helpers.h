#pragma once

#include <core/pvs/data_manager.h>
#include <core/xdmf/xdmf.h>

#include <set>
#include <string>
#include <tuple>
#include <vector>

namespace XDMF {struct Channel;}

namespace CheckpointHelpers
{

std::tuple<std::vector<float3>,
           std::vector<float3>,
           std::vector<int64_t>>
splitAndShiftPosVel(const DomainInfo &domain,
                    const PinnedBuffer<float4>& pos4,
                    const PinnedBuffer<float4>& vel4);

std::vector<XDMF::Channel> extractShiftPersistentData(const DomainInfo& domain,
                                                      const DataManager& extraData,
                                                      const std::set<std::string>& blackList={});

} // namespace CheckpointHelpers
