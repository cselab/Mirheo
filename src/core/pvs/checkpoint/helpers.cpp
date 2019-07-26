#include "helpers.h"

#include <core/xdmf/type_map.h>
#include <core/xdmf/xdmf.h>
#include <core/utils/cuda_common.h>
#include <core/utils/type_shift.h>
#include <core/containers.h>

namespace CheckpointHelpers
{

template<typename Container>
static void shiftElementsLocal2Global(Container& data, const DomainInfo domain)
{
    auto shift = domain.local2global({0.f, 0.f, 0.f});
    for (auto& d : data) TypeShift::apply(d, shift);    
}

std::vector<XDMF::Channel> extractShiftPersistentData(const DomainInfo& domain,
                                                      const DataManager& extraData,
                                                      const std::set<std::string>& blackList)
{
    std::vector<XDMF::Channel> channels;
    
    for (auto& namedChannelDesc : extraData.getSortedChannels())
    {        
        auto channelName = namedChannelDesc.first;
        auto channelDesc = namedChannelDesc.second;        
        
        if (channelDesc->persistence != DataManager::PersistenceMode::Active)
            continue;

        if (blackList.find(channelName) != blackList.end())
            continue;

        mpark::visit([&](auto bufferPtr)
        {
            using T = typename std::remove_pointer<decltype(bufferPtr)>::type::value_type;
            bufferPtr->downloadFromDevice(defaultStream, ContainersSynch::Synch);

            if (channelDesc->needShift())
                shiftElementsLocal2Global(*bufferPtr, domain);
            
            auto formtype   = XDMF::getDataForm<T>();
            auto numbertype = XDMF::getNumberType<T>();
            auto datatype   = DataTypeWrapper<T>();
            channels.push_back(XDMF::Channel(channelName,
                                             bufferPtr->data(),
                                             formtype, numbertype, datatype));
        }, channelDesc->varDataPtr);
    }

    return channels;
}

} // namespace CheckpointHelpers


