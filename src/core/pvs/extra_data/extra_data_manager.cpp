#include "extra_data_manager.h"

void ExtraDataManager::requireExchange(const std::string& name)
{
    auto& desc = getChannelDescOrDie(name);
    desc.needExchange = true;
}

void ExtraDataManager::requireShift(const std::string& name, int datatypeSize)
{
    if (datatypeSize != sizeof(float) && datatypeSize != sizeof(double))
        die("Can only shift float3 or double3 data for MPI communications");

    auto& desc = getChannelDescOrDie(name);

    if ( (desc.container->datatype_size() % sizeof(float4)) != 0)
        die("Incorrect alignment of channel '%s' elements. Size (now %d) should be divisible by 16",
            name.c_str(), desc.container->datatype_size());

    if (desc.container->datatype_size() < 3*datatypeSize)
        die("Size of an element of the channel '%s' (%d) is too small to apply shift, need to be at least %d",
            name.c_str(), desc.container->datatype_size(), 4*datatypeSize);

    desc.shiftTypeSize = datatypeSize;
}

GPUcontainer* ExtraDataManager::getGenericData(const std::string& name)
{
    auto& desc = getChannelDescOrDie(name);
    return desc.container.get();
}
    
void* ExtraDataManager::getGenericPtr(const std::string& name)
{
    auto& desc = getChannelDescOrDie(name);
    return desc.container->genericDevPtr();
}

bool ExtraDataManager::checkChannelExists(const std::string& name) const
{
    return channelMap.find(name) != channelMap.end();
}

const std::vector<ExtraDataManager::NamedChannelDesc>& ExtraDataManager::getSortedChannels() const
{
    return sortedChannels;
}

bool ExtraDataManager::checkNeedExchange(const std::string& name) const
{
    auto& desc = getChannelDescOrDie(name);
    return desc.needExchange;
}

int ExtraDataManager::shiftTypeSize(const std::string& name) const
{
    auto& desc = getChannelDescOrDie(name);
    return desc.shiftTypeSize;
}

void ExtraDataManager::resize(int n, cudaStream_t stream)
{
    for (auto& kv : channelMap)
        kv.second.container->resize(n, stream);
}

void ExtraDataManager::resize_anew(int n)
{
    for (auto& kv : channelMap)
        kv.second.container->resize_anew(n);
}




void ExtraDataManager::sortChannels()
{
    std::sort(sortedChannels.begin(), sortedChannels.end(), [] (NamedChannelDesc ch1, NamedChannelDesc ch2) {
            return ch1.second->container->datatype_size() > ch2.second->container->datatype_size();
        });
}

ExtraDataManager::ChannelDescription& ExtraDataManager::getChannelDescOrDie(const std::string& name)
{
    auto it = channelMap.find(name);
    if (it == channelMap.end())
        die("No such channel: '%s'", name.c_str());

    return it->second;
}

const ExtraDataManager::ChannelDescription& ExtraDataManager::getChannelDescOrDie(const std::string& name) const
{
    auto it = channelMap.find(name);
    if (it == channelMap.end())
        die("No such channel: '%s'", name.c_str());

    return it->second;
}
