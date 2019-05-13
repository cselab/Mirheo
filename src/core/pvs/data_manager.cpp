#include "data_manager.h"

void DataManager::setPersistenceMode(const std::string& name, DataManager::PersistenceMode persistence)
{
    if (persistence == PersistenceMode::None) return;
    auto& desc = getChannelDescOrDie(name);    
    desc.persistence = persistence;
}

void DataManager::requireShift(const std::string& name, size_t datatypeSize)
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

GPUcontainer* DataManager::getGenericData(const std::string& name)
{
    auto& desc = getChannelDescOrDie(name);
    return desc.container.get();
}
    
void* DataManager::getGenericPtr(const std::string& name)
{
    auto& desc = getChannelDescOrDie(name);
    return desc.container->genericDevPtr();
}

bool DataManager::checkChannelExists(const std::string& name) const
{
    return channelMap.find(name) != channelMap.end();
}

const std::vector<DataManager::NamedChannelDesc>& DataManager::getSortedChannels() const
{
    return sortedChannels;
}

bool DataManager::checkPersistence(const std::string& name) const
{
    auto& desc = getChannelDescOrDie(name);
    return desc.persistence == PersistenceMode::Persistent;
}

int DataManager::shiftTypeSize(const std::string& name) const
{
    auto& desc = getChannelDescOrDie(name);
    return desc.shiftTypeSize;
}

void DataManager::resize(int n, cudaStream_t stream)
{
    for (auto& kv : channelMap)
        kv.second.container->resize(n, stream);
}

void DataManager::resize_anew(int n)
{
    for (auto& kv : channelMap)
        kv.second.container->resize_anew(n);
}




void DataManager::sortChannels()
{
    std::sort(sortedChannels.begin(), sortedChannels.end(), [] (NamedChannelDesc ch1, NamedChannelDesc ch2) {
            return ch1.second->container->datatype_size() > ch2.second->container->datatype_size();
        });
}

DataManager::ChannelDescription& DataManager::getChannelDescOrDie(const std::string& name)
{
    auto it = channelMap.find(name);
    if (it == channelMap.end())
        die("No such channel: '%s'", name.c_str());

    return it->second;
}

const DataManager::ChannelDescription& DataManager::getChannelDescOrDie(const std::string& name) const
{
    auto it = channelMap.find(name);
    if (it == channelMap.end())
        die("No such channel: '%s'", name.c_str());

    return it->second;
}
