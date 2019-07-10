#include "data_manager.h"

CudaVarPtr getDevPtr(VarPinnedBufferPtr varPinnedBuf)
{
    CudaVarPtr ptr;
    mpark::visit([&](auto pinnedPtr)
    {
        ptr = pinnedPtr->devPtr();
    }, varPinnedBuf);
    return ptr;
}

void DataManager::setPersistenceMode(const std::string& name, DataManager::PersistenceMode persistence)
{
    if (persistence == PersistenceMode::None) return;
    auto& desc = getChannelDescOrDie(name);    
    desc.persistence = persistence;
}

void DataManager::setShiftMode(const std::string& name, DataManager::ShiftMode shift)
{
    if (shift == ShiftMode::None) return;
    auto& desc = getChannelDescOrDie(name);
    desc.shift = shift;
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
    return desc.persistence == PersistenceMode::Active;
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
