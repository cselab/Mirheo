#include "data_manager.h"

DataManager::DataManager(const DataManager& b)
{
    for (const auto& entry : b.channelMap)
    {
        const auto& name = entry.first;
        const auto& desc = entry.second;

        auto& myDesc = channelMap[name];
        myDesc.persistence = desc.persistence;
        myDesc.shift       = desc.shift;
            
        mpark::visit([&](auto pinnedPtr)
        {
            using T = typename std::remove_pointer<decltype(pinnedPtr)>::type::value_type;
            auto ptr = std::make_unique<PinnedBuffer<T>>(*pinnedPtr);
            myDesc.varDataPtr = ptr.get();
            myDesc.container  = std::move(ptr);
        }, desc.varDataPtr);

        sortedChannels.push_back({name, &channelMap[name]});
    }
    sortChannels();
}

DataManager& DataManager::operator=(const DataManager& b)
{
    DataManager tmp(b);
    swap(*this, tmp);
    return *this;
}

DataManager::DataManager(DataManager&& b)
{
    swap(*this, b);
}

DataManager& DataManager::operator=(DataManager&& b)
{
    swap(*this, b);
    return *this;
}

void swap(DataManager& a, DataManager& b)
{
    std::swap(a.channelMap,     b.channelMap);
    std::swap(a.sortedChannels, b.sortedChannels);
}

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
    std::sort(sortedChannels.begin(),
              sortedChannels.end(),
              [] (NamedChannelDesc ch1, NamedChannelDesc ch2)
    {
        auto size1 = ch1.second->container->datatype_size();
        auto size2 = ch2.second->container->datatype_size();
        return size1 > size2;
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
