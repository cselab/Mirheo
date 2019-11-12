#include "data_manager.h"

#include <algorithm>

namespace mirheo
{

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
    if (this == &b) return *this;
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
    if (this == &b) return *this;
    swap(*this, b);
    return *this;
}

void DataManager::copyChannelMap(const DataManager &other)
{
    for (const auto &pair : other.channelMap)
    {
        auto it = channelMap.find(pair.first);
        mpark::visit([&pair, it, this](const auto *pinnedBuffer)
        {
            using Buffer = std::decay_t<decltype(*pinnedBuffer)>;
            using T = typename Buffer::value_type;

            if (it == channelMap.end()) {
                this->createData<T>(pair.first);
            } else if (!mpark::holds_alternative<Buffer*>(it->second.varDataPtr)) {
                this->deleteChannel(pair.first);
                this->createData<T>(pair.first);
            }
            this->setPersistenceMode(pair.first, pair.second.persistence);
            this->setShiftMode      (pair.first, pair.second.shift);
        }, pair.second.varDataPtr);
    }

    if (channelMap.size() != other.channelMap.size()) {
        static_assert(
                std::is_same<decltype(channelMap), std::map<std::string, ChannelDescription>>::value,
                "Not anymore using a std::map? Check if it's allowed to delete elements while iterating.");
        // We have too many channels, delete the surplus.
        for (const auto &pair : channelMap)
            if (other.channelMap.find(pair.first) == other.channelMap.end())
                deleteChannel(pair.first);
    }

    sortChannels();
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
        return size1 != size2 ? size1 > size2 : ch1.first < ch2.first;
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


void DataManager::deleteChannel(const std::string& name)
{
    if (!channelMap.erase(name)) {
        die("Channel '%s' not found.", name.c_str());
        return;
    }
    for (auto it = sortedChannels.begin(); it != sortedChannels.end(); ++it) {
        if (it->first == name) {
            sortedChannels.erase(it);
            return;
        }
    }
}

} // namespace mirheo
