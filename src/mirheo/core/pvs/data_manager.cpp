// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "data_manager.h"

#include <algorithm>

namespace mirheo
{

DataManager::DataManager(const DataManager& b)
{
    for (const auto& entry : b.channelMap_)
    {
        const auto& name = entry.first;
        const auto& desc = entry.second;

        auto& myDesc = channelMap_[name];
        myDesc.persistence = desc.persistence;
        myDesc.shift       = desc.shift;

        std::visit([&](auto pinnedPtr)
        {
            using T = typename std::remove_pointer<decltype(pinnedPtr)>::type::value_type;
            auto ptr = std::make_unique<PinnedBuffer<T>>(*pinnedPtr);
            myDesc.varDataPtr = ptr.get();
            myDesc.container  = std::move(ptr);
        }, desc.varDataPtr);

        sortedChannels_.push_back({name, &channelMap_[name]});
    }
    _sortChannels();
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
    for (const auto &pair : other.channelMap_)
    {
        auto it = channelMap_.find(pair.first);
        std::visit([&pair, it, this](const auto *pinnedBuffer)
        {
            using Buffer = std::decay_t<decltype(*pinnedBuffer)>;
            using T = typename Buffer::value_type;

            if (it == channelMap_.end()) {
                this->createData<T>(pair.first);
            } else if (!std::holds_alternative<Buffer*>(it->second.varDataPtr)) {
                this->_deleteChannel(pair.first);
                this->createData<T>(pair.first);
            }
            this->setPersistenceMode(pair.first, pair.second.persistence);
            this->setShiftMode      (pair.first, pair.second.shift);
        }, pair.second.varDataPtr);
    }

    if (channelMap_.size() != other.channelMap_.size()) {
        static_assert(
                std::is_same<decltype(channelMap_), std::map<std::string, ChannelDescription>>::value,
                "Not anymore using a std::map? Check if it's allowed to delete elements while iterating.");
        // We have too many channels, delete the surplus.
        for (const auto &pair : channelMap_)
            if (other.channelMap_.find(pair.first) == other.channelMap_.end())
                _deleteChannel(pair.first);
    }

    _sortChannels();
}

void swap(DataManager& a, DataManager& b)
{
    std::swap(a.channelMap_,     b.channelMap_);
    std::swap(a.sortedChannels_, b.sortedChannels_);
}

CudaVarPtr getDevPtr(VarPinnedBufferPtr varPinnedBuf)
{
    CudaVarPtr ptr;
    std::visit([&](auto pinnedPtr)
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
    return channelMap_.find(name) != channelMap_.end();
}

const std::vector<DataManager::NamedChannelDesc>& DataManager::getSortedChannels() const
{
    return sortedChannels_;
}

bool DataManager::checkPersistence(const std::string& name) const
{
    auto& desc = getChannelDescOrDie(name);
    return desc.persistence == PersistenceMode::Active;
}

void DataManager::resize(int n, cudaStream_t stream)
{
    for (auto& kv : channelMap_)
        kv.second.container->resize(n, stream);
}

void DataManager::resize_anew(int n)
{
    for (auto& kv : channelMap_)
        kv.second.container->resize_anew(n);
}



void DataManager::_sortChannels()
{
    std::sort(sortedChannels_.begin(),
              sortedChannels_.end(),
              [] (NamedChannelDesc ch1, NamedChannelDesc ch2)
    {
        auto size1 = ch1.second->container->datatype_size();
        auto size2 = ch2.second->container->datatype_size();
        return size1 != size2 ? size1 > size2 : ch1.first < ch2.first;
    });
}

DataManager::ChannelDescription *DataManager::getChannelDesc(const std::string& name)
{
    auto it = channelMap_.find(name);
    return it != channelMap_.end() ? &it->second : nullptr;
}

const DataManager::ChannelDescription *DataManager::getChannelDesc(const std::string& name) const
{
    auto it = channelMap_.find(name);
    return it != channelMap_.end() ? &it->second : nullptr;
}

DataManager::ChannelDescription& DataManager::getChannelDescOrDie(const std::string& name)
{
    auto * const ptr = getChannelDesc(name);
    if (ptr == nullptr)
        die("No such channel: '%s'", name.c_str());
    return *ptr;
}

const DataManager::ChannelDescription& DataManager::getChannelDescOrDie(const std::string& name) const
{
    const auto * const ptr = getChannelDesc(name);
    if (ptr == nullptr)
        die("No such channel: '%s'", name.c_str());
    return *ptr;
}


void DataManager::_deleteChannel(const std::string& name)
{
    if (!channelMap_.erase(name))
    {
        die("Channel '%s' not found.", name.c_str());
        return;
    }

    for (auto it = sortedChannels_.begin(); it != sortedChannels_.end(); ++it)
    {
        if (it->first == name)
        {
            sortedChannels_.erase(it);
            return;
        }
    }
}

} // namespace mirheo
