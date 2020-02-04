#pragma once

#include <mirheo/core/containers.h>
#include <mirheo/core/logger.h>
#include <mirheo/core/types/type_list.h>
#include <mirheo/core/types/variant_type_device.h>

#include <extern/variant/include/mpark/variant.hpp>

#include <cuda_runtime.h>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace mirheo
{

using VarPinnedBufferPtr = mpark::variant<
#define MAKE_WRAPPER(a) PinnedBuffer<a>*
    MIRHEO_TYPE_TABLE_COMMA(MAKE_WRAPPER)
#undef MAKE_WRAPPER
    >;

CudaVarPtr getDevPtr(VarPinnedBufferPtr varPinnedBuf);

/**
 * \class DataManager
 *
 * @brief Class that holds and manages \c PinnedBuffers (just buffers later, or channels) of arbitrary data
 *
 * Used by \c ParticleVector and \c ObjectVector to hold data per particle and per object correspondingly
 * Holds data itself, knows whether the data should migrate with the particles in MPI functions
 * and whether the data should be changed by coordinate shift when crossing to another MPI rank
 */
class DataManager
{
public:

    enum class PersistenceMode { None, Active };
    enum class ShiftMode       { None, Active };
    
    struct ChannelDescription 
    {
        std::unique_ptr<GPUcontainer> container;
        VarPinnedBufferPtr varDataPtr;
        
        PersistenceMode persistence {PersistenceMode::None};
        ShiftMode       shift       {ShiftMode::None};

        inline bool needShift() const {return shift == ShiftMode::Active;}
    };

    using NamedChannelDesc = std::pair< std::string, const ChannelDescription* >;


    DataManager() = default;

    DataManager           (const DataManager& b);
    DataManager& operator=(const DataManager& b);

    DataManager           (DataManager&& b);
    DataManager& operator=(DataManager&& b);

    ~DataManager() = default;

    friend void swap(DataManager& a, DataManager& b);

    /// Copy channel names and their types from a given DataManager.
    /// Does not copy data or resize buffers. New buffers are empty.
    void copyChannelMap(const DataManager &);
    
    /**
     * Allocate a new \c PinnedBuffer of data
     *
     * @param name buffer name
     * @param size resize buffer to \c size elements
     * \tparam T datatype of the buffer element. \c sizeof(T) should be divisible by 4
     */
    template<typename T>
    void createData(const std::string& name, int size = 0)
    {
        static_assert(sizeof(T) % 4 == 0, "Size of an element of the channel must be divisible by 4");

        using HeldType = PinnedBuffer<T>;

        if (checkChannelExists(name))
        {
            if (!mpark::holds_alternative< HeldType* >(channelMap_[name].varDataPtr))
                die("Tried to create channel with existing name '%s' but different type",
                    name.c_str());

            debug("Channel '%s' has already been created", name.c_str());
            return;
        }

        info("Creating new channel '%s'", name.c_str());

        auto &desc = channelMap_[name];
        auto ptr = std::make_unique<HeldType>(size);
        desc.varDataPtr = ptr.get();
        desc.container  = std::move(ptr);

        sortedChannels_.push_back({name, &channelMap_[name]});
        sortChannels();
    }

    /**
     * set persistence of the data: the data will stick to the particles/objects
     * can only add persistence; does nothing otherwise
     */
    void setPersistenceMode(const std::string& name, PersistenceMode persistence);

    void setShiftMode(const std::string& name, ShiftMode shift);

    /**
     * Get gpu buffer by name
     *
     * @param name buffer name
     * @return pointer to \c GPUcontainer corresponding to the given name
     */
    GPUcontainer* getGenericData(const std::string& name);    
    
    /**
     * Get buffer by name
     *
     * @param name buffer name
     * \tparam T type of the element of the \c PinnedBuffer
     * @return pointer to \c PinnedBuffer<T> corresponding to the given name
     */
    template<typename T>
    PinnedBuffer<T>* getData(const std::string& name)
    {
        using HeldType = PinnedBuffer<T>*;

        auto& desc = getChannelDescOrDie(name);

        if (!mpark::holds_alternative< HeldType >(desc.varDataPtr))
            die("Channel '%s' is holding a different type than the required one", name.c_str());

        return mpark::get<HeldType>(desc.varDataPtr);
    }

    /**
     * Get device buffer pointer regardless type
     *
     * @param name buffer name
     * @return pointer to device data held by the corresponding \c PinnedBuffer
     */
    void* getGenericPtr(const std::string& name);

    /**
     * \c true if channel with given \c name exists, \c false otherwise
     */
    bool checkChannelExists(const std::string& name) const;

    /**
     * @return vector of channels sorted (descending) by size of their elements
     */
    const std::vector<NamedChannelDesc>& getSortedChannels() const;

    /**
     * Returns true if the channel is persistent
     */
    bool checkPersistence(const std::string& name) const;

    /// Resize all the channels, keep their data
    void resize(int n, cudaStream_t stream);

    /// Resize all the channels, don't care about existing data
    void resize_anew(int n);

    /// Get entry from channelMap or die if it is not found
    ChannelDescription& getChannelDescOrDie(const std::string& name);

    /// Get constant entry from channelMap or die if it is not found
    const ChannelDescription& getChannelDescOrDie(const std::string& name) const;

private:
    /// Delete the channel with the given name.
    void deleteChannel(const std::string &name);

    using ChannelMap = std::map< std::string, ChannelDescription >;

    /// Quick access to the channels by name
    ChannelMap channelMap_;

    /**
     * Channels sorted by their element size (large to small)
     * Used by the packers so that larger elements are packed first
     */
    std::vector<NamedChannelDesc> sortedChannels_;

private:
    
    void sortChannels();
};

} // namespace mirheo
