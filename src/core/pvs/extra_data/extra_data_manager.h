#pragma once

#include <map>
#include <string>
#include <vector>
#include <core/logger.h>
#include <core/containers.h>
#include <core/utils/make_unique.h>

#include <cuda_runtime.h>

class ParticlePacker;
class ObjectExtraPacker;

/**
 * \class ExtraDataManager
 *
 * @brief Class that holds and manages \c PinnedBuffers (just buffers later, or channels) of arbitrary data
 *
 * Used by \c ParticleVector and \c ObjectVector to hold data per particle and per object correspondingly
 * Holds data itself, knows whether the data should migrate with the particles in MPI functions
 * and whether the data should be changed by coordinate shift when crossing to another MPI rank
 */
class ExtraDataManager
{
public:

    /**
     * Struct that contains of data itself (as a unique_ptr to \c GPUcontainer)
     * and its properties: needExchange (for MPI) and shiftTypeSize (for shift)
     */
    struct ChannelDescription
    {
        std::unique_ptr<GPUcontainer> container;
        bool needExchange = false;
        int shiftTypeSize = 0;
    };

    using NamedChannelDesc = std::pair< std::string, const ChannelDescription* >;


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
        if (checkChannelExists(name))
        {
            if (dynamic_cast< PinnedBuffer<T>* >(channelMap[name].container.get()) == nullptr)
                die("Tried to create channel with existing name '%s' but different type", name.c_str());

            debug("Channel '%s' has already been created", name.c_str());
            return;
        }

        if (sizeof(T) % 4 != 0)
            die("Size of an element of the channel '%s' (%d) must be divisible by 4",
                    name.c_str(), sizeof(T));

        info("Creating new channel '%s'", name.c_str());

        auto ptr = std::make_unique< PinnedBuffer<T> >(size);
        channelMap[name].container = std::move(ptr);

        sortedChannels.push_back({name, &channelMap[name]});
        std::sort(sortedChannels.begin(), sortedChannels.end(), [] (NamedChannelDesc ch1, NamedChannelDesc ch2) {
            return ch1.second->container->datatype_size() > ch2.second->container->datatype_size();
        });
    }

    /**
     * Make buffer be communicated by MPI
     */
    void requireExchange(const std::string& name)
    {
        auto& desc = getChannelDescOrDie(name);
        desc.needExchange = true;
    }

    /**
     * @brief Make buffer elements be shifted when migrating to another MPI rank
     *
     * When elements of the corresponding buffer are migrated
     * to another MPI subdomain, a coordinate shift will be applied
     * to the 'first' 3 floats (if \c datatypeSize == 4) or
     * to the 'first' 3 doubles (if \c datatypeSize == 8) of the element
     * 'first' refers to the representation of the element as an array of floats or doubles
     *
     * Therefore supported structs should look like this:
     * \code{.cpp}
     * struct NeedShift { float3 / double3 cooToShift; int exampleOtherVar1; double2 exampleOtherVar2; ... };
     * \endcode
     *
     * \rst
     * .. note::
     *    Elements of the buffer should be aligned to 16-byte boundary
     *    Use \c __align__(16) when defining your structure
     * \endrst
     *
     * @param name channel name
     * @param datatypeSize treat coordinates as \c float (== 4) or as \c double (== 8)
     * Other values are not allowed
     */
    void requireShift(const std::string& name, int datatypeSize)
    {
        if (datatypeSize != 4 && datatypeSize != 8)
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
        auto& desc = getChannelDescOrDie(name);
        return dynamic_cast< PinnedBuffer<T>* > ( desc.container.get() );
    }

    /**
     * Get device buffer pointer regardless type
     *
     * @param name buffer name
     * @return pointer to device data held by the corresponding \c PinnedBuffer
     */
    void* getGenericPtr(const std::string& name)
    {
        auto& desc = getChannelDescOrDie(name);
        return desc.container->genericDevPtr();
    }

    /**
     * \c true if channel with given \c name exists, \c false otherwise
     */
    bool checkChannelExists(const std::string& name) const
    {
        return channelMap.find(name) != channelMap.end();
    }

    /**
     * @return vector of channels sorted (descending) by size of their elements
     */
    const std::vector<NamedChannelDesc>& getSortedChannels() const
    {
        return sortedChannels;
    }

    /**
     * Returns true if the channel has to be exchanged by MPI
     */
    bool checkNeedExchange(const std::string& name) const
    {
        auto& desc = getChannelDescOrDie(name);
        return desc.needExchange;
    }

    /**
     * Returns 0 if no shift needed, 4 if shift with floats needed, 8 -- if shift with doubles
     */
    int shiftTypeSize(const std::string& name) const
    {
        auto& desc = getChannelDescOrDie(name);
        return desc.shiftTypeSize;
    }


    /// Resize all the channels, keep their data
    void resize(int n, cudaStream_t stream)
    {
        for (auto& kv : channelMap)
            kv.second.container->resize(n, stream);
    }

    /// Resize all the channels, don't care about existing data
    void resize_anew(int n)
    {
        for (auto& kv : channelMap)
            kv.second.container->resize_anew(n);
    }

private:


    /// Map of name --> data
    using ChannelMap = std::map< std::string, ChannelDescription >;

    /// Quick access to the channels by name
    ChannelMap channelMap;

    /**
     * Channels sorted by their element size (large to small)
     * Used by the packers so that larger elements are packed first
     */
    std::vector<NamedChannelDesc> sortedChannels;

    /// Helper buffers, used by a Packer
    PinnedBuffer<int>   channelSizes, channelShiftTypes;

    /// Helper buffer, used by a Packer
    PinnedBuffer<char*> channelPtrs;

    friend class ParticlePacker;
    friend class ObjectExtraPacker;

    /// Get entry from channelMap or die if it is not found
    ChannelDescription& getChannelDescOrDie(const std::string& name)
    {
        auto it = channelMap.find(name);
        if (it == channelMap.end())
            die("No such channel: '%s'", name.c_str());

        return it->second;
    }

    /// Get constant entry from channelMap or die if it is not found
    const ChannelDescription& getChannelDescOrDie(const std::string& name) const
    {
        auto it = channelMap.find(name);
        if (it == channelMap.end())
            die("No such channel: '%s'", name.c_str());

        return it->second;
    }
};
