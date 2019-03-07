#pragma once

#include <map>
#include <string>
#include <vector>
#include <core/logger.h>
#include <core/containers.h>
#include <core/utils/make_unique.h>
#include <core/utils/typeMap.h>

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

    enum class PersistenceMode
    {
        None, Persistent
    };
    
    /**
     * Struct that contains of data itself (as a unique_ptr to \c GPUcontainer)
     * and its properties: needExchange (for MPI) and shiftTypeSize (for shift)
     */
    struct ChannelDescription 
    {
        std::unique_ptr<GPUcontainer> container;
        PersistenceMode persistence = PersistenceMode::None;
        int shiftTypeSize = 0;
        DataType dataType;
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
        channelMap[name].dataType  = typeTokenize<T>();

        sortedChannels.push_back({name, &channelMap[name]});
        sortChannels();
    }

    /**
     * set persistence of the data: the data will stick to the particles/objects
     * can only add persistence; does nothing otherwise
     */
    void setPersistenceMode(const std::string& name, PersistenceMode persistence);

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
    void requireShift(const std::string& name, size_t datatypeSize);

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
        return dynamic_cast< PinnedBuffer<T>* > ( getGenericData(name) );
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

    /**
     * Returns 0 if no shift needed, 4 if shift with floats needed, 8 -- if shift with doubles
     */
    int shiftTypeSize(const std::string& name) const;

    /// Resize all the channels, keep their data
    void resize(int n, cudaStream_t stream);

    /// Resize all the channels, don't care about existing data
    void resize_anew(int n);

    /// Get entry from channelMap or die if it is not found
    ChannelDescription& getChannelDescOrDie(const std::string& name);

    /// Get constant entry from channelMap or die if it is not found
    const ChannelDescription& getChannelDescOrDie(const std::string& name) const;

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

    friend class DevicePacker;
    friend class ParticlePacker;
    friend class ObjectExtraPacker;    

    void sortChannels();
};
