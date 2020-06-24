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

/// a variant that points to a PinnedBuffer of one of the supported types (see <mirheo/core/types/type_list.h>)
using VarPinnedBufferPtr = mpark::variant<
#define MAKE_WRAPPER(a) PinnedBuffer<a>*
    MIRHEO_TYPE_TABLE_COMMA(MAKE_WRAPPER)
#undef MAKE_WRAPPER
    >;

/// transform a VarPinnedBufferPtr into a cuda variant, usable in device code.
CudaVarPtr getDevPtr(VarPinnedBufferPtr varPinnedBuf);

/** \brief Container for multiple channels on device and host.

    Used by ParticleVector and ObjectVector to hold data per particle and per object correspondingly.
    All channels are stored as PinnedBuffer, which allows to easily transfer the data between host and device.
    Channels can hold data of types listed in VarPinnedBufferPtr variant.
    See ChannelDescription for the description of one channel.
 */
class DataManager
{
public:
#ifndef DOXYGEN_SHOULD_SKIP_THIS // bug in breathe
    /** The persistence mode describes if the data of a channel
        must be conserved when redistribution accross ranks occurs.

        Example: The velocities of the particles must be copied and reordered together with the
        positions of the particles.
     */
    enum class PersistenceMode { None, Active };

    /** The shift mode describes if the data of a channel
        must be shifted in space when redistribution or exchange accross ranks occurs.

        Example: the positions of the particles must be converted to the local coordinate system
        of the neighbour rank if transfered there, while the velocities require no "shift".
     */
    enum class ShiftMode { None, Active };
#endif // DOXYGEN_SHOULD_SKIP_THIS

    /** \brief Holds information and data of a single channel.

        A channel has a type, persistence mode and shift mode.
     */
    struct ChannelDescription
    {
        std::unique_ptr<GPUcontainer> container; ///< The data stored in the channel. Internally stored as a PinnedBuffer
        VarPinnedBufferPtr varDataPtr; ///< Pointer to container that holds the correct type.

        PersistenceMode persistence {PersistenceMode::None}; ///< The persistence mode of the channel
        ShiftMode       shift       {ShiftMode::None};       ///< The shift mode of the channel

        /// returns true if the channel's data needs to be shifted when exchanged or redistributed.
        inline bool needShift() const {return shift == ShiftMode::Active;}
    };

    /// The full description of a channel, contains its name and description
    using NamedChannelDesc = std::pair< std::string, const ChannelDescription* >;


    DataManager() = default;

    /// copy and move constructors
    /// \{
    DataManager           (const DataManager& b);
    DataManager& operator=(const DataManager& b);

    DataManager           (DataManager&& b);
    DataManager& operator=(DataManager&& b);
    /// \}
    ~DataManager() = default;

    /// swap two DataManager
    friend void swap(DataManager& a, DataManager& b);

    /// Copy channel names and their types from a given DataManager.
    /// Does not copy data or resize buffers. New buffers are empty.
    void copyChannelMap(const DataManager &);

    /** \brief Allocate a new channel
        \tparam T datatype of the buffer element. \c sizeof(T) should be compatible with VarPinnedBufferPtr
        \param [in] name buffer name
        \param [in] size resize buffer to \p size elements

        This method will die if a channel with different type but same name already exists.
        If a channel with the same name and same type exists, this method will not allocate a new channel.
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
        _sortChannels();
    }

    /** \brief Set the persistence mode of the data
        \param [in] name The name of the channel to modify
        \param [in] persistence Persistence mode to add to the channel.

        \rst
        This method will die if the required name does not exist.

        .. warning::
            This method can only increase the persistence. If the channel is already persistent,
            this method can not set its persistent mode to None.

        \endrst
     */
    void setPersistenceMode(const std::string& name, PersistenceMode persistence);

    /** \brief Set the shift mode of the data
        \param [in] name The name of the channel to modify
        \param [in] shift Shift mode to add to the channel.

        \rst
        This method will die if the required name does not exist.

        .. warning::
            This method can only increase the shift mode. If the channel already needs shift,
            this method can not set its shift mode to None.
        \endrst
     */
    void setShiftMode(const std::string& name, ShiftMode shift);

    /** \brief Get gpu buffer by name
        \param [in] name buffer name
        \return pointer to \c GPUcontainer corresponding to the given name

        This method will die if the required name does not exist.
     */
    GPUcontainer* getGenericData(const std::string& name);

    /** \brief Get buffer by name
        \param [in] name buffer name
        \tparam T type of the element of the PinnedBuffer
        \return pointer to PinnedBuffer<T> corresponding to the given name

        This method will die if the required name does not exist or if T is of the wrong type.
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

    /** \brief Get device buffer pointer regardless of its type
        \param [in] name buffer name
        \return pointer to device data held by the corresponding PinnedBuffer

        This method will die if the required name does not exist.
     */
    void* getGenericPtr(const std::string& name);

    /// \c true if channel with given \c name exists, \c false otherwise
    bool checkChannelExists(const std::string& name) const;

    /// \return vector of channels sorted (descending) by size of their elements (and then name)
    const std::vector<NamedChannelDesc>& getSortedChannels() const;

    /// \return \c true if the channel is persistent
    bool checkPersistence(const std::string& name) const;

    /// Resize all the channels and preserve their data
    void resize(int n, cudaStream_t stream);

    /// Resize all the channels without preserving the data
    void resize_anew(int n);

    /// Get channel from its name or die if it is not found
    /// \{
    ChannelDescription& getChannelDescOrDie(const std::string& name);
    const ChannelDescription& getChannelDescOrDie(const std::string& name) const;
    /// \}

private:
    /// Delete the channel with the given name.
    /// dies if the channel does not exist
    void _deleteChannel(const std::string &name);

    /// Sort the channels such that largest types are first
    void _sortChannels();

private:
    using ChannelMap = std::map< std::string, ChannelDescription >;

    /// Quick access to the channels by name
    ChannelMap channelMap_;

    /** Channels sorted by their element size (large to small)
        Used by the packers so that larger elements are packed first
     */
    std::vector<NamedChannelDesc> sortedChannels_;
};

} // namespace mirheo
