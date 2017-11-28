#pragma once

#include <map>
#include <string>
#include <core/logger.h>
#include <core/datatypes.h>

#if __cplusplus < 201400L
namespace std
{
	template<typename T, typename... Args>
	std::unique_ptr<T> make_unique(Args&&... args)
	{
		return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
	}
}
#endif

class ParticlePacker;
class ObjectExtraPacker;

/**
 * \brief Class that holds and manages PinnedBuffers (just buffers later, or channels) of arbitrary data
 *
 * \detailed Used by ParticleVector and ObjectVector to hold data per particle and per object correspondingly
 * Holds data itself, knows whether the data should migrate with the particles in MPI functions
 * and whether the data should be changed by coordinate shift when crossing to another MPI rank
 */
class ExtraDataManager
{
public:

	/**
	 * Struct that contains of data itself (as a unique_ptr to GPUcontainer)
	 * and its properties: needExchange (for MPI) and shiftTypeSize (for shift)
	 */
	struct ChannelDescription
	{
		std::unique_ptr<GPUcontainer> container;
		bool needExchange = false;
		int shiftTypeSize = 0;
	};

	/**
	 * Map of name --> data
	 */
	using ChannelMap = std::map< std::string, ChannelDescription >;


	/**
	 * \brief Allocate a new PinnedBuffer of data
	 *
	 * \param name buffer name
	 * \param size resize buffer to size elements
	 * \tparam T datatype of the buffer element. Sizeof(T) should be divisible by 4
	 */
	template<typename T>
	void createData(const std::string& name, int size = 0)
	{
		if (checkChannelExists(name))
		{
			if (dynamic_cast< PinnedBuffer<T>* >(channelMap[name].container.get()) == nullptr)
				die("Tried to create channel with existing name '%s' but different type", name.c_str());

			debug("Channel '%s' has already been created");
			return;
		}

		if (sizeof(T) % 4 != 0)
			die("Size of an element of the channel '%s' (%d) must be dibisible by 4",
					name.c_str(), sizeof(T));

		info("Creating new channel '%s'");

		auto ptr = std::make_unique< PinnedBuffer<T> >(size);
		channelMap[name].container = std::move(ptr);
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
	 * \brief Make buffer elements be shifted when migrating to another MPI rank
	 *
	 * \detailed When elements of the corresponding buffer are migrated
	 * to another MPI subdomain, a coordinate shift will be applied
	 * to the 'first' 3 floats (if datatypeSize == 4) or
	 * to the 'first' 3 doubles (if datatypeSize == 8) of the element
	 * 'first' refers to the representation of the element as an array of floats or doubles
	 *
	 * Therefore supported structs should look like this:
	 * struct NeedShift { float3 / double3 cooToShift; int exampleOtherVar1; double2 exampleOtherVar2; ... };
	 *
	 * Byte size of the element of the buffer should be at least float4 or double4
	 * (note that 4 is required because of optimal data alignment)
	 *
	 * \param datatypeSize treat coordinates as floats (== 4) or as doubles (== 8)
	 * Other values are not allowed
	 */
	void setShiftType(const std::string& name, int datatypeSize)
	{
		if (datatypeSize != 4 && datatypeSize != 8)
			die("Can only shift float3 or double3 data for MPI communications");

		auto& desc = getChannelDescOrDie(name);

		if (desc.container->datatype_size() < 4*datatypeSize)
			die("Size of an element of the channel '%s' (%d) is too small to apply shift, need to be at least %d",
					name.c_str(), desc.container->datatype_size(), 4*datatypeSize);

		desc.shiftTypeSize = datatypeSize;
	}

	/**
	 * Get buffer by name
	 *
	 * \param name buffer name
	 * \tparam T type of the element of the PinnedBuffer
	 * \return PinnedBuffer<T> corresponding to the given name
	 */
	template<typename T>
	PinnedBuffer<T>* getData(const std::string& name)
	{
		auto& desc = getChannelDescOrDie(name);
		return dynamic_cast< PinnedBuffer<T>* > ( desc.container.get() );
	}

	/**
	 * true if channel with given name exists, false otherwise
	 */
	bool checkChannelExists(const std::string& name) const
	{
		return channelMap.find(name) != channelMap.end();
	}

	/**
	 * Returns constant reference to the channelMap
	 */
	const ChannelMap& getDataMap() const
	{
		return channelMap;
	}

	/**
	 * Returns modifiable reference to the channelMap
	 */
	ChannelMap& getDataMap()
	{
		return channelMap;
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


	void resize(int n, cudaStream_t stream)
	{
		for (auto& kv : channelMap)
			kv.second.container->resize(n, stream);
	}

	void resize_anew(int n)
	{
		for (auto& kv : channelMap)
			kv.second.container->resize_anew(n);
	}

private:

	ChannelMap channelMap;

	/// Helper buffers, used by a Packer
	PinnedBuffer<int>   channelSizes, channelShiftTypes;
	PinnedBuffer<char*> channelPtrs;

	friend class ParticlePacker;
	friend class ObjectExtraPacker;

	/**
	 * Get entry from channelMap or die if it is not found
	 */
	ChannelDescription& getChannelDescOrDie(const std::string& name)
	{
		auto it = channelMap.find(name);
		if (it == channelMap.end())
			die("No such channel: '%s'", name.c_str());

		return it->second;
	}

	/**
	 * Get constant entry from channelMap or die if it is not found
	 */
	const ChannelDescription& getChannelDescOrDie(const std::string& name) const
	{
		auto it = channelMap.find(name);
		if (it == channelMap.end())
			die("No such channel: '%s'", name.c_str());

		return it->second;
	}
};
