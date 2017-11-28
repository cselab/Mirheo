#pragma once

#include <map>
#include <string>
#include <core/logger.h>

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

class ExtraDataManager
{
public:

	using DataMap = std::map< std::string, std::unique_ptr<GPUcontainer> >;
	using ShiftMap = std::map<std::string, std::pair<int, int> >;

	template<typename T>
	void createData(const std::string& name, int size = 0)
	{
		if (checkDataExists(name))
		{
			debug("Extra data entry PER PARTICLE '%s' has already been created");
			return;
		}

		info("Creating new extra data entry PER PARTICLE '%s'");

		auto ptr = std::make_unique< PinnedBuffer<T> >(size);
		dataMap[name] = std::move(ptr);
		needExchangeMap[name] = false;
		shiftOffsetMap[name] = {-1, 0};
	}

	void requireExchange(const std::string& name)
	{
		if (checkDataExists(name))
			needExchangeMap[name] = true;
		else
			die("Requested extra data entry PER PARTICLE '%s' is absent", name.c_str());
	}

	void setShiftOffsetType(const std::string& name, int offset, int datatypeSize)
	{
		if (checkDataExists(name))
			shiftOffsetMap[name] = {offset, datatypeSize};
		else
			die("Requested extra data entry PER PARTICLE '%s' is absent", name.c_str());
	}

	template<typename T>
	PinnedBuffer<T>* getData(const std::string& name)
	{
		auto it = dataMap.find(name);
		if (it == dataMap.end())
			die("Requested extra data entry PER PARTICLE '%s' is absent", name.c_str());

		return dynamic_cast< PinnedBuffer<T>* > ( it->second.get() );
	}

	bool checkDataExists(const std::string& name) const
	{
		return dataMap.find(name) != dataMap.end();
	}


	const DataMap& getDataMap() const
	{
		return dataMap;
	}

	DataMap& getDataMap()
	{
		return dataMap;
	}


	bool needExchange(const std::string& name)
	{
		auto it = needExchangeMap.find(name);
		if (it == needExchangeMap.end())
			return false;

		return it->second;
	}

	std::pair<int, int> shiftOffsetType(const std::string& name)
	{
		auto it = shiftOffsetMap.find(name);
		if (it == shiftOffsetMap.end())
			return {-1, 0};

		return it->second;
	}


	void resize(int n, cudaStream_t stream)
	{
		for (auto& kv : dataMap)
			kv.second->resize(n, stream);
	}

	void resize_anew(int n)
	{
		for (auto& kv : dataMap)
			kv.second->resize_anew(n);
	}

private:

	DataMap dataMap;
	ShiftMap shiftOffsetMap;
	std::map<std::string, bool> needExchangeMap;

	// Helper buffers, used by a packer
	PinnedBuffer<int>   channelSizes, channelShiftOffsets, channelShiftTypes;
	PinnedBuffer<char*> channelPtrs;

	friend class ParticlePacker;
	friend class ObjectExtraPacker;
};
