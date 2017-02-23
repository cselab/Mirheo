#pragma once

#include "../core/datatypes.h"

// Only POD types and vectors/HostBuffers/PinnedBuffers of POD are supported
// Container size will be serialized too
class SimpleSerializer
{
private:
	template<typename T>
	static int sizeOfOne(HostBuffer<T>& v)
	{
		return v.size() * sizeof(T) + sizeof(int);
	}

	template<typename T>
	static int sizeOfOne(PinnedBuffer<T>& v)
	{
		return v.size() * sizeof(T) + sizeof(int);
	}

	template<typename T>
	static int sizeOfOne(std::vector<T>& v)
	{
		return (int)v.size() * sizeof(T) + sizeof(int);
	}

	template<typename Arg>
	static int sizeOfOne(Arg& arg)
	{
		return sizeof(Arg);
	}

	//============================================================================

public:
	static int totSize()
	{
		return 0;
	}

	template<typename Arg>
	static int totSize(Arg& arg)
	{
		return sizeOfOne(arg);
	}

	template<typename Arg, typename... OthArgs>
	static int totSize(Arg& arg, OthArgs&... othArgs)
	{
		return sizeOfOne(arg) + totSize(othArgs...);
	}

	//============================================================================

private:
	template<typename T>
	static void packOne(char* buf, HostBuffer<T>& v)
	{
		*((int*)buf) = v.size();
		buf += sizeof(int);
		memcpy(buf, v.hostPtr(), sizeOfOne(v));
	}

	template<typename T>
	static void packOne(char* buf, PinnedBuffer<T>& v)
	{
		*((int*)buf) = v.size();
		buf += sizeof(int);
		memcpy(buf, v.hostPtr(), sizeOfOne(v));
	}

	template<typename T>
	static void packOne(char* buf, std::vector<T>& v)
	{
		*((int*)buf) = (int)v.size();
		buf += sizeof(int);
		memcpy(buf, v.data(), sizeOfOne(v));
	}

	template<typename T>
	static void packOne(char* buf, T& v)
	{
		memcpy(buf, &v, sizeOfOne(v));
	}

	//============================================================================

	template<typename Arg>
	static void pack(char* buf, Arg& arg)
	{
		packOne(buf, arg);
	}

	template<typename Arg, typename... OthArgs>
	static void pack(char* buf, Arg& arg, OthArgs&... othArgs)
	{
		packOne(buf, arg);
		buf += sizeOfOne(arg);
		pack(buf, othArgs...);
	}

	//============================================================================

	template<typename T>
	static void unpackOne(char* buf, HostBuffer<T>& v)
	{
		const int sz = *((int*)buf);
		assert(sz >= 0);
		v.resize(sz);
		buf += sizeof(int);

		memcpy(v.hostPtr(), buf, sizeOfOne(v));
	}

	template<typename T>
	static void unpackOne(char* buf, PinnedBuffer<T>& v)
	{
		const int sz = *((int*)buf);
		assert(sz >= 0);
		v.resize(sz);
		buf += sizeof(int);

		memcpy(v.hostPtr(), buf, sizeOfOne(v));
	}

	template<typename T>
	static void unpackOne(char* buf, std::vector<T>& v)
	{
		const int sz = *((int*)buf);
		assert(sz >= 0);
		v.resize(sz);
		buf += sizeof(int);

		memcpy(v.data(), buf, sizeOfOne(v));
	}

	template<typename T>
	static void unpackOne(char* buf, T& v)
	{
		memcpy(&v, buf, sizeOfOne(v));
	}

	//============================================================================

	template<typename Arg>
	static void unpack(char* buf, Arg& arg)
	{
		unpackOne(buf, arg);
	}

	template<typename Arg, typename... OthArgs>
	static void unpack(char* buf, Arg& arg, OthArgs&... othArgs)
	{
		unpackOne(buf, arg);
		buf += sizeOfOne(arg);
		unpack(buf, othArgs...);
	}

	//============================================================================

public:
	template<typename... Args>
	static void serialize(HostBuffer<char>& buf, Args&... args)
	{
		const int szInBytes = totSize(args...);
		buf.resize(szInBytes);
		pack(buf.hostPtr(), args...);
	}

	template<typename... Args>
	static void deserialize(HostBuffer<char>& buf, Args&... args)
	{
		unpack(buf.hostPtr(), args...);
	}
};

