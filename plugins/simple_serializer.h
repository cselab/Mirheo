#pragma once

#include <core/containers.h>
#include <core/datatypes.h>

#include <string>

// Only POD types and std::vectors/HostBuffers/PinnedBuffers of POD and std::strings are supported
// Container size will be serialized too
class SimpleSerializer
{
private:
	template<typename T>
	static int sizeOfOne(const HostBuffer<T>& v)
	{
		return v.size() * sizeof(T) + sizeof(int);
	}

	template<typename T>
	static int sizeOfOne(const PinnedBuffer<T>& v)
	{
		return v.size() * sizeof(T) + sizeof(int);
	}

	template<typename T>
	static int sizeOfOne(const std::vector<T>& v)
	{
		return (int)v.size() * sizeof(T) + sizeof(int);
	}

	static int sizeOfOne(const std::string& s)
	{
		return (int)s.length() + sizeof(int);
	}

	template<typename Arg>
	static int sizeOfOne(const Arg& arg)
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
	static int totSize(const Arg& arg)
	{
		return sizeOfOne(arg);
	}

	template<typename Arg, typename... OthArgs>
	static int totSize(const Arg& arg, const OthArgs&... othArgs)
	{
		return sizeOfOne(arg) + totSize(othArgs...);
	}

	//============================================================================

private:
	template<typename T>
	static void packOne(char* buf, const HostBuffer<T>& v)
	{
		*((int*)buf) = v.size();
		buf += sizeof(int);
		memcpy(buf, v.hostPtr(), v.size()*sizeof(T));
	}

	template<typename T>
	static void packOne(char* buf, const PinnedBuffer<T>& v)
	{
		*((int*)buf) = v.size();
		buf += sizeof(int);
		memcpy(buf, v.hostPtr(), v.size()*sizeof(T));
	}

	template<typename T>
	static void packOne(char* buf, const std::vector<T>& v)
	{
		*((int*)buf) = (int)v.size();
		buf += sizeof(int);
		memcpy(buf, v.data(), v.size()*sizeof(T));
	}

	static void packOne(char* buf, const std::string& s)
	{
		*((int*)buf) = (int)s.length();
		buf += sizeof(int);
		memcpy(buf, s.c_str(), s.length());
	}

	template<typename T>
	static void packOne(char* buf, T& v)
	{
		memcpy(buf, &v, sizeOfOne(v));
	}

	//============================================================================

	template<typename Arg>
	static void pack(char* buf, const Arg& arg)
	{
		packOne(buf, arg);
	}

	template<typename Arg, typename... OthArgs>
	static void pack(char* buf, const Arg& arg, const OthArgs&... othArgs)
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

		memcpy(v.hostPtr(), buf, v.size()*sizeof(T));
	}

	template<typename T>
	static void unpackOne(char* buf, PinnedBuffer<T>& v)
	{
		const int sz = *((int*)buf);
		assert(sz >= 0);
		v.resize(sz);
		buf += sizeof(int);

		memcpy(v.hostPtr(), buf, v.size()*sizeof(T));
	}

	template<typename T>
	static void unpackOne(char* buf, std::vector<T>& v)
	{
		const int sz = *((int*)buf);
		assert(sz >= 0);
		v.resize(sz);
		buf += sizeof(int);

		memcpy(v.data(), buf, v.size()*sizeof(T));
	}

	static void unpackOne(char* buf, std::string& s)
	{
		const int sz = *((int*)buf);
		assert(sz >= 0);
		buf += sizeof(int);

		s.assign(buf, buf+sz);
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
	static void serialize(std::vector<char>& buf, Args&... args)
	{
		const int szInBytes = totSize(args...);
		buf.resize(szInBytes);
		pack(buf.data(), args...);
	}

	template<typename... Args>
	static void deserialize(std::vector<char>& buf, Args&... args)
	{
		unpack(buf.data(), args...);
	}


	// Unsafe variants
	template<typename... Args>
	static void serialize(char* to, Args&... args)
	{
		pack(to, args...);
	}

	template<typename... Args>
	static void deserialize(char* from, Args&... args)
	{
		unpack(from, args...);
	}
};

