// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/containers.h>

#include <cassert>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

namespace mirheo
{

/** Helper class To serialize and deserialize data.
    This is used to communicate data between simulation and postprocess plugins.

    Only POD types and std::vectors/HostBuffers/PinnedBuffers of POD and std::strings are supported.
 */
class SimpleSerializer
{
private:

    static constexpr int padded(int size, int pad = sizeof(int))
    {
        const int n = (size + pad - 1) / pad;
        return n * pad;
    }

    // Some template shorthand definitions

#if __CUDACC_VER_MAJOR__ >= 9
    template<typename Vec>
    using ValType = typename std::remove_reference< decltype(std::declval<Vec>().operator[](0)) >::type;

    template <typename T>
    using EnableIfPod    = typename std::enable_if<  std::is_pod<T>::value >::type*;

    template <typename T>
    using EnableIfNonPod = typename std::enable_if< !std::is_pod<T>::value >::type*;
#else
    // Patch for the CUDA 8 compiler bug
    // https://devtalk.nvidia.com/default/topic/1018200/nvcc-bug-when-trying-simple-template-metaprogramming/
    template<typename Vec>
    using ValType = Vec;

    template <typename T>
    using EnableIfPod    = typename std::enable_if< std::is_pod<T>::error >::type*;

    template <typename T>
    using EnableIfNonPod = typename std::enable_if< true >::type*;
#endif


    /// Overload for the vectors of NON POD : other vectors or strings
    template<typename Vec, EnableIfNonPod<ValType<Vec>> = nullptr>
    static int sizeOfVec(const Vec& v)
    {
        int tot = sizeof(int);
        for (auto& element : v)
            tot += sizeOfOne(element);

        return padded(tot);
    }

    /// Overload for the vectors of plain old data
    template<typename Vec, EnableIfPod<ValType<Vec>> = nullptr>
    static int sizeOfVec(const Vec& v)
    {
        return padded(v.size() * sizeof(ValType<Vec>) + sizeof(int));
    }

    template<typename T> static int sizeOfOne(const std::vector <T>& v) { return sizeOfVec(v); }
    template<typename T> static int sizeOfOne(const HostBuffer  <T>& v) { return sizeOfVec(v); }
    template<typename T> static int sizeOfOne(const PinnedBuffer<T>& v) { return sizeOfVec(v); }

    static int sizeOfOne(const std::string& s)
    {
        return padded(static_cast<int>(s.length() + sizeof(int)));
    }

    template<typename Arg>
    static int sizeOfOne(__UNUSED const Arg& arg)
    {
        return padded(sizeof(Arg));
    }

    //============================================================================

public:
    /// \return The default total size of one element.
    static int totSize()
    {
        return 0;
    }

    /** The total size of one element.
        \tparam Arg The type of the element
        \param arg The element instance.
        \return The size in bytes of the element.
    */
    template<typename Arg>
    static int totSize(const Arg& arg)
    {
        return sizeOfOne(arg);
    }

    /** The total size of one element.
        \tparam Arg The type of the element
        \tparam OthArgs The types of the element other elements
        \param arg The element instance.
        \param othArgs The other element instances.
        \return The size in bytes of all elements.
    */
    template<typename Arg, typename... OthArgs>
    static int totSize(const Arg& arg, const OthArgs&... othArgs)
    {
        return sizeOfOne(arg) + totSize(othArgs...);
    }

    //============================================================================

private:
    /// Overload for the vectors of plain old data
    template<typename Vec, EnableIfPod<ValType<Vec>> = nullptr>
    static void packVec(char* buf, const Vec& v)
    {
        *((int*)buf) = v.size();
        buf += sizeof(int);

        memcpy(buf, v.data(), v.size()*sizeof(ValType<Vec>));
    }

    /// Overload for the vectors of NON POD : other vectors or strings
    template<typename Vec, EnableIfNonPod<ValType<Vec>> = nullptr>
    static void packVec(char* buf, const Vec& v)
    {
        *((int*)buf) = static_cast<int>(v.size());
        buf += sizeof(int);

        for (auto& element : v)
        {
            packOne(buf, element);
            buf += totSize(element);
        }
    }

    template<typename T> static void packOne(char* buf, const std::vector <T>& v) { packVec(buf, v); }
    template<typename T> static void packOne(char* buf, const HostBuffer  <T>& v) { packVec(buf, v); }
    template<typename T> static void packOne(char* buf, const PinnedBuffer<T>& v) { packVec(buf, v); }

    static void packOne(char* buf, const std::string& s)
    {
        *((int*)buf) = (int)s.length();
        buf += sizeof(int);
        memcpy(buf, s.c_str(), s.length());
    }

    template<typename T>
    static void packOne(char* buf, const T& v)
    {
        memcpy(buf, &v, sizeof(v));
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

     /// Overload for the vectors of plain old data
    template<typename Vec, typename Resize, EnableIfPod<ValType<Vec>> = nullptr>
    static void unpackVec(const char* buf, Vec& v, Resize resize)
    {
        const int sz = *((int*)buf);
        assert(sz >= 0);
        (v.*resize)(sz);
        buf += sizeof(int);

        memcpy(v.data(), buf, v.size()*sizeof(ValType<Vec>));
    }

    /// Overload for the vectors of NON POD : other vectors or strings
    template<typename Vec, typename Resize, EnableIfNonPod<ValType<Vec>> = nullptr>
    static void unpackVec(const char* buf, Vec& v, Resize resize)
    {
        const int sz = *((int*)buf);
        assert(sz >= 0);
        (v.*resize)(sz);
        buf += sizeof(int);

        for (auto& element : v)
        {
            unpackOne(buf, element);
            buf += totSize(element);
        }
    }

    template<typename T> static void unpackOne(const char* buf, std::vector <T>& v) { unpackVec(buf, v, static_cast<void (std::vector<T>::*)(size_t)>(&std::vector<T>::resize) ); }
    template<typename T> static void unpackOne(const char* buf, HostBuffer  <T>& v) { unpackVec(buf, v, &HostBuffer  <T>::resize); }
    template<typename T> static void unpackOne(const char* buf, PinnedBuffer<T>& v) { unpackVec(buf, v, &PinnedBuffer<T>::resize_anew); }

    static void unpackOne(const char* buf, std::string& s)
    {
        const int sz = *((int*)buf);
        assert(sz >= 0);
        buf += sizeof(int);

        s.assign(buf, buf+sz);
    }

    template<typename T>
    static void unpackOne(const char* buf, T& v)
    {
        memcpy(&v, buf, sizeof(v));
    }

    //============================================================================

    template<typename Arg>
    static void unpack(const char* buf, Arg& arg)
    {
        unpackOne(buf, arg);
    }

    template<typename Arg, typename... OthArgs>
    static void unpack(const char* buf, Arg& arg, OthArgs&... othArgs)
    {
        unpackOne(buf, arg);
        buf += sizeOfOne(arg);
        unpack(buf, othArgs...);
    }

    //============================================================================

public:
    /** Serialize multiple elements into a buffer.
        The buffer will be allocated to the correct size.
        \tparam Args The types the elements to serialize.
        \param [in] args The elements to serialize.
        \param [out] buf The buffer that will contain the serialized data.
    */
    template<typename... Args>
    static void serialize(std::vector<char>& buf, const Args&... args)
    {
        const int szInBytes = totSize(args...);
        buf.resize(szInBytes);
        pack(buf.data(), args...);
    }

    /** Deserialize multiple elements from a buffer.
        \tparam Args The types the elements to deserialize.
        \param [out] args The deserialized elements.
        \param [in] buf The buffer that contains the serialized data.
    */
    template<typename... Args>
    static void deserialize(const std::vector<char>& buf, Args&... args)
    {
        unpack(buf.data(), args...);
    }


    /** Serialize multiple elements into a buffer.
        The buffer will **NOT** be allocated to the correct size.
        \tparam Args The types the elements to serialize.
        \param [in] args The elements to serialize.
        \param [out] to The buffer that will contain the serialized data. Must be sufficiently large.
    */
    template<typename... Args>
    static void serialize(char* to, const Args&... args)
    {
        pack(to, args...);
    }

    /** Deserialize multiple elements from a buffer.
        \tparam Args The types the elements to deserialize.
        \param [out] args The deserialized elements.
        \param [in] from The buffer that contains the serialized data.
    */
    template<typename... Args>
    static void deserialize(const char* from, Args&... args)
    {
        unpack(from, args...);
    }
};

} // namespace mirheo
