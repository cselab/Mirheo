#pragma once

#include <mirheo/core/containers.h>
#include <string>
#include <cstring>
#include <vector>
#include <type_traits>

// Only POD types and std::vectors/HostBuffers/PinnedBuffers of POD and std::strings are supported
// Container size will be serialized too
class SimpleSerializer
{
private:
    
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
        
        return tot;
    }
    
    /// Overload for the vectors of plain old data
    template<typename Vec, EnableIfPod<ValType<Vec>> = nullptr>
    static int sizeOfVec(const Vec& v)
    {        
        return v.size() * sizeof(ValType<Vec>) + sizeof(int);
    }
    
    template<typename T> static int sizeOfOne(const std::vector <T>& v) { return sizeOfVec(v); }
    template<typename T> static int sizeOfOne(const HostBuffer  <T>& v) { return sizeOfVec(v); }
    template<typename T> static int sizeOfOne(const PinnedBuffer<T>& v) { return sizeOfVec(v); }
    
    static int sizeOfOne(const std::string& s)
    {
        return (int)s.length() + sizeof(int);
    }

    template<typename Arg>
    static int sizeOfOne(__UNUSED const Arg& arg)
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
        *((int*)buf) = v.size();
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
        memcpy(&v, buf, sizeOfOne(v));
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
    template<typename... Args>
    static void serialize(std::vector<char>& buf, const Args&... args)
    {
        const int szInBytes = totSize(args...);
        buf.resize(szInBytes);
        pack(buf.data(), args...);
    }

    template<typename... Args>
    static void deserialize(const std::vector<char>& buf, Args&... args)
    {
        unpack(buf.data(), args...);
    }


    // Unsafe variants
    template<typename... Args>
    static void serialize(char* to, const Args&... args)
    {
        pack(to, args...);
    }

    template<typename... Args>
    static void deserialize(const char* from, Args&... args)
    {
        unpack(from, args...);
    }
};

