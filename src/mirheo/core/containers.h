#pragma once

#include <mirheo/core/logger.h>

#include <cstring>
#include <type_traits>
#include <utility>
#include <cmath>

#include <cuda_runtime.h>

namespace mirheo
{

// Some forward declarations
template<typename T> class PinnedBuffer;

enum class ContainersSynch
{
    Synch,
    Asynch
};

/**
 * Interface of containers of device (GPU) data
 */
class GPUcontainer
{
public:
    GPUcontainer() = default;
    GPUcontainer           (const GPUcontainer&) = delete;
    GPUcontainer& operator=(const GPUcontainer&) = delete;
    GPUcontainer           (GPUcontainer&&) = delete;
    GPUcontainer& operator=(GPUcontainer&&) = delete;
    virtual ~GPUcontainer() = default;
    
    virtual size_t size() const = 0;                                ///< @return number of stored elements
    virtual size_t datatype_size() const = 0;                       ///< @return sizeof( element )

    virtual void* genericDevPtr() const = 0;                        ///< @return device pointer to the data

    virtual void resize_anew(size_t n) = 0;                         ///< Resize container, don't care about the data. @param n new size, must be >= 0
    virtual void resize     (size_t n, cudaStream_t stream) = 0;    ///< Resize container, keep stored data
                                                                    ///< @param n new size, must be >= 0
                                                                    ///< @param stream data will be copied on that CUDA stream

    virtual void clearDevice(cudaStream_t stream) = 0;
    
    virtual GPUcontainer* produce() const = 0;                      ///< Create a new instance of the concrete container implementation
};

//==================================================================================================================
// Device Buffer
//==================================================================================================================

/**
 * This container keeps data only on the device (GPU)
 *
 * Never releases any memory, keeps a buffer big enough to
 * store maximum number of elements it ever held
 */
template<typename T>
class DeviceBuffer : public GPUcontainer
{
public:

    using value_type = T;
    
    DeviceBuffer(size_t n = 0)
    {
        resize_anew(n);
    }

    DeviceBuffer(const DeviceBuffer& b) :
        GPUcontainer{}
    {
        this->copy(b);
    }
    
    DeviceBuffer& operator=(const DeviceBuffer& b)
    {
        this->copy(b);
        return *this;
    }
    
    /// To enable \c std::swap()
    DeviceBuffer (DeviceBuffer&& b)
    {
        *this = std::move(b);
    }

    /// To enable \c std::swap()
    DeviceBuffer& operator=(DeviceBuffer&& b)
    {
        if (this != &b)
        {
            if (devPtr_)
                CUDA_Check(cudaFree(devPtr_));

            capacity_ = b.capacity_;
            size_     = b.size_;
            devPtr_    = b.devPtr_;

            b.capacity_ = 0;
            b.size_     = 0;
            b.devPtr_    = nullptr;
        }

        return *this;
    }

    /// Release resources and report if debug level is high enough
    ~DeviceBuffer()
    {
        debug4("Destroying DeviceBuffer<%s> of capacity %d X %d",
               typeid(T).name(), capacity_, sizeof(T));
        if (devPtr_ != nullptr)
        {
            CUDA_Check(cudaFree(devPtr_));
        }
    }

    inline size_t datatype_size() const final { return sizeof(T); }
    inline size_t size()          const final { return size_; }

    inline void* genericDevPtr() const final { return (void*) devPtr(); }

    inline void resize     (size_t n, cudaStream_t stream) final { _resize(n, stream, true);  }
    inline void resize_anew(size_t n)                      final { _resize(n, 0,      false); }

    inline GPUcontainer* produce() const final { return new DeviceBuffer<T>(); }

    /// @return typed device pointer to data
    inline T* devPtr() const { return devPtr_; }

    /// Set all the bytes to 0
    inline void clearDevice(cudaStream_t stream) override
    {
        if (size_ > 0)
            CUDA_Check( cudaMemsetAsync(devPtr_, 0, sizeof(T) * size_, stream) );
    }
    
    inline void clear(cudaStream_t stream) {
        clearDevice(stream);
    }

    /**
     * Copy data from another container of the same template type
     * Only can copy from another DeviceBuffer of HostBuffer, but not PinnedBuffer
     */
    template<typename Cont>
    auto copy(const Cont& cont, cudaStream_t stream) -> decltype((void)(cont.devPtr()), void())
    {
        static_assert(std::is_same<decltype(devPtr_), decltype(cont.devPtr())>::value, "can't copy buffers of different types");

        resize_anew(cont.size());
        if (size_ > 0) CUDA_Check( cudaMemcpyAsync(devPtr_, cont.devPtr(), sizeof(T) * size_, cudaMemcpyDeviceToDevice, stream) );
    }

    template<typename Cont>
    auto copy(const Cont& cont, cudaStream_t stream) -> decltype((void)(cont.hostPtr()), void())
    {
        static_assert(std::is_same<decltype(devPtr_), decltype(cont.hostPtr())>::value, "can't copy buffers of different types");

        resize_anew(cont.size());
        if (size_ > 0) CUDA_Check( cudaMemcpyAsync(devPtr_, cont.hostPtr(), sizeof(T) * size_, cudaMemcpyHostToDevice, stream) );
    }

    // synchronous copy
    auto copy(const DeviceBuffer<T>& cont)
    {
        resize_anew(cont.size());
        if (size_ > 0)
            CUDA_Check( cudaMemcpy(devPtr_, cont.devPtr(), sizeof(T) * size_, cudaMemcpyDeviceToDevice) );
    }
    
    /**
     * Copy data from PinnedBuffer of the same template type
     * Need to specify if we copy from host or device side
     */
    void copyFromDevice(const PinnedBuffer<T>& cont, cudaStream_t stream)
    {
        resize_anew(cont.size());
        if (size_ > 0) CUDA_Check( cudaMemcpyAsync(devPtr_, cont.devPtr(), sizeof(T) * size_, cudaMemcpyDeviceToDevice, stream) );
    }
    void copyFromHost(const PinnedBuffer<T>& cont, cudaStream_t stream)
    {
        resize_anew(cont.size());
        if (size_ > 0) CUDA_Check( cudaMemcpyAsync(devPtr_, cont.hostPtr(), sizeof(T) * size_, cudaMemcpyHostToDevice, stream) );
    }

private:
    size_t capacity_  {0}; ///< Storage buffer size
    size_t size_      {0}; ///< Number of elements stored now
    T *devPtr_  {nullptr}; ///< Device pointer to data

    /**
     * Set #size_ = \p n. If n > #capacity, allocate more memory
     * and copy the old data on CUDA stream \p stream (only if \c copy is true)
     *
     * If debug level is high enough, will report cases when the buffer had to grow
     *
     * @param n new size, must be >= 0
     * @param stream data will be copied on that CUDA stream
     * @param copy if we need to copy old data
     */
    void _resize(size_t n, cudaStream_t stream, bool copy)
    {
        T *dold = devPtr_;
        const size_t oldsize = size_;

        size_ = n;
        if (capacity_ >= n) return;

        const size_t conservative_estimate = static_cast<size_t>(std::ceil(1.1 * static_cast<double>(n) + 10.0));
        capacity_ = 128 * ((conservative_estimate + 127) / 128);

        CUDA_Check(cudaMalloc(&devPtr_, sizeof(T) * capacity_));

        if (copy && dold != nullptr)
            if (oldsize > 0) CUDA_Check(cudaMemcpyAsync(devPtr_, dold, sizeof(T) * oldsize, cudaMemcpyDeviceToDevice, stream));

        CUDA_Check(cudaFree(dold));

        debug4("Allocating DeviceBuffer<%s> from %d x %d  to %d x %d",
                typeid(T).name(),
                oldsize, datatype_size(),
                size_,   datatype_size());
    }
};



//==================================================================================================================
// Host Buffer
//==================================================================================================================

/**
 * This container keeps data only on the host (CPU)
 *
 * Allocates pinned memory on host, to speed up host-device data migration
 *
 * Never releases any memory, keeps a buffer big enough to
 * store maximum number of elements it ever held
 */
template<typename T>
class HostBuffer
{
public:
    HostBuffer(size_t n = 0) { resize_anew(n); }

    HostBuffer           (const HostBuffer& b)
    {
        this->copy(b);
    }
    
    HostBuffer& operator=(const HostBuffer& b)
    {
        this->copy(b);
        return *this;
    }

    /// To enable \c std::swap()
    HostBuffer(HostBuffer&& b)
    {
        *this = std::move(b);
    }

    /// To enable \c std::swap()
    HostBuffer& operator=(HostBuffer&& b)
    {
        if (this != &b)
        {
            if (hostPtr_)
                CUDA_Check(cudaFreeHost(hostPtr_));
            
            capacity_ = b.capacity_;
            size_    = b.size_;
            hostPtr_  = b.hostPtr_;

            b.capacity_ = 0;
            b.size_    = 0;
            b.hostPtr_  = nullptr;
        }

        return *this;
    }

    /// Release resources and report if debug level is high enough
    ~HostBuffer()
    {
        debug4("Destroying HostBuffer<%s> of capacity %d X %d",
               typeid(T).name(), capacity_, sizeof(T));
        CUDA_Check(cudaFreeHost(hostPtr_));
    }

    inline size_t datatype_size() const { return sizeof(T); }
    inline size_t size()          const { return size_; }

    inline T* hostPtr() const { return hostPtr_; }
    inline T* data()    const { return hostPtr_; } /// For uniformity with std::vector

    inline       T& operator[](size_t i)       { return hostPtr_[i]; }
    inline const T& operator[](size_t i) const { return hostPtr_[i]; }

    inline void resize     (size_t n) { _resize(n, true);  }
    inline void resize_anew(size_t n) { _resize(n, false); }

    inline       T* begin()       { return hostPtr_; }          /// To support range-based loops
    inline       T* end()         { return hostPtr_ + size_; }  /// To support range-based loops
    
    inline const T* begin() const { return hostPtr_; }          /// To support range-based loops
    inline const T* end()   const { return hostPtr_ + size_; }  /// To support range-based loops

    /// Set all the bytes to 0
    void clear()
    {
        memset(hostPtr_, 0, sizeof(T) * size_);
    }
    
    /// Copy data from a HostBuffer of the same template type
    template<typename Cont>
    auto copy(const Cont& cont) -> decltype((void)(cont.hostPtr()), void())
    {
        static_assert(std::is_same<decltype(hostPtr_),
                      decltype(cont.hostPtr())>::value,
                      "can't copy buffers of different types");

        resize(cont.size());
        memcpy(hostPtr_, cont.hostPtr(), sizeof(T) * size_);
    }

    /// Copy data from a DeviceBuffer of the same template type
    template<typename Cont>
    auto copy(const Cont& cont, cudaStream_t stream) -> decltype((void)(cont.devPtr()), void())
    {
        static_assert(std::is_same<decltype(hostPtr_), decltype(cont.devPtr())>::value, "can't copy buffers of different types");

        resize(cont.size());
        if (size_ > 0) CUDA_Check( cudaMemcpyAsync(hostPtr_, cont.devPtr(), sizeof(T) * size_, cudaMemcpyDeviceToHost, stream) );
    }
    
    
    /// Copy data from an arbitrary GPUcontainer, no need to know the type.
    /// Note the type sizes must be compatible (equal or multiple of each other)
    void genericCopy(const GPUcontainer* cont, cudaStream_t stream)
    {
        if (cont->datatype_size() % sizeof(T) != 0)
            die("Incompatible underlying datatype sizes when copying: %d %% %d != 0",
                cont->datatype_size(), sizeof(T));
        
        const size_t typeSizeFactor = cont->datatype_size() / sizeof(T);
        
        resize(cont->size() * typeSizeFactor);
        if (size_ > 0) CUDA_Check( cudaMemcpyAsync(hostPtr_, cont->genericDevPtr(), sizeof(T) * size_, cudaMemcpyDeviceToHost, stream) );
    }
    
private:
    size_t capacity_  {0}; ///< Storage buffer size
    size_t size_      {0}; ///< Number of elements stored now
    T* hostPtr_ {nullptr}; ///< Host pointer to data

    /**
     * Set #size_ = \e n. If \e n > #capacity_, allocate more memory
     * and copy the old data (only if \e copy is true)
     *
     * If debug level is high enough, will report cases when the buffer had to grow
     *
     * @param n new size, must be >= 0
     * @param copy if we need to copy old data
     */
    void _resize(size_t n, bool copyOldData)
    {
        T * hold = hostPtr_;
        const size_t oldsize = size_;

        size_ = n;
        if (capacity_ >= n) return;

        const size_t conservative_estimate = static_cast<size_t> (std::ceil(1.1 * static_cast<double>(n) + 10.0));
        capacity_ = 128 * ((conservative_estimate + 127) / 128);

        CUDA_Check(cudaHostAlloc(&hostPtr_, sizeof(T) * capacity_, 0));

        if (copyOldData && hold != nullptr)
            if (oldsize > 0) memcpy(hostPtr_, hold, sizeof(T) * oldsize);

        CUDA_Check(cudaFreeHost(hold));

        debug4("Allocating HostBuffer<%s> from %d x %d  to %d x %d",
                typeid(T).name(),
                oldsize, datatype_size(),
                size_,   datatype_size());
    }
};

//==================================================================================================================
// Pinned Buffer
//==================================================================================================================


/**
 * This container keeps data on the device (GPU) and on the host (CPU)
 *
 * Allocates pinned memory on host, to speed up host-device data migration
 *
 * \rst
 * .. note::
 *    Host and device data are not automatically synchronized!
 *    Use downloadFromDevice() and uploadToDevice() MANUALLY to sync
 * \endrst
 *
 * Never releases any memory, keeps a buffer big enough to
 * store maximum number of elements it ever held
 */
template<typename T>
class PinnedBuffer : public GPUcontainer
{
public:

    using value_type = T;
    
    PinnedBuffer(size_t n = 0)
    {
        resize_anew(n);
    }

    PinnedBuffer(const PinnedBuffer& b) :
        GPUcontainer{}
    {
        this->copy(b);
    }

    PinnedBuffer& operator=(const PinnedBuffer& b)
    {
        this->copy(b);
        return *this;
    }

    /// To enable \c std::swap()
    PinnedBuffer (PinnedBuffer&& b)
    {
        *this = std::move(b);
    }

    /// To enable \c std::swap()
    PinnedBuffer& operator=(PinnedBuffer&& b)
    {
        if (this!=&b)
        {
            capacity_ = b.capacity_;
            size_ = b.size_;
            hostPtr_ = b.hostPtr_;
            devPtr_ = b.devPtr_;

            b.capacity_ = 0;
            b.size_ = 0;
            b.devPtr_ = nullptr;
            b.hostPtr_ = nullptr;
        }

        return *this;
    }

    /// Release resources and report if debug level is high enough
    ~PinnedBuffer()
    {
        debug4("Destroying PinnedBuffer<%s> of capacity %d X %d",
               typeid(T).name(), capacity_, sizeof(T));
        if (devPtr_ != nullptr)
        {
            CUDA_Check(cudaFreeHost(hostPtr_));
            CUDA_Check(cudaFree(devPtr_));
        }
    }

    inline size_t datatype_size() const final { return sizeof(T); }
    inline size_t size()          const final { return size_; }

    inline void* genericDevPtr() const final { return (void*) devPtr(); }

    inline void resize     (size_t n, cudaStream_t stream) final { _resize(n, stream, true);  }
    inline void resize_anew(size_t n)                      final { _resize(n, 0,      false); }

    inline GPUcontainer* produce() const final { return new PinnedBuffer<T>(); }

    inline T* hostPtr() const { return hostPtr_; }  ///< @return typed host pointer to data
    inline T* data()    const { return hostPtr_; }  /// For uniformity with std::vector
    inline T* devPtr()  const { return devPtr_; }   ///< @return typed device pointer to data

    inline       T& operator[](size_t i)       { return hostPtr_[i]; }  ///< allow array-like bracketed access to HOST data
    inline const T& operator[](size_t i) const { return hostPtr_[i]; }

    
    inline       T* begin()       { return hostPtr_; }          /// To support range-based loops
    inline       T* end()         { return hostPtr_ + size_; }  /// To support range-based loops
    
    inline const T* begin() const { return hostPtr_; }          /// To support range-based loops
    inline const T* end()   const { return hostPtr_ + size_; }  /// To support range-based loops
    /**
     * Copy data from device to host
     *
     * @param synchronize if false, the call is fully asynchronous.
     * if true, host data will be readily available on the call return.
     */
    inline void downloadFromDevice(cudaStream_t stream, ContainersSynch synch = ContainersSynch::Synch)
    {
        // TODO: check if we really need to do that
        // maybe everything is already downloaded
    	debug4("GPU -> CPU (D2H) transfer of PinnedBuffer<%s>, size %d x %d",
    	                typeid(T).name(), size_, datatype_size());

        if (size_ > 0) CUDA_Check( cudaMemcpyAsync(hostPtr_, devPtr_, sizeof(T) * size_, cudaMemcpyDeviceToHost, stream) );
        if (synch == ContainersSynch::Synch) CUDA_Check( cudaStreamSynchronize(stream) );
    }

    /// Copy data from host to device
    inline void uploadToDevice(cudaStream_t stream)
    {
    	debug4("CPU -> GPU (H2D) transfer of PinnedBuffer<%s>, size %d x %d",
    	                typeid(T).name(), size_, datatype_size());

        if (size_ > 0) CUDA_Check(cudaMemcpyAsync(devPtr_, hostPtr_, sizeof(T) * size_, cudaMemcpyHostToDevice, stream));
    }

    /// Set all the bytes to 0 on both host and device
    inline void clear(cudaStream_t stream)
    {
        clearDevice(stream);
        clearHost();
    }

    /// Set all the bytes to 0 on device only
    inline void clearDevice(cudaStream_t stream) override
    {
    	debug4("Clearing device memory of PinnedBuffer<%s>, size %d x %d",
    	                typeid(T).name(), size_, datatype_size());

        if (size_ > 0) CUDA_Check( cudaMemsetAsync(devPtr_, 0, sizeof(T) * size_, stream) );
    }

    /// Set all the bytes to 0 on host only
    inline void clearHost()
    {
    	debug4("Clearing host memory of PinnedBuffer<%s>, size %d x %d",
    	                typeid(T).name(), size_, datatype_size());

        if (size_ > 0) memset(static_cast<void*>(hostPtr_), 0, sizeof(T) * size_);
    }

    /// Copy data from a DeviceBuffer of the same template type
    void copy(const DeviceBuffer<T>& cont, cudaStream_t stream)
    {
        resize_anew(cont.size());
        if (size_ > 0) CUDA_Check( cudaMemcpyAsync(devPtr_, cont.devPtr(), sizeof(T) * size_, cudaMemcpyDeviceToDevice, stream) );
    }

    /// Copy data from a HostBuffer of the same template type
    void copy(const HostBuffer<T>& cont)
    {
        resize_anew(cont.size());
        memcpy(static_cast<void*>(hostPtr_), static_cast<void*>(cont.hostPtr()), sizeof(T) * size_);
    }

    /// Copy data from a PinnedBuffer of the same template type
    void copy(const PinnedBuffer<T>& cont, cudaStream_t stream)
    {
        resize_anew(cont.size());

        if (size_ > 0)
        {
            CUDA_Check( cudaMemcpyAsync(devPtr_, cont.devPtr(), sizeof(T) * size_, cudaMemcpyDeviceToDevice, stream) );
            memcpy(static_cast<void*>(hostPtr_), static_cast<void*>(cont.hostPtr()), sizeof(T) * size_);
        }
    }

    /// Copy data from device pointer of a PinnedBuffer of the same template type
    void copyDeviceOnly(const PinnedBuffer<T>& cont, cudaStream_t stream)
    {
        resize_anew(cont.size());

        if (size_ > 0)
            CUDA_Check( cudaMemcpyAsync(devPtr_, cont.devPtr(), sizeof(T) * size_, cudaMemcpyDeviceToDevice, stream) );
    }

    /// synchronous copy
    void copy(const PinnedBuffer<T>& cont)
    {
        resize_anew(cont.size());

        if (size_ > 0)
        {
            CUDA_Check( cudaMemcpy(devPtr_, cont.devPtr(), sizeof(T) * size_, cudaMemcpyDeviceToDevice) );
            memcpy(static_cast<void*>(hostPtr_), static_cast<void*>(cont.hostPtr()), sizeof(T) * size_);
        }
    }

private:
    size_t capacity_  {0}; ///< Storage buffers size
    size_t size_     {0}; ///< Number of elements stored now
    T* hostPtr_ {nullptr}; ///< Host pointer to data
    T* devPtr_  {nullptr}; ///< Device pointer to data

    /**
     * Set #size_ = \p n. If n > #capacity_, allocate more memory
     * and copy the old data on CUDA stream \p stream (only if \p copy is true)
     * Copy both host and device data if \p copy is true
     *
     * If debug level is high enough, will report cases when the buffer had to grow
     *
     * @param n new size, must be >= 0
     * @param stream data will be copied on that CUDA stream
     * @param copy if we need to copy old data
     */
    void _resize(size_t n, cudaStream_t stream, bool copy)
    {
        T * hold = hostPtr_;
        T * dold = devPtr_;
        size_t oldsize = size_;

        size_ = n;
        if (capacity_ >= n) return;

        const size_t conservative_estimate = static_cast<size_t>(std::ceil(1.1 * static_cast<double>(n) + 10.0));
        capacity_ = 128 * ((conservative_estimate + 127) / 128);

        debug4("Allocating PinnedBuffer<%s> from %d x %d  to %d x %d",
                typeid(T).name(),
                oldsize, datatype_size(),
                size_,   datatype_size());

        CUDA_Check(cudaHostAlloc(&hostPtr_, sizeof(T) * capacity_, 0));
        CUDA_Check(cudaMalloc(&devPtr_, sizeof(T) * capacity_));

        if (copy && hold != nullptr && oldsize > 0)
        {
            memcpy(static_cast<void*>(hostPtr_), static_cast<void*>(hold), sizeof(T) * oldsize);
            CUDA_Check( cudaMemcpyAsync(devPtr_, dold, sizeof(T) * oldsize, cudaMemcpyDeviceToDevice, stream) );
            CUDA_Check( cudaStreamSynchronize(stream) );
        }

        CUDA_Check(cudaFreeHost(hold));
        CUDA_Check(cudaFree(dold));
    }
};

} // namespace mirheo
