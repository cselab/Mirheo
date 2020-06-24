// Copyright 2020 ETH Zurich. All Rights Reserved.
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

/// Used to control the synchronicity of given operations in containers
enum class ContainersSynch
{
    Synch, ///< synchronous
    Asynch ///< asynchronous
};

/** Interface of containers of device (GPU) data
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

    virtual size_t size() const = 0;                                ///< \return number of stored elements
    virtual size_t datatype_size() const = 0;                       ///< \return the size (in bytes) of a single element

    virtual void* genericDevPtr() const = 0;                        ///< \return pointer to device data

    /** \brief resize the internal array. No guarantee to keep the current data.
        \param n New size (in number of elements). Must be non negative.
     */
    virtual void resize_anew(size_t n) = 0;

    /** \brief resize the internal array. Keeps the current data.
        \param n New size (in number of elements). Must be non negative.
        \param stream Used to copy the data internally
     */
    virtual void resize(size_t n, cudaStream_t stream) = 0;

    /** \brief Call cudaMemset on the array.
        \param stream Execution stream
     */
    virtual void clearDevice(cudaStream_t stream) = 0;

    /// Create a new instance of the concrete container implementation
    virtual GPUcontainer* produce() const = 0;
};

//==================================================================================================================
// Device Buffer
//==================================================================================================================

/** \brief Data only on the device (GPU)

    Never releases any memory, keeps a buffer big enough to
    store maximum number of elements it ever held (except in the destructor).

    \tparam T The type of a single element to store.
 */
template<typename T>
class DeviceBuffer : public GPUcontainer
{
public:
#ifndef DOXYGEN_SHOULD_SKIP_THIS // breathe warnings
    /// alias for T. Consistent with std::vector
    using value_type = T;
#endif // DOXYGEN_SHOULD_SKIP_THIS

    /** Construct a DeviceBuffer of given size.
        \param n The initial number of elements
     */
    DeviceBuffer(size_t n = 0)
    {
        resize_anew(n);
    }

    /// Copy constructor
    DeviceBuffer(const DeviceBuffer& b) :
        GPUcontainer{}
    {
        this->copy(b);
    }

    /// Assignment operator
    DeviceBuffer& operator=(const DeviceBuffer& b)
    {
        this->copy(b);
        return *this;
    }

    /// Move constructor; To enable \c std::swap()
    DeviceBuffer(DeviceBuffer&& b)
    {
        *this = std::move(b);
    }

    /// Move assignment; To enable \c std::swap()
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

    ~DeviceBuffer()
    {
        debug4("Destroying DeviceBuffer<%s> of capacity %zu X %zu",
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

    /// \return device pointer to data
    inline T* devPtr() const { return devPtr_; }

    inline void clearDevice(cudaStream_t stream) override
    {
        if (size_ > 0)
            CUDA_Check( cudaMemsetAsync(devPtr_, 0, sizeof(T) * size_, stream) );
    }

    /// clear the device data
    inline void clear(cudaStream_t stream) {
        clearDevice(stream);
    }

    /** \brief Copy data from another container of the same template type.
        Can only copy from another DeviceBuffer of HostBuffer, but not PinnedBuffer.

        \tparam Cont The source container type. Must have the same data type than the current instance.
        \param [in] cont The source container
        \param [in] stream Execution stream
     */
    template<typename Cont>
    auto copy(const Cont& cont, cudaStream_t stream) -> decltype((void)(cont.devPtr()), void()) // use SFINAE here because no forward declaration
    {
        static_assert(std::is_same<decltype(devPtr_), decltype(cont.devPtr())>::value, "can't copy buffers of different types");

        resize_anew(cont.size());
        if (size_ > 0) CUDA_Check( cudaMemcpyAsync(devPtr_, cont.devPtr(), sizeof(T) * size_, cudaMemcpyDeviceToDevice, stream) );
    }

#ifndef DOXYGEN_SHOULD_SKIP_THIS // breathe warnings
    /// \brief copy from host to device.
    template<typename Cont>
    auto copy(const Cont& cont, cudaStream_t stream) -> decltype((void)(cont.hostPtr()), void()) // use SFINAE here because no forward declaration
    {
        static_assert(std::is_same<decltype(devPtr_), decltype(cont.hostPtr())>::value, "can't copy buffers of different types");

        resize_anew(cont.size());
        if (size_ > 0) CUDA_Check( cudaMemcpyAsync(devPtr_, cont.hostPtr(), sizeof(T) * size_, cudaMemcpyHostToDevice, stream) );
    }
#endif // DOXYGEN_SHOULD_SKIP_THIS

    /// synchronous copy
    auto copy(const DeviceBuffer<T>& cont)
    {
        resize_anew(cont.size());
        if (size_ > 0)
            CUDA_Check( cudaMemcpy(devPtr_, cont.devPtr(), sizeof(T) * size_, cudaMemcpyDeviceToDevice) );
    }

    /** \brief Copy the device data of a PinnedBuffer to the internal buffer.
        \param [in] cont the source container
        \param [in] stream The stream used to copy the data.

        \note The copy is performed asynchronously.
        The user must manually synchronize with the stream if needed.
     */
    void copyFromDevice(const PinnedBuffer<T>& cont, cudaStream_t stream)
    {
        resize_anew(cont.size());
        if (size_ > 0) CUDA_Check( cudaMemcpyAsync(devPtr_, cont.devPtr(), sizeof(T) * size_, cudaMemcpyDeviceToDevice, stream) );
    }

    /** \brief Copy the host data of a PinnedBuffer to the internal buffer.
        \param [in] cont the source container
        \param [in] stream The stream used to copy the data.

        \note The copy is performed asynchronously.
        The user must manually synchronize with the stream if needed.
     */
    void copyFromHost(const PinnedBuffer<T>& cont, cudaStream_t stream)
    {
        resize_anew(cont.size());
        if (size_ > 0) CUDA_Check( cudaMemcpyAsync(devPtr_, cont.hostPtr(), sizeof(T) * size_, cudaMemcpyHostToDevice, stream) );
    }

private:
    size_t capacity_  {0}; ///< Storage buffer size
    size_t size_      {0}; ///< Number of elements stored now
    T *devPtr_  {nullptr}; ///< Device pointer to data

    /** \brief Implementation of resize methods.
        \param n new size, must be >= 0
        \param stream data will be copied on that CUDA stream
        \param copy if we need to copy old data to the new allocated buffer
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

        debug4("Allocating DeviceBuffer<%s> from %zu x %zu  to %zu x %zu",
                typeid(T).name(),
                oldsize, datatype_size(),
                size_,   datatype_size());
    }
};



//==================================================================================================================
// Host Buffer
//==================================================================================================================

/** \brief Data only on the host.

    The data is allocated as pinned memory using the CUDA utilities.
    This allows to transfer asynchronously data from the device (e.g. DeviceBuffer).

    Never releases any memory, keeps a buffer big enough to
    store maximum number of elements it ever held (except in the destructor).

    \tparam T The type of a single element to store.
 */
template<typename T>
class HostBuffer
{
public:
    /** \brief construct a HostBuffer with a given size
        \param [in] n The initial number of elements
    */
    HostBuffer(size_t n = 0) { resize_anew(n); }

    /// copy constructor.
    HostBuffer(const HostBuffer& b)
    {
        this->copy(b);
    }

    /// Assignment operator.
    HostBuffer& operator=(const HostBuffer& b)
    {
        this->copy(b);
        return *this;
    }

    /// Move constructor; To enable \c std::swap()
    HostBuffer(HostBuffer&& b)
    {
        *this = std::move(b);
    }

    /// Move assignment; To enable \c std::swap()
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

    ~HostBuffer()
    {
        debug4("Destroying HostBuffer<%s> of capacity %zu X %zu",
               typeid(T).name(), capacity_, sizeof(T));
        CUDA_Check(cudaFreeHost(hostPtr_));
    }

    size_t datatype_size() const { return sizeof(T); } ///< \return the size of a single element (in bytes)
    size_t size()          const { return size_; }     ///< \return the number of elements

    T* hostPtr() const { return hostPtr_; } ///< \return pointer to host memory
    T* data()    const { return hostPtr_; } ///< For uniformity with std::vector

          T& operator[](size_t i)       { return hostPtr_[i]; } ///< \return element with given index
    const T& operator[](size_t i) const { return hostPtr_[i]; } ///< \return element with given index

    /** \brief resize the internal array. Keeps the current data.
        \param n New size (in number of elements). Must be non negative.
     */
    void resize(size_t n) { _resize(n, true);  }

    /** \brief resize the internal array. No guarantee to keep the current data.
        \param n New size (in number of elements). Must be non negative.
     */
    void resize_anew(size_t n) { _resize(n, false); }

    T* begin() { return hostPtr_; }          ///< To support range-based loops
    T* end()   { return hostPtr_ + size_; }  ///< To support range-based loops

    const T* begin() const { return hostPtr_; }          ///< To support range-based loops
    const T* end()   const { return hostPtr_ + size_; }  ///< To support range-based loops

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


    /** \brief Copy data from an arbitrary \c GPUcontainer.
        \param [in] cont a pointer to the source container.
        \param [in] stream Stream used to copy the data.
        \note the type sizes must be compatible (equal or multiple of each other)
    */
    void genericCopy(const GPUcontainer *cont, cudaStream_t stream)
    {
        if (cont->datatype_size() % sizeof(T) != 0)
            die("Incompatible underlying datatype sizes when copying: %zu %% %zu != 0",
                cont->datatype_size(), sizeof(T));

        const size_t typeSizeFactor = cont->datatype_size() / sizeof(T);

        resize(cont->size() * typeSizeFactor);
        if (size_ > 0) CUDA_Check( cudaMemcpyAsync(hostPtr_, cont->genericDevPtr(), sizeof(T) * size_, cudaMemcpyDeviceToHost, stream) );
    }

private:
    size_t capacity_  {0}; ///< Storage buffer size
    size_t size_      {0}; ///< Number of elements stored now
    T* hostPtr_ {nullptr}; ///< Host pointer to data

    /** \brief Implementation of resize methods.
        \param n new size, must be >= 0
        \param copyOldData if we need to copy old data to the new allocated buffer
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

        debug4("Allocating HostBuffer<%s> from %zu x %zu  to %zu x %zu",
                typeid(T).name(),
                oldsize, datatype_size(),
                size_,   datatype_size());
    }
};

//==================================================================================================================
// Pinned Buffer
//==================================================================================================================

/** \brief Device data with mirror host data. Useful to transfer arrays between host and device memory.

    The host data is allocated as pinned memory using the CUDA utilities.
    This allows to transfer asynchronously data from the device.

    Never releases any memory, keeps a buffer big enough to
    store maximum number of elements it ever held (except in the destructor).

    \rst
    .. note::
        Host and device data are not automatically synchronized!
        Use downloadFromDevice() and uploadToDevice() MANUALLY to sync
    \endrst

    \tparam T The type of a single element to store.
 */
template<typename T>
class PinnedBuffer : public GPUcontainer
{
public:
#ifndef DOXYGEN_SHOULD_SKIP_THIS // breathe warnings
    /// alias for T. Consistent with std::vector
    using value_type = T;
#endif // DOXYGEN_SHOULD_SKIP_THIS

    /** Construct a PinnedBuffer with given number of elements
        \param [in] n initial number of elements. Must be non negative.
     */
    PinnedBuffer(size_t n = 0)
    {
        resize_anew(n);
    }

    /// Copy constructor
    PinnedBuffer(const PinnedBuffer& b) :
        GPUcontainer{}
    {
        this->copy(b);
    }

    /// assignment operator
    PinnedBuffer& operator=(const PinnedBuffer& b)
    {
        this->copy(b);
        return *this;
    }

    /// Move constructor; To enable \c std::swap()
    PinnedBuffer (PinnedBuffer&& b)
    {
        *this = std::move(b);
    }

    /// Move assignment; To enable \c std::swap()
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

    ~PinnedBuffer()
    {
        debug4("Destroying PinnedBuffer<%s> of capacity %zu X %zu",
               typeid(T).name(), capacity_, sizeof(T));
        if (devPtr_ != nullptr)
        {
            CUDA_Check(cudaFreeHost(hostPtr_));
            CUDA_Check(cudaFree(devPtr_));
        }
    }

    size_t datatype_size() const final { return sizeof(T); }
    size_t size()          const final { return size_; }

    void* genericDevPtr() const final { return (void*) devPtr(); }

    void resize     (size_t n, cudaStream_t stream) final { _resize(n, stream, true);  }
    void resize_anew(size_t n)                      final { _resize(n, 0,      false); }

    GPUcontainer* produce() const final { return new PinnedBuffer<T>(); }

    T* hostPtr() const { return hostPtr_; }  ///< \return pointer to host data
    T* data()    const { return hostPtr_; }  ///< For uniformity with std::vector
    T* devPtr()  const { return devPtr_; }   ///< \return pointer to device data

    inline       T& operator[](size_t i)       { return hostPtr_[i]; }  ///< allow array-like bracketed access to HOST data
    inline const T& operator[](size_t i) const { return hostPtr_[i]; }  ///< allow array-like bracketed access to HOST data

    T* begin() { return hostPtr_; }          ///< To support range-based loops
    T* end()   { return hostPtr_ + size_; }  ///< To support range-based loops

    const T* begin() const { return hostPtr_; }          ///< To support range-based loops
    const T* end()   const { return hostPtr_ + size_; }  ///< To support range-based loops

    /** \brief Copy internal data from device to host.
        \param stream The stream used to perform the copy
        \param synch Synchronicity of the operation. If synchronous, the call will block until the operation is done.
     */
    void downloadFromDevice(cudaStream_t stream, ContainersSynch synch = ContainersSynch::Synch)
    {
        // TODO: check if we really need to do that
        // maybe everything is already downloaded
        debug4("GPU -> CPU (D2H) transfer of PinnedBuffer<%s>, size %zu x %zu",
               typeid(T).name(), size_, datatype_size());

        if (size_ > 0) CUDA_Check( cudaMemcpyAsync(hostPtr_, devPtr_, sizeof(T) * size_, cudaMemcpyDeviceToHost, stream) );
        if (synch == ContainersSynch::Synch) CUDA_Check( cudaStreamSynchronize(stream) );
    }

    /** \brief Copy the internal data from host to device.
        \param stream The stream used to perform the copy
     */
    void uploadToDevice(cudaStream_t stream)
    {
        debug4("CPU -> GPU (H2D) transfer of PinnedBuffer<%s>, size %zu x %zu",
               typeid(T).name(), size_, datatype_size());

        if (size_ > 0) CUDA_Check(cudaMemcpyAsync(devPtr_, hostPtr_, sizeof(T) * size_, cudaMemcpyHostToDevice, stream));
    }

    /// Set all the bytes to 0 on both host and device
    void clear(cudaStream_t stream)
    {
        clearDevice(stream);
        clearHost();
    }

    /// Set all the bytes to 0 on device only
    void clearDevice(cudaStream_t stream) override
    {
        debug4("Clearing device memory of PinnedBuffer<%s>, size %zu x %zu",
               typeid(T).name(), size_, datatype_size());

        if (size_ > 0) CUDA_Check( cudaMemsetAsync(devPtr_, 0, sizeof(T) * size_, stream) );
    }

    /// Set all the bytes to 0 on host only
    void clearHost()
    {
        debug4("Clearing host memory of PinnedBuffer<%s>, size %zu x %zu",
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

    /** \brief Implementation of resize methods.
        \param n new size, must be >= 0
        \param stream data will be copied on that CUDA stream
        \param copy if we need to copy old data to the new allocated buffer
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

        debug4("Allocating PinnedBuffer<%s> from %zu x %zu  to %zu x %zu",
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
