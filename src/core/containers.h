#pragma once

#include <core/logger.h>

#include <cassert>
#include <type_traits>
#include <utility>
#include <algorithm>
#include <typeinfo>


/**
 * Interface of containers of device (GPU) data
 */
class GPUcontainer
{
public:
	virtual int size() const = 0;                                      ///< @return number of stored elements
	virtual int datatype_size() const = 0;                             ///< @return sizeof( element )

	virtual void* genericDevPtr() const = 0;                           ///< @return device pointer to the data

	virtual void resize_anew(const int n) = 0;                         ///< Resize container, don't care about the data. @param n new size, must be >= 0
	virtual void resize     (const int n, cudaStream_t stream) = 0;    ///< Resize container, keep stored data
	                                                                   ///< @param n new size, must be >= 0
                                                                       ///< @param stream data will be copied on that CUDA stream

	virtual GPUcontainer* produce() const = 0;                         ///< Create a new instance of the concrete container implementation

	virtual ~GPUcontainer() = default;
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
private:
	int capacity;  ///< Storage buffer size
	int _size;     ///< Number of elements stored now
	T* devptr;     ///< Device pointer to data

	/**
	 * Set #_size = \p n. If n > #capacity, allocate more memory
	 * and copy the old data on CUDA stream \p stream (only if \c copy is true)
	 *
	 * If debug level is high enough, will report cases when the buffer had to grow
	 *
	 * @param n new size, must be >= 0
	 * @param stream data will be copied on that CUDA stream
	 * @param copy if we need to copy old data
	 */
	void _resize(const int n, cudaStream_t stream, bool copy)
	{
		T * dold = devptr;
		int oldsize = _size;

		if (n < 0) die("Requested negative size %d", n);
		_size = n;
		if (capacity >= n) return;

		const int conservative_estimate = (int)ceil(1.1 * n + 10);
		capacity = 128 * ((conservative_estimate + 127) / 128);

		CUDA_Check(cudaMalloc(&devptr, sizeof(T) * capacity));

		if (copy && dold != nullptr)
			if (oldsize > 0) CUDA_Check(cudaMemcpyAsync(devptr, dold, sizeof(T) * oldsize, cudaMemcpyDeviceToDevice, stream));

		CUDA_Check(cudaFree(dold));

		debug4("Allocating DeviceBuffer<%s> from %d x %d  to %d x %d",
				typeid(T).name(),
				oldsize, datatype_size(),
				_size,   datatype_size());
	}

public:

	DeviceBuffer(int n = 0) :
		capacity(0), _size(0), devptr(nullptr)
	{
		resize_anew(n);
	}

	/// To enable \c std::swap()
	DeviceBuffer (DeviceBuffer&& b)
	{
		*this = std::move(b);
	}

	/// To enable \c std::swap()
	DeviceBuffer& operator=(DeviceBuffer&& b)
	{
		if (this!=&b)
		{
			capacity = b.capacity;
			_size = b._size;
			devptr = b.devptr;

			b.capacity = 0;
			b._size = 0;
			b.devptr = nullptr;
		}

		return *this;
	}

	/// Release resources and report if debug level is high enough
	~DeviceBuffer()
	{
		if (devptr != nullptr)
		{
			CUDA_Check(cudaFree(devptr));
			debug4("Destroying DeviceBuffer<%s>", typeid(T).name());
		}
	}

	inline int datatype_size() const final { return sizeof(T); }
	inline int size()          const final { return _size; }

	inline void* genericDevPtr() const final { return (void*) devPtr(); }

	inline void resize     (const int n, cudaStream_t stream) final { _resize(n, stream, true);  }
	inline void resize_anew(const int n)                      final { _resize(n, 0,      false); }

	inline GPUcontainer* produce() const final { return new DeviceBuffer<T>(); }

	/// @return typed device pointer to data
	inline T* devPtr() const { return devptr; }

	/// Set all the bytes to 0
	inline void clear(cudaStream_t stream)
	{
		if (_size > 0) CUDA_Check( cudaMemsetAsync(devptr, 0, sizeof(T) * _size, stream) );
	}

	/**
	 * Copy data from another container of the same template type
	 * Only can copy from another DeviceBuffer of HostBuffer, but not PinnedBuffer
	 */
	template<typename Cont>
	auto copy(const Cont& cont, cudaStream_t stream) -> decltype((void)(cont.devPtr()), void())
	{
		static_assert(std::is_same<decltype(devptr), decltype(cont.devPtr())>::value, "can't copy buffers of different types");

		resize_anew(cont.size());
		if (_size > 0) CUDA_Check( cudaMemcpyAsync(devptr, cont.devPtr(), sizeof(T) * _size, cudaMemcpyDeviceToDevice, stream) );
	}

	template<typename Cont>
	auto copy(const Cont& cont, cudaStream_t stream) -> decltype((void)(cont.hostPtr()), void())
	{
		static_assert(std::is_same<decltype(devptr), decltype(cont.hostPtr())>::value, "can't copy buffers of different types");

		resize_anew(cont.size());
		if (_size > 0) CUDA_Check( cudaMemcpyAsync(devptr, cont.hostPtr(), sizeof(T) * _size, cudaMemcpyHostToDevice, stream) );
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
private:
	int capacity;   ///< Storage buffer size
	int _size;      ///< Number of elements stored now
	T * hostptr;    ///< Host pointer to data

	/**
	 * Set #_size = \e n. If \e n > #capacity, allocate more memory
	 * and copy the old data (only if \e copy is true)
	 *
	 * If debug level is high enough, will report cases when the buffer had to grow
	 *
	 * @param n new size, must be >= 0
	 * @param copy if we need to copy old data
	 */
	void _resize(const int n, bool copy)
	{
		T * hold = hostptr;
		int oldsize = _size;

		if (n < 0) die("Requested negative size %d", n);
		_size = n;
		if (capacity >= n) return;

		const int conservative_estimate = (int)ceil(1.1 * n + 10);
		capacity = 128 * ((conservative_estimate + 127) / 128);

		hostptr = (T*) malloc(sizeof(T) * capacity);

		if (copy && hold != nullptr)
			if (oldsize > 0) memcpy(hostptr, hold, sizeof(T) * oldsize);

		free(hold);

		debug4("Allocating HostBuffer<%s> from %d x %d  to %d x %d",
				typeid(T).name(),
				oldsize, datatype_size(),
				_size,   datatype_size());
	}

public:
	HostBuffer(int n = 0): capacity(0), _size(0), hostptr(nullptr) { resize_anew(n); }

	/// To enable \c std::swap()
	HostBuffer(HostBuffer&& b)
	{
		*this = std::move(b);
	}

	/// To enable \c std::swap()
	HostBuffer& operator=(HostBuffer&& b)
	{
		if (this!=&b)
		{
			capacity = b.capacity;
			_size = b._size;
			hostptr = b.hostptr;

			b.capacity = 0;
			b._size = 0;
			b.hostptr = nullptr;
		}

		return *this;
	}

	/// Release resources and report if debug level is high enough
	~HostBuffer()
	{
		free(hostptr);
		debug4("Destroying HostBuffer<%s>", typeid(T).name());
	}

	inline int datatype_size() const { return sizeof(T); }
	inline int size()          const { return _size; }

	inline T* hostPtr() const { return hostptr; }

	inline       T& operator[](int i)       { return hostptr[i]; }
	inline const T& operator[](int i) const { return hostptr[i]; }

	inline void resize     (const int n) { _resize(n, true);  }
	inline void resize_anew(const int n) { _resize(n, false); }

	inline T* begin() { return hostptr; }          /// To support range-based loops
	inline T* end()   { return hostptr + _size; }  /// To support range-based loops

	/// Set all the bytes to 0
	void clear()
	{
		memset(hostptr, 0, sizeof(T) * _size);
	}

	/// Copy data from a HostBuffer of the same template type
	template<typename Cont>
	auto copy(const Cont& cont) -> decltype((void)(cont.hostPtr()), void())
	{
		static_assert(std::is_same<decltype(hostptr), decltype(cont.hostPtr())>::value, "can't copy buffers of different types");

		resize(cont.size());
		memcpy(hostptr, cont.hostPtr(), sizeof(T) * _size);
	}

	/// Copy data from a DeviceBuffer of the same template type
	template<typename Cont>
	auto copy(const Cont& cont, cudaStream_t stream) -> decltype((void)(cont.devPtr()), void())
	{
		static_assert(std::is_same<decltype(hostptr), decltype(cont.devPtr())>::value, "can't copy buffers of different types");

		resize(cont.size());
		if (_size > 0) CUDA_Check( cudaMemcpyAsync(hostptr, cont.devPtr(), sizeof(T) * _size, cudaMemcpyDeviceToHost, stream) );
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
private:
	int capacity;   ///< Storage buffers size
	int _size;      ///< Number of elements stored now
	T * hostptr;    ///< Host pointer to data
	T * devptr;     ///< Device pointer to data

	/**
	 * Set #_size = \p n. If n > #capacity, allocate more memory
	 * and copy the old data on CUDA stream \p stream (only if \p copy is true)
	 * Copy both host and device data if \p copy is true
	 *
	 * If debug level is high enough, will report cases when the buffer had to grow
	 *
	 * @param n new size, must be >= 0
	 * @param stream data will be copied on that CUDA stream
	 * @param copy if we need to copy old data
	 */
	void _resize(const int n, cudaStream_t stream, bool copy)
	{
		T * hold = hostptr;
		T * dold = devptr;
		int oldsize = _size;

		if (n < 0) die("Requested negative size %d", n);
		_size = n;
		if (capacity >= n) return;

		const int conservative_estimate = (int)ceil(1.1 * n + 10);
		capacity = 128 * ((conservative_estimate + 127) / 128);

		CUDA_Check(cudaHostAlloc(&hostptr, sizeof(T) * capacity, 0));
		CUDA_Check(cudaMalloc(&devptr, sizeof(T) * capacity));

		if (copy && hold != nullptr && oldsize > 0)
		{
			memcpy(hostptr, hold, sizeof(T) * oldsize);
			CUDA_Check( cudaMemcpyAsync(devptr, dold, sizeof(T) * oldsize, cudaMemcpyDeviceToDevice, stream) );
			CUDA_Check( cudaStreamSynchronize(stream) );
		}

		CUDA_Check(cudaFreeHost(hold));
		CUDA_Check(cudaFree(dold));

		debug4("Allocating PinnedBuffer<%s> from %d x %d  to %d x %d",
				typeid(T).name(),
				oldsize, datatype_size(),
				_size,   datatype_size());
	}

public:
	PinnedBuffer(int n = 0) :
		capacity(0), _size(0), hostptr(nullptr), devptr(nullptr)
	{
		resize_anew(n);
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
			capacity = b.capacity;
			_size = b._size;
			hostptr = b.hostptr;
			devptr = b.devptr;

			b.capacity = 0;
			b._size = 0;
			b.devptr = nullptr;
			b.hostptr = nullptr;
		}

		return *this;
	}

	/// Release resources and report if debug level is high enough
	~PinnedBuffer()
	{
		if (devptr != nullptr)
		{
			CUDA_Check(cudaFreeHost(hostptr));
			CUDA_Check(cudaFree(devptr));
			debug4("Destroying PinnedBuffer<%s>", typeid(T).name());
		}
	}

	inline int datatype_size() const final { return sizeof(T); }
	inline int size()          const final { return _size; }

	inline void* genericDevPtr() const final { return (void*) devPtr(); }

	inline void resize     (const int n, cudaStream_t stream) final { _resize(n, stream, true);  }
	inline void resize_anew(const int n)                      final { _resize(n, 0,      false); }

	inline GPUcontainer* produce() const final { return new PinnedBuffer<T>(); }

	inline T* hostPtr() const { return hostptr; }  ///< @return typed host pointer to data
	inline T* devPtr()  const { return devptr; }   ///< @return typed device pointer to data

	inline       T& operator[](int i)       { return hostptr[i]; }  ///< allow array-like bracketed access to HOST data
	inline const T& operator[](int i) const { return hostptr[i]; }

	inline T* begin() { return hostptr; }          /// To support range-based loops
	inline T* end()   { return hostptr + _size; }  /// To support range-based loops


	/**
	 * Copy data from device to host
	 *
	 * @param synchronize if false, the call is fully asynchronous.
	 * if true, host data will be readily available on the call return.
	 */
	inline void downloadFromDevice(cudaStream_t stream, bool synchronize = true)
	{
		// TODO: check if we really need to do that
		// maybe everything is already downloaded
		if (_size > 0) CUDA_Check( cudaMemcpyAsync(hostptr, devptr, sizeof(T) * _size, cudaMemcpyDeviceToHost, stream) );
		if (synchronize) CUDA_Check( cudaStreamSynchronize(stream) );
	}

	/// Copy data from host to device
	inline void uploadToDevice(cudaStream_t stream)
	{
		if (_size > 0) CUDA_Check(cudaMemcpyAsync(devptr, hostptr, sizeof(T) * _size, cudaMemcpyHostToDevice, stream));
	}

	/// Set all the bytes to 0 on both host and device
	inline void clear(cudaStream_t stream)
	{
		clearDevice(stream);
		clearHost();
	}

	/// Set all the bytes to 0 on device only
	inline void clearDevice(cudaStream_t stream)
	{
		if (_size > 0) CUDA_Check( cudaMemsetAsync(devptr, 0, sizeof(T) * _size, stream) );
	}

	/// Set all the bytes to 0 on host only
	inline void clearHost()
	{
		if (_size > 0) memset(hostptr, 0, sizeof(T) * _size);
	}

	/// Copy data from a DeviceBuffer of the same template type
	void copy(const DeviceBuffer<T>& cont, cudaStream_t stream)
	{
		resize_anew(cont.size());
		if (_size > 0) CUDA_Check( cudaMemcpyAsync(devptr, cont.devPtr(), sizeof(T) * _size, cudaMemcpyDeviceToDevice, stream) );
	}

	/// Copy data from a HostBuffer of the same template type
	void copy(const HostBuffer<T>& cont)
	{
		resize_anew(cont.size());
		memcpy(hostptr, cont.hostPtr(), sizeof(T) * _size);
	}

	/// Copy data from a PinnedBuffer of the same template type
	void copy(const PinnedBuffer<T>& cont, cudaStream_t stream)
	{
		resize_anew(cont.size());

		if (_size > 0)
		{
			CUDA_Check( cudaMemcpyAsync(devptr, cont.devPtr(), sizeof(T) * _size, cudaMemcpyDeviceToDevice, stream) );
			memcpy(hostptr, cont.hostPtr(), sizeof(T) * _size);
		}
	}
};


