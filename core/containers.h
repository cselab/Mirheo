#pragma once

#include <core/logger.h>

#include <cassert>
#include <type_traits>
#include <utility>
#include <algorithm>
#include <typeinfo>


class GPUcontainer
{
public:
	virtual int size() const = 0;
	virtual int datatype_size() const = 0;

	virtual void* genericDevPtr() const = 0;

	virtual void resize     (const int n, cudaStream_t stream) = 0;
	virtual void resize_anew(const int n)                      = 0;

	virtual ~GPUcontainer() = default;
};

//==================================================================================================================
// Device Buffer
//==================================================================================================================

template<typename T>
class DeviceBuffer : public GPUcontainer
{
private:
	int capacity, _size;
	T* devptr;

	void _resize(const int n, cudaStream_t stream, bool copy)
	{
		T * dold = devptr;
		int oldsize = _size;

		assert(n >= 0);
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

	// For std::swap
	DeviceBuffer (DeviceBuffer&& b)
	{
		*this = std::move(b);
	}

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

	~DeviceBuffer()
	{
		if (devptr != nullptr)
		{
			CUDA_Check(cudaFree(devptr));
			debug4("Destroying DeviceBuffer<%s>", typeid(T).name());
		}
	}

	// Override section
	inline int datatype_size() const override final { return sizeof(T); }
	inline int size()          const override final { return _size; }

	inline void* genericDevPtr() const override final { return (void*) devPtr(); }

	inline void resize     (const int n, cudaStream_t stream) override final { _resize(n, stream, true);  }
	inline void resize_anew(const int n)                      override final { _resize(n, 0,      false); }

	// Other methods
	inline T* devPtr() const { return devptr; }

	inline void clear(cudaStream_t stream)
	{
		if (_size > 0) CUDA_Check( cudaMemsetAsync(devptr, 0, sizeof(T) * _size, stream) );
	}

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
// Pinned Buffer
//==================================================================================================================

template<typename T>
class PinnedBuffer : public GPUcontainer
{
private:
	int capacity, _size;
	T * hostptr, * devptr;

	void _resize(const int n, cudaStream_t stream, bool copy)
	{
		T * hold = hostptr;
		T * dold = devptr;
		int oldsize = _size;

		assert(n >= 0);
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

	// For std::swap
	PinnedBuffer (PinnedBuffer&& b)
	{
		*this = std::move(b);
	}

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

	~PinnedBuffer()
	{
		if (devptr != nullptr)
		{
			CUDA_Check(cudaFreeHost(hostptr));
			CUDA_Check(cudaFree(devptr));
			debug4("Destroying PinnedBuffer<%s>", typeid(T).name());
		}
	}

	// Override section
	inline int datatype_size() const override final { return sizeof(T); }
	inline int size()          const override final { return _size; }

	inline void* genericDevPtr() const override final { return (void*) devPtr(); }

	inline void resize     (const int n, cudaStream_t stream) override final { _resize(n, stream, true);  }
	inline void resize_anew(const int n)                      override final { _resize(n, 0,      false); }


	// Other methods
	inline T* devPtr()  const { return devptr; }
	inline T* hostPtr() const { return hostptr; }

	inline       T& operator[](int i)       { return hostptr[i]; }
	inline const T& operator[](int i) const { return hostptr[i]; }

	inline void downloadFromDevice(cudaStream_t stream, bool synchronize = true)
	{
		// TODO: check if we really need to do that
		// maybe everything is already downloaded
		if (_size > 0) CUDA_Check( cudaMemcpyAsync(hostptr, devptr, sizeof(T) * _size, cudaMemcpyDeviceToHost, stream) );
		if (synchronize) CUDA_Check( cudaStreamSynchronize(stream) );
	}

	inline void uploadToDevice(cudaStream_t stream)
	{
		if (_size > 0) CUDA_Check(cudaMemcpyAsync(devptr, hostptr, sizeof(T) * _size, cudaMemcpyHostToDevice, stream));
	}

	inline void clear(cudaStream_t stream)
	{
		clearDevice(stream);
		clearHost();
	}

	inline void clearDevice(cudaStream_t stream)
	{
		if (_size > 0) CUDA_Check( cudaMemsetAsync(devptr, 0, sizeof(T) * _size, stream) );
	}

	inline void clearHost()
	{
		if (_size > 0) memset(hostptr, 0, sizeof(T) * _size);
	}

	template<typename Cont>
	auto copy(const Cont& cont, cudaStream_t stream) -> decltype((void)(cont.devPtr()), void())
	{
		static_assert(std::is_same<decltype(devptr), decltype(cont.devPtr())>::value, "can't copy buffers of different types");

		resize_anew(cont.size());
		if (_size > 0) CUDA_Check( cudaMemcpyAsync(devptr, cont.devPtr(), sizeof(T) * _size, cudaMemcpyDeviceToDevice, stream) );
	}

	template<typename Cont>
	auto copy(const Cont& cont) -> decltype((void)(cont.hostPtr()), void())
	{
		static_assert(std::is_same<decltype(hostptr), decltype(cont.hostPtr())>::value, "can't copy buffers of different types");

		resize_anew(cont.size());
		memcpy(hostptr, cont.hostPtr(), sizeof(T) * _size);
	}
};


//==================================================================================================================
// Host Buffer
//==================================================================================================================

template<typename T>
class HostBuffer
{
private:
	int capacity, _size;
	T * hostptr;

	void _resize(const int n, bool copy)
	{
		T * hold = hostptr;
		int oldsize = _size;

		assert(n >= 0);
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

	// For std::swap
	HostBuffer(HostBuffer&& b)
	{
		*this = std::move(b);
	}

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

	void clear()
	{
		memset(hostptr, 0, sizeof(T) * _size);
	}

	template<typename Cont>
	auto copy(const Cont& cont) -> decltype((void)(cont.hostPtr()), void())
	{
		static_assert(std::is_same<decltype(hostptr), decltype(cont.hostPtr())>::value, "can't copy buffers of different types");

		resize(cont.size());
		memcpy(hostptr, cont.hostPtr(), sizeof(T) * _size);
	}

	template<typename Cont>
	auto copy(const Cont& cont, cudaStream_t stream) -> decltype((void)(cont.devPtr()), void())
	{
		static_assert(std::is_same<decltype(hostptr), decltype(cont.devPtr())>::value, "can't copy buffers of different types");

		resize(cont.size());
		if (_size > 0) CUDA_Check( cudaMemcpyAsync(hostptr, cont.devPtr(), sizeof(T) * _size, cudaMemcpyDeviceToHost, stream) );
	}
};

