#pragma once

#include <cuda.h>
#include <cstdlib>
#include <cstdint>
#include <cassert>
#include <type_traits>
#include <utility>
#include <stack>
#include <algorithm>

#include "logger.h"

//==================================================================================================================
// Basic types
//==================================================================================================================

struct Particle
{
	// 4-th coordinate is void
	float x[3];
	int32_t i1;
	float u[3];
	int32_t i2;
};

struct Force
{
	// 4-th coordinate is void
	float f[4];
};


enum ResizeKind
{
	resizeAnew     = 0,
	resizePreserve = 1
};

//==================================================================================================================
// swap functions
//==================================================================================================================

template<typename T> class DeviceBuffer;
template<typename T> class PinnedBuffer;
template<typename T> class HostBuffer;

template<typename T> void containerSwap(DeviceBuffer<T>& a, DeviceBuffer<T>& b);
template<typename T> void containerSwap(HostBuffer<T>& a,   HostBuffer<T>& b);
template<typename T> void containerSwap(PinnedBuffer<T>& a, PinnedBuffer<T>& b);


//==================================================================================================================
// Device Buffer
//==================================================================================================================

template<typename T>
class DeviceBuffer
{
private:
	int capacity, _size;
	T* devptr;
	cudaStream_t stream;
	std::stack<cudaStream_t> streams;

public:
	friend void containerSwap<>(DeviceBuffer<T>&, DeviceBuffer<T>&);

	DeviceBuffer(int n = 0, cudaStream_t stream = 0) :
		capacity(0), _size(0), devptr(nullptr), stream(stream)
	{
		streams.push(stream);
		resize(n, resizeAnew);
	}

	~DeviceBuffer()
	{
		if (devptr != nullptr) CUDA_Check(cudaFree(devptr));
	}

	void pushStream(cudaStream_t stream)
	{
		this->streams.push(stream);
		this->stream = stream;
	}

	void popStream()
	{
		streams.pop();

		if (streams.size() == 0)
			die("Error in stream manipulation");
		stream = streams.top();
	}

	T* 	devPtr() const { return devptr; }
	int	size()   const { return _size; }

	void resize(const int n, ResizeKind kind = resizePreserve)
	{
		T * dold = devptr;
		int oldsize = _size;

		assert(n >= 0);
		_size = n;
		if (capacity >= n) return;

		const int conservative_estimate = (int)ceil(1.1 * n + 10);
		capacity = 128 * ((conservative_estimate + 129) / 128);

		CUDA_Check(cudaMalloc(&devptr, sizeof(T) * capacity));

		if (kind == resizePreserve && dold != nullptr)
		{
			if (oldsize > 0) CUDA_Check(cudaMemcpyAsync(devptr, dold, sizeof(T) * oldsize, cudaMemcpyDeviceToDevice, stream));
		}

		CUDA_Check(cudaFree(dold));
	}

	void clear()
	{
		CUDA_Check( cudaMemsetAsync(devptr, 0, sizeof(T) * _size, stream) );
	}

	template<typename Cont>
	auto copy(const Cont& cont) -> decltype((void)(cont.devPtr()), void())
	{
		static_assert(std::is_same<decltype(devptr), decltype(cont.devPtr())>::value, "can't copy buffers of different types");

		resize(cont.size());
		if (_size > 0) CUDA_Check( cudaMemcpyAsync(devptr, cont.devPtr(), sizeof(T) * _size, cudaMemcpyDeviceToDevice, stream) );
	}

	template<typename Cont>
	auto copy(const Cont& cont) -> decltype((void)(cont.hostPtr()), void())
	{
		static_assert(std::is_same<decltype(devptr), decltype(cont.hostPtr())>::value, "can't copy buffers of different types");

		resize(cont.size());
		if (_size > 0) CUDA_Check( cudaMemcpyAsync(devptr, cont.hostPtr(), sizeof(T) * _size, cudaMemcpyHostToDevice, stream) );
	}
};


//==================================================================================================================
// Pinned Buffer
//==================================================================================================================

template<typename T>
class PinnedBuffer
{
private:
	int capacity, _size;
	T * hostptr, * devptr;
	bool hostChanged, devChanged;
	cudaStream_t stream;
	std::stack<cudaStream_t> streams;

public:
	friend void containerSwap<>(PinnedBuffer<T>&, PinnedBuffer<T>&);

	PinnedBuffer(int n = 0, cudaStream_t stream = 0) :
		capacity(0), _size(0), hostptr(nullptr), devptr(nullptr), hostChanged(false), devChanged(false), stream(stream)
	{
		streams.push(stream);
		resize(n, resizeAnew);
	}

	~PinnedBuffer()
	{
		if (hostptr != nullptr) CUDA_Check(cudaFreeHost(hostptr));
		if (devptr  != nullptr) CUDA_Check(cudaFree(devptr));
	}

	void pushStream(cudaStream_t stream)
	{
		this->streams.push(stream);
		this->stream = stream;
	}

	void popStream()
	{
		streams.pop();

		if (streams.size() == 0)
			die("Error in stream manipulation");
		stream = streams.top();
	}

	T* devPtr()  const { return devptr; }
	T* hostPtr() const { return hostptr; }
	int	size()   const { return _size; }

	T& operator[](int i)
	{
		return hostptr[i];
	}

	void downloadFromDevice(bool synchronize = true)
	{
		// TODO: check if we really need to do that
		// maybe everything is already downloaded
		if (_size > 0) CUDA_Check( cudaMemcpyAsync(hostptr, devptr, sizeof(T) * _size, cudaMemcpyDeviceToHost, stream) );
		if (synchronize) CUDA_Check( cudaStreamSynchronize(stream) );
	}

	void uploadToDevice()
	{
		if (_size > 0) CUDA_Check(cudaMemcpyAsync(devptr, hostptr, sizeof(T) * _size, cudaMemcpyHostToDevice, stream));
	}

	void resize(const int n, ResizeKind kind = resizePreserve)
	{
		T * hold = hostptr;
		T * dold = devptr;
		int oldsize = _size;

		assert(n >= 0);
		_size = n;
		if (capacity >= n) return;

		const int conservative_estimate = (int)ceil(1.1 * n + 10);
		capacity = 128 * ((conservative_estimate + 129) / 128);

		CUDA_Check(cudaHostAlloc(&hostptr, sizeof(T) * capacity, 0));
		CUDA_Check(cudaMalloc(&devptr, sizeof(T) * capacity));

		if (kind == resizePreserve && hold != nullptr)
		{
			memcpy(hostptr, hold, sizeof(T) * oldsize);
			if (oldsize > 0) if (oldsize > 0) CUDA_Check(cudaMemcpyAsync(devptr, dold, sizeof(T) * oldsize, cudaMemcpyDeviceToDevice, stream));
		}

		CUDA_Check(cudaFreeHost(hold));
		CUDA_Check(cudaFree(dold));
	}

	void clear()
	{
		CUDA_Check( cudaMemsetAsync(devptr, 0, sizeof(T) * _size, stream) );
		memset(hostptr, 0, sizeof(T) * _size);
	}

	void clearDevice()
	{
		CUDA_Check( cudaMemsetAsync(devptr, 0, sizeof(T) * _size, stream) );
	}

	template<typename Cont>
	auto copy(const Cont& cont) -> decltype((void)(cont.devPtr()), void())
	{
		static_assert(std::is_same<decltype(devptr), decltype(cont.devPtr())>::value, "can't copy buffers of different types");

		resize(cont.size());
		if (_size > 0) CUDA_Check( cudaMemcpyAsync(devptr, cont.devPtr(), sizeof(T) * _size, cudaMemcpyDeviceToDevice, stream) );
	}

	template<typename Cont>
	auto copy(const Cont& cont) -> decltype((void)(cont.hostPtr()), void())
	{
		static_assert(std::is_same<decltype(hostptr), decltype(cont.hostPtr())>::value, "can't copy buffers of different types");

		resize(cont.size());
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

public:
	friend void containerSwap<>(HostBuffer<T>&, HostBuffer<T>&);

	HostBuffer(int n = 0): capacity(0), _size(0), hostptr(nullptr) { resize(n); }

	~HostBuffer()
	{
		if (hostptr != nullptr) free(hostptr);
	}

	T* 	hostPtr() const { return hostptr; }
	int	size()    const { return _size; }

	T& operator[](int i)
	{
		return hostptr[i];
	}

	void resize(const int n, ResizeKind kind = resizePreserve)
	{
		T * hold = hostptr;
		int oldsize = _size;

		assert(n >= 0);
		_size = n;
		if (capacity >= n) return;

		const int conservative_estimate = (int)ceil(1.1 * n + 10);
		capacity = 128 * ((conservative_estimate + 129) / 128);

		hostptr = (T*) malloc(sizeof(T) * capacity);

		if (kind == resizePreserve && hold != nullptr)
		{
			memcpy(hostptr, hold, sizeof(T) * oldsize);
		}

		free(hold);
	}

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



template<typename T>
void containerSwap(DeviceBuffer<T>& a, DeviceBuffer<T>& b)
{
	std::swap(a.devptr, b.devptr);

	a.resize(b.size(), resizePreserve);
	b.resize(a.size(), resizePreserve);
}

template<typename T>
void containerSwap(HostBuffer<T>& a, HostBuffer<T>& b)
{
	std::swap(a.hostptr, b.hostptr);

	a.resize(b.size(), resizePreserve);
	b.resize(a.size(), resizePreserve);
}

template<typename T>
void containerSwap(PinnedBuffer<T>& a, PinnedBuffer<T>& b)
{
	std::swap(a.devptr,  b.devptr);
	std::swap(a.hostptr, b.hostptr);

	std::swap(a.devChanged,  b.devChanged);
	std::swap(a.hostChanged, b.hostChanged);

	a.resize(b.size(), resizePreserve);
	b.resize(a.size(), resizePreserve);
}

namespace std
{
	template<typename T> void swap(DeviceBuffer<T>& a, DeviceBuffer<T>& b) = delete;
	template<typename T> void swap(HostBuffer<T>& a, HostBuffer<T>& b) = delete;
	template<typename T> void swap(PinnedBuffer<T>& a, PinnedBuffer<T>& b)= delete;
}

