#pragma once

#include <cuda.h>
#include <cstdlib>
#include <cstdint>
#include <cassert>
#include <type_traits>
#include <utility>

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
// Device Buffer
//==================================================================================================================

template<typename T>
class DeviceBuffer
{
private:
	int capacity, _size;
	T* devptr;
	cudaStream_t stream;

public:
	DeviceBuffer(int n = 0, cudaStream_t stream = 0) : capacity(0), _size(0), devptr(nullptr), stream(stream)
	{
		resize(n, resizeAnew);
	}

	~DeviceBuffer()
	{
		if (devptr != nullptr) CUDA_Check(cudaFree(devptr));
	}

	T* 		 devPtr() 			 { return devptr; }
	const T* constDevPtr() const { return devptr; }
	int		 size()				 { return _size; }

	void resize(const int n, ResizeKind kind = resizeAnew, cudaStream_t stream = 0)
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
			CUDA_Check(cudaMemcpyAsync(devptr, dold, sizeof(T) * oldsize, cudaMemcpyDeviceToDevice, stream));
		}

		CUDA_Check(cudaFree(dold));
	}

	void clear()
	{
		CUDA_Check( cudaMemsetAsync(devptr, 0, sizeof(T) * _size, stream) );
	}

	template<typename Cont>
	auto copy(Cont& cont) -> decltype((void)(cont.devptr), void())
	{
		static_assert(std::is_same<decltype(devptr), decltype(cont.devptr)>::value, "can't copy buffers of different types");

		resize(cont.size);
		CUDA_Check( cudaMemcpyAsync(devptr, cont.devptr, sizeof(T) * _size, cudaMemcpyDeviceToDevice, stream) );
	}

	template<typename Cont>
	auto copy(Cont& cont) -> decltype((void)(cont.hostptr), void())
	{
		static_assert(std::is_same<decltype(devptr), decltype(cont.hostptr)>::value, "can't copy buffers of different types");

		resize(cont.size);
		CUDA_Check( cudaMemcpyAsync(devptr, cont.hostptr, sizeof(T) * _size, cudaMemcpyHostToDevice, stream) );
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

private:
	void syncHost()
	{
		CUDA_Check( cudaMemcpyAsync(hostptr, devptr, sizeof(T) * _size, cudaMemcpyDeviceToHost, stream) );
		CUDA_Check( cudaStreamSynchronize(stream) );
	}

	void syncDev()
	{
		CUDA_Check(cudaMemcpyAsync(devptr, hostptr, sizeof(T) * _size, cudaMemcpyHostToDevice, stream));
	}

public:
	PinnedBuffer(int n = 0, cudaStream_t stream = 0) :
		capacity(0), _size(0), hostptr(nullptr), devptr(nullptr), hostChanged(false), devChanged(false), stream(stream)
	{
		resize(n, resizeAnew);
	}

	~PinnedBuffer()
	{
		if (hostptr != nullptr) CUDA_Check(cudaFreeHost(hostptr));
	}

	T* devPtr()
	{
		if (hostChanged) syncDev();
		devChanged = true;
		return devptr;
	}
	const T* constDevPtr()
	{
		if (hostChanged) syncDev();
		return devptr;
	}

	T* hostPtr()
	{
		if (devChanged) syncHost();
		hostChanged = true;
		return hostptr;
	}
	const T* constHostPtr()
	{
		if (devChanged) syncHost();
		hostChanged = true;
		return hostptr;
	}

	int	size() const
	{
		return _size;
	}

	void resize(const int n, ResizeKind kind = resizeAnew, cudaStream_t stream = 0)
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
			CUDA_Check(cudaMemcpyAsync(devptr, dold, sizeof(T) * oldsize, cudaMemcpyDeviceToDevice, stream));
		}

		CUDA_Check(cudaFreeHost(hold));
		CUDA_Check(cudaFree(dold));
	}

	void clear(cudaStream_t stream = 0)
	{
		CUDA_Check( cudaMemsetAsync(devptr, 0, sizeof(T) * _size, stream) );
		memset(hostptr, 0, sizeof(T) * _size);

		hostChanged = devChanged = false;
	}

	template<typename Cont>
	auto copy(Cont& cont, cudaStream_t stream = 0) -> decltype((void)(cont.devptr), void())
	{
		static_assert(std::is_same<decltype(devptr), decltype(cont.devptr)>::value, "can't copy buffers of different types");

		resize(cont.size);
		CUDA_Check( cudaMemcpyAsync(devptr, cont.devptr, sizeof(T) * _size, cudaMemcpyDeviceToDevice, stream) );
	}

	template<typename Cont>
	auto copy(Cont& cont, cudaStream_t stream = 0) -> decltype((void)(cont.hostptr), void())
	{
		static_assert(std::is_same<decltype(hostptr), decltype(cont.hostptr)>::value, "can't copy buffers of different types");

		resize(cont.size);
		memcpy(hostptr, cont.hostptr, sizeof(T) * _size);
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
	HostBuffer(int n = 0): capacity(0), _size(0), hostptr(nullptr) { resize(n); }

	~HostBuffer()
	{
		if (hostptr != nullptr) free(hostptr);
	}

	T* 		 hostPtr() 				{ return hostptr; }
	const T* constHostPtr() const	{ return hostptr; }
	int		 size()					{ return _size; }

	void resize(const int n, ResizeKind kind = resizeAnew)
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
	auto copy(Cont& cont, cudaStream_t stream = 0) -> decltype((void)(cont.hostptr), void())
	{
		static_assert(std::is_same<decltype(hostptr), decltype(cont.hostptr)>::value, "can't copy buffers of different types");

		resize(cont.size);
		memcpy(hostptr, cont.hostptr, sizeof(T) * _size);
	}

	template<typename Cont>
	auto copy(Cont& cont, cudaStream_t stream = 0) -> decltype((void)(cont.devptr), void())
	{
		static_assert(std::is_same<decltype(hostptr), decltype(cont.devptr)>::value, "can't copy buffers of different types");

		resize(cont.size);
		CUDA_Check( cudaMemcpyAsync(hostptr, cont.devptr, sizeof(T) * _size, cudaMemcpyDeviceToHost, stream) );
	}
};


template<typename T>
void swap(DeviceBuffer<T>& a, DeviceBuffer<T>& b, cudaStream_t stream = 0)
{
	a.resize(b.size, resizePreserve, stream);
	b.resize(a.size, resizePreserve, stream);

	std::swap(a.devptr, b.devptr);
}

template<typename T>
void swap(HostBuffer<T>& a, HostBuffer<T>& b, cudaStream_t stream = 0)
{
	a.resize(b.size, resizePreserve, stream);
	b.resize(a.size, resizePreserve, stream);

	std::swap(a.hostptr, b.hostptr);
}

template<typename T>
void swap(PinnedBuffer<T>& a, PinnedBuffer<T>& b, cudaStream_t stream = 0)
{
	a.resize(b.size, resizePreserve, stream);
	b.resize(a.size, resizePreserve, stream);

	std::swap(a.devptr,  b.devptr);
	std::swap(a.hostptr, b.hostptr);
}









