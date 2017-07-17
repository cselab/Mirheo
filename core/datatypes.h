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

struct __align__(16) Particle
{
	// We're targeting little-endian systems here, note that!

	// Free particles will have their id in i1 (or in s21*2^32 + i1)
	// Object particles will have their id (in object) in s21 and object id in i1
	// s22 is arbitrary

	float3 r;
	union
	{
		int32_t i1;
		struct { int16_t s11 /*least significant*/, s12; };
	};

	float3 u;
	union
	{
		int32_t i2;
		struct { int16_t s21 /*least significant*/, s22; };
	};

	__host__ __device__ inline Particle() {};
	__host__ __device__ inline Particle(const float4 r4, const float4 u4)
	{
		r = make_float3(r4.x, r4.y, r4.z);
		u = make_float3(u4.x, u4.y, u4.z);

#ifdef __CUDA_ARCH__
		i1 = __float_as_int(r4.w);
		i2 = __float_as_int(u4.w);
#else
		union {int i; float f;} u;
		u.f = r4.w;
		i1 = u.i;

		u.f = u4.w;
		i2 = u.i;
#endif
	}
};

struct __align__(16) Float3_int
{
	float3 v;
	union
	{
		int32_t i;
		struct { int16_t s1, s2; };
	};

	__host__ __device__ inline Float3_int() {};
	__host__ __device__ inline Float3_int(const float3 v, int i) : v(v), i(i) {};

	__host__ __device__ inline Float3_int(const float4 f4)
	{
		v = make_float3(f4.x, f4.y, f4.z);

#ifdef __CUDA_ARCH__
		i = __float_as_int(f4.w);
#else
		union {int i; float f;} u;
		u.f = f4.w;
		i = u.i;
#endif
	}


	__host__ __device__ inline float4 toFloat4()
	{
		float f;

#ifdef __CUDA_ARCH__
		f = __int_as_float(i);
#else
		union {int i; float f;} u;
		u.i = i;
		f = u.f;
#endif

		return make_float4(v.x, v.y, v.z, f);
	}
};

struct __align__(16) Force
{
	float3 f;
	int32_t i1;
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
template<typename T> void containerSwap(HostBuffer  <T>& a, HostBuffer  <T>& b);
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
		if (streams.size() > 1) streams.pop();
		stream = streams.top();
	}

	__host__ __device__ T* 	devPtr() const { return devptr; }
	__host__ __device__ int	size()   const { return _size; }

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
		if (streams.size() > 1) streams.pop();
		stream = streams.top();
	}

	__host__ __device__ T* devPtr()  const { return devptr; }
	__host__            T* hostPtr() const { return hostptr; }
	__host__ __device__ int	size()   const { return _size; }

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

