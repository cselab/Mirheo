#pragma once

#include <cassert>
#include <type_traits>
#include <utility>
#include <algorithm>

enum class ResizeKind
{
	resizeAnew,
	resizePreserve
};

//==================================================================================================================
// swap functions
//==================================================================================================================

template<typename T> class DeviceBuffer;
template<typename T> class PinnedBuffer;
template<typename T> class HostBuffer;

template<typename T> void containerSwap(DeviceBuffer<T>& a, DeviceBuffer<T>& b, cudaStream_t stream);
template<typename T> void containerSwap(HostBuffer  <T>& a, HostBuffer  <T>& b, cudaStream_t stream);
template<typename T> void containerSwap(PinnedBuffer<T>& a, PinnedBuffer<T>& b, cudaStream_t stream);


//==================================================================================================================
// Device Buffer
//==================================================================================================================

template<typename T>
class DeviceBuffer
{
private:
	int capacity, _size;
	T* devptr;

public:
	friend void containerSwap<>(DeviceBuffer<T>&, DeviceBuffer<T>&, cudaStream_t stream);

	DeviceBuffer(int n = 0, cudaStream_t stream = 0) :
		capacity(0), _size(0), devptr(nullptr)
	{
		resize(n, stream, ResizeKind::resizeAnew);
	}

	~DeviceBuffer()
	{
		if (devptr != nullptr) CUDA_Check(cudaFree(devptr));
	}

	__host__ __device__ T* 	devPtr() const { return devptr; }
	__host__ __device__ int	size()   const { return _size; }

	__device__ inline T& operator[](int i)
	{
		return devptr[i];
	}

	__device__ inline const T& operator[](int i) const
	{
		return devptr[i];
	}

	void resize(const int n, cudaStream_t stream, ResizeKind kind = ResizeKind::resizePreserve)
	{
		T * dold = devptr;
		int oldsize = _size;

		assert(n >= 0);
		_size = n;
		if (capacity >= n) return;

		const int conservative_estimate = (int)ceil(1.1 * n + 10);
		capacity = 128 * ((conservative_estimate + 129) / 128);

		CUDA_Check(cudaMalloc(&devptr, sizeof(T) * capacity));

		if (kind == ResizeKind::resizePreserve && dold != nullptr)
		{
			if (oldsize > 0) CUDA_Check(cudaMemcpyAsync(devptr, dold, sizeof(T) * oldsize, cudaMemcpyDeviceToDevice, stream));
		}

		CUDA_Check(cudaFree(dold));
	}

	void clear(cudaStream_t stream)
	{
		CUDA_Check( cudaMemsetAsync(devptr, 0, sizeof(T) * _size, stream) );
	}

	template<typename Cont>
	auto copy(const Cont& cont, cudaStream_t stream) -> decltype((void)(cont.devPtr()), void())
	{
		static_assert(std::is_same<decltype(devptr), decltype(cont.devPtr())>::value, "can't copy buffers of different types");

		resize(cont.size(), stream, ResizeKind::resizeAnew);
		if (_size > 0) CUDA_Check( cudaMemcpyAsync(devptr, cont.devPtr(), sizeof(T) * _size, cudaMemcpyDeviceToDevice, stream) );
	}

	template<typename Cont>
	auto copy(const Cont& cont, cudaStream_t stream) -> decltype((void)(cont.hostPtr()), void())
	{
		static_assert(std::is_same<decltype(devptr), decltype(cont.hostPtr())>::value, "can't copy buffers of different types");

		resize(cont.size(), stream, ResizeKind::resizeAnew);
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

public:
	friend void containerSwap<>(PinnedBuffer<T>&, PinnedBuffer<T>&, cudaStream_t stream);

	PinnedBuffer(int n = 0, cudaStream_t stream = 0) :
		capacity(0), _size(0), hostptr(nullptr), devptr(nullptr), hostChanged(false), devChanged(false)
	{
		resize(n, stream, ResizeKind::resizeAnew);
	}

	~PinnedBuffer()
	{
		if (hostptr != nullptr) CUDA_Check(cudaFreeHost(hostptr));
		if (devptr  != nullptr) CUDA_Check(cudaFree(devptr));
	}

	__host__ __device__ inline T*  devPtr()  const { return devptr; }
	__host__ __device__ inline T*  hostPtr() const { return hostptr; }
	__host__ __device__ inline int size()    const { return _size; }

	__host__ __device__ inline T& operator[](int i)
	{
#ifdef __CUDA_ARCH__
		return devptr[i];
#else
		return hostptr[i];
#endif
	}

	__host__ __device__ inline const T& operator[](int i) const
	{
#ifdef __CUDA_ARCH__
		return devptr[i];
#else
		return hostptr[i];
#endif
	}

	void downloadFromDevice(cudaStream_t stream, bool synchronize = true)
	{
		// TODO: check if we really need to do that
		// maybe everything is already downloaded
		if (_size > 0) CUDA_Check( cudaMemcpyAsync(hostptr, devptr, sizeof(T) * _size, cudaMemcpyDeviceToHost, stream) );
		if (synchronize) CUDA_Check( cudaStreamSynchronize(stream) );
	}

	void uploadToDevice(cudaStream_t stream)
	{
		if (_size > 0) CUDA_Check(cudaMemcpyAsync(devptr, hostptr, sizeof(T) * _size, cudaMemcpyHostToDevice, stream));
	}

	void resize(const int n, cudaStream_t stream, ResizeKind kind = ResizeKind::resizePreserve)
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

		if (kind == ResizeKind::resizePreserve && hold != nullptr)
		{
			if (oldsize > 0) memcpy(hostptr, hold, sizeof(T) * oldsize);
			if (oldsize > 0) CUDA_Check(cudaMemcpyAsync(devptr, dold, sizeof(T) * oldsize, cudaMemcpyDeviceToDevice, stream));
		}

		CUDA_Check(cudaFreeHost(hold));
		CUDA_Check(cudaFree(dold));
	}

	void clear(cudaStream_t stream)
	{
		CUDA_Check( cudaMemsetAsync(devptr, 0, sizeof(T) * _size, stream) );
		memset(hostptr, 0, sizeof(T) * _size);
	}

	void clearDevice(cudaStream_t stream)
	{
		CUDA_Check( cudaMemsetAsync(devptr, 0, sizeof(T) * _size, stream) );
	}

	template<typename Cont>
	auto copy(const Cont& cont, cudaStream_t stream) -> decltype((void)(cont.devPtr()), void())
	{
		static_assert(std::is_same<decltype(devptr), decltype(cont.devPtr())>::value, "can't copy buffers of different types");

		resize(cont.size(), stream, ResizeKind::resizeAnew);
		if (_size > 0) CUDA_Check( cudaMemcpyAsync(devptr, cont.devPtr(), sizeof(T) * _size, cudaMemcpyDeviceToDevice, stream) );
	}

	template<typename Cont>
	auto copy(const Cont& cont) -> decltype((void)(cont.hostPtr()), void())
	{
		static_assert(std::is_same<decltype(hostptr), decltype(cont.hostPtr())>::value, "can't copy buffers of different types");

		resize(cont.size(), ResizeKind::resizeAnew);
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
	friend void containerSwap<>(HostBuffer<T>&, HostBuffer<T>&, cudaStream_t stream);

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

	const T& operator[](int i) const
	{
		return hostptr[i];
	}

	void resize(const int n, ResizeKind kind = ResizeKind::resizePreserve)
	{
		T * hold = hostptr;
		int oldsize = _size;

		assert(n >= 0);
		_size = n;
		if (capacity >= n) return;

		const int conservative_estimate = (int)ceil(1.1 * n + 10);
		capacity = 128 * ((conservative_estimate + 129) / 128);

		hostptr = (T*) malloc(sizeof(T) * capacity);

		if (kind == ResizeKind::resizePreserve && hold != nullptr)
		{
			if (oldsize > 0) memcpy(hostptr, hold, sizeof(T) * oldsize);
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
void containerSwap(DeviceBuffer<T>& a, DeviceBuffer<T>& b, cudaStream_t stream)
{
	std::swap(a.devptr, b.devptr);

	a.resize(b.size(), stream, ResizeKind::resizePreserve);
	b.resize(a.size(), stream, ResizeKind::resizePreserve);
}

template<typename T>
void containerSwap(HostBuffer<T>& a, HostBuffer<T>& b, cudaStream_t stream)
{
	std::swap(a.hostptr, b.hostptr);

	a.resize(b.size(), ResizeKind::resizePreserve);
	b.resize(a.size(), ResizeKind::resizePreserve);
}

template<typename T>
void containerSwap(PinnedBuffer<T>& a, PinnedBuffer<T>& b, cudaStream_t stream)
{
	std::swap(a.devptr,  b.devptr);
	std::swap(a.hostptr, b.hostptr);

	std::swap(a.devChanged,  b.devChanged);
	std::swap(a.hostChanged, b.hostChanged);

	a.resize(b.size(), stream, ResizeKind::resizePreserve);
	b.resize(a.size(), stream, ResizeKind::resizePreserve);
}

namespace std
{
	template<typename T> void swap(DeviceBuffer<T>& a, DeviceBuffer<T>& b) = delete;
	template<typename T> void swap(HostBuffer<T>& a, HostBuffer<T>& b) = delete;
	template<typename T> void swap(PinnedBuffer<T>& a, PinnedBuffer<T>& b)= delete;
}
