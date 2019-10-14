#pragma once

template<typename T>
struct CollisionTable
{
    const int maxSize;
    int* total;
    T* indices;

    __device__ void push_back(T idx)
    {
        int i = atomicAdd(total, 1);
        if (i < maxSize) indices[i] = idx;
    }
};


template<typename T>
__device__ static inline T fmin_vec(T v)
{
    return v;
}

template<typename T, typename... Args>
__device__ static inline T fmin_vec(T v, Args... args)
{
    return fminf(v, fmin_vec(args...));
}

template<typename T>
__device__ static inline T fmax_vec(T v)
{
    return v;
}

template<typename T, typename... Args>
__device__ static inline T fmax_vec(T v, Args... args)
{
    return fmaxf(v, fmax_vec(args...));
}
