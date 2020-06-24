// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

namespace mirheo
{

/**
   \brief Holds a set of collision information
   \tparam T Information type of one collision
 */
template<typename T>
struct CollisionTable
{
    const int maxSize; ///< the maximum number of collisions
    int* total;        ///< the current number of collisions registered
    T* indices;        ///< information holding the registered collisions

    /** \brief register a collision to the table
        \param [in] idx The information about the collision
     */
    __device__ void push_back(T idx)
    {
        const int i = atomicAdd(total, 1);
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
    return math::min(v, fmin_vec(args...));
}

template<typename T>
__device__ static inline T fmax_vec(T v)
{
    return v;
}

template<typename T, typename... Args>
__device__ static inline T fmax_vec(T v, Args... args)
{
    return math::max(v, fmax_vec(args...));
}

} // namespace mirheo
