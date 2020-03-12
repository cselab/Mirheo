#pragma once

#include <mpi.h>

namespace mirheo
{

/// convenience function to get the MPI_Datatype of a real number corresponding to the given precision
template <typename T> MPI_Datatype inline getMPIFloatType();
template <> MPI_Datatype inline getMPIFloatType<float> () {return MPI_FLOAT;}  ///< single precision floating point
template <> MPI_Datatype inline getMPIFloatType<double>() {return MPI_DOUBLE;} ///< double precision floating point

/// convenience function to get the MPI_Datatype of an integer corresponding to the given precision
template <typename T> MPI_Datatype inline getMPIIntType();
template <> MPI_Datatype inline getMPIIntType<int> () {return MPI_INT;}  ///< single precision integer
template <> MPI_Datatype inline getMPIIntType<unsigned long long> () {return MPI_UNSIGNED_LONG_LONG;} ///< double precision unsigned integer

} // namespace mirheo
