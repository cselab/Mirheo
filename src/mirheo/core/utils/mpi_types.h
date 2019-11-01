#pragma once

#include <mpi.h>

namespace mirheo
{

template <typename T> MPI_Datatype inline getMPIFloatType();
template <> MPI_Datatype inline getMPIFloatType<float> () {return MPI_FLOAT;}
template <> MPI_Datatype inline getMPIFloatType<double>() {return MPI_DOUBLE;}

template <typename T> MPI_Datatype inline getMPIIntType();
template <> MPI_Datatype inline getMPIIntType<int> () {return MPI_INT;}
template <> MPI_Datatype inline getMPIIntType<unsigned long long> () {return MPI_UNSIGNED_LONG_LONG;}

} // namespace mirheo
