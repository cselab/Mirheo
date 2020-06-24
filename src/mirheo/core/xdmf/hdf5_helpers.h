// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <string>
#include <vector>

#include <mpi.h>
#include <hdf5.h>

#include "grids.h"

namespace mirheo
{

namespace XDMF
{
namespace HDF5
{

hid_t create      (const std::string& filename, MPI_Comm comm);
hid_t openReadOnly(const std::string& filename, MPI_Comm comm);

void writeDataSet(hid_t file_id, const GridDims *gridDims, const Channel& channel);
void writeData   (hid_t file_id, const GridDims *gridDims, const std::vector<Channel>& channels);

void readDataSet (hid_t file_id, const GridDims *gridDims, Channel& channel);
void readData    (hid_t file_id, const GridDims *gridDims, std::vector<Channel>& channels);

void close       (hid_t file_id);


void write(const std::string& filename, MPI_Comm comm, const Grid *grid, const std::vector<Channel>& channels);
void read (const std::string& filename, MPI_Comm comm, Grid *grid, std::vector<Channel>& channels);

} // namespace HDF5
} // namespace XDMF

} // namespace mirheo
