#pragma once

#include <string>
#include <vector>

#include <mpi.h>
#include <hdf5.h>

#include "grids.h"

namespace XDMF
{
    namespace HDF5
    {
        hid_t create(std::string filename, MPI_Comm comm);
        void writeDataSet(hid_t file_id, const Grid* grid, const Channel& channel);
        void writeData   (hid_t file_id, const Grid* grid, const std::vector<Channel>& channels);
        void close       (hid_t file_id);
    
        void write(std::string filename, MPI_Comm comm, const Grid* grid, const std::vector<Channel>& channels);
    }
}
