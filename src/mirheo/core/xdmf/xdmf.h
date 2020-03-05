#pragma once

#include "grids.h"

#include <mirheo/core/pvs/rigid_object_vector.h>

#include <memory>
#include <mpi.h>
#include <string>
#include <vector>

namespace mirheo
{

/// namespace for all functions related to I/O with XDMF + hdf5
namespace XDMF
{
/** \brief Dump a set of channels with associated geometry in hdf5+xmf format
    \param filename Base file name (without extension); two files will be created: xmf and hdf5
    \param grid The geometry description of the data. See \c Grid.
    \param channels A list of channel descriptions and associated data to dump
    \param time A time stamp, useful when dumping sequences of files
    \param comm MPI communicator shared by all ranks containing the data (simulation OR postprocess ranks)
 */
void write(const std::string& filename, const Grid *grid,
           const std::vector<Channel>& channels, MirState::TimeType time, MPI_Comm comm);

/// \overload write()
void write(const std::string& filename, const Grid *grid,
           const std::vector<Channel>& channels, MPI_Comm comm);

/** \brief the data read by readVertexData()

    Represents particles data
 */
struct VertexChannelsData
{
    std::vector<real3> positions;        ///< the position of the particles (in global coordinates)
    std::vector<Channel> descriptions;   ///< metadata associated to each channel
    std::vector<std::vector<char>> data; ///< channel data
};

/** \brief Read particle data from a pair of xmf+hdf5 files
    \param filename the xdmf file name (with extension)
    \param comm The communicator used in the I/O process
    \param chunkSize The smallest piece that processors can split
    \return The read data (on the local rank)
 */
VertexChannelsData readVertexData(const std::string& filename, MPI_Comm comm, int chunkSize);

} // namespace XDMF

} // namespace mirheo
