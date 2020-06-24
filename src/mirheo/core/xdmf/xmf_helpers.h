#pragma once

#include "grids.h"
#include <mirheo/core/mirheo_state.h>

#include <extern/pugixml/src/pugixml.hpp>

#include <mpi.h>
#include <string>
#include <vector>


namespace mirheo
{

namespace XDMF
{
namespace XMF
{

void writeDataSet(pugi::xml_node node, const std::string& h5filename,
                  const Grid *grid, const Channel& channel);
void writeData   (pugi::xml_node node, const std::string& h5filename, const Grid *grid,
                  const std::vector<Channel>& channels);
void write(const std::string& filename, const std::string& h5filename, MPI_Comm comm,
           const Grid *grid, const std::vector<Channel>& channels, MirState::TimeType time);

std::tuple<std::string /*h5filename*/, std::vector<Channel>>
read(const std::string& filename, MPI_Comm comm, Grid *grid);

} // namespace XMF
} // namespace XDMF
} // namespace mirheo
