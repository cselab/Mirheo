#pragma once

#include <string>
#include <vector>

#include <mpi.h>
#include <extern/pugixml/src/pugixml.hpp>

#include "grids.h"

namespace XDMF
{
namespace XMF
{        

void writeDataSet(pugi::xml_node node, std::string h5filename, const Grid* grid, const Channel& channel);
void writeData   (pugi::xml_node node, std::string h5filename, const Grid* grid, const std::vector<Channel>& channels);
void write(std::string filename, std::string h5filename, MPI_Comm comm, const Grid* grid, const std::vector<Channel>& channels, float time);

void read(std::string filename, MPI_Comm comm, std::string &h5filename, Grid *grid, std::vector<Channel> &channels);

} // namespace XMF
} // namespace XDMF
