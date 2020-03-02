#pragma once

#include <mirheo/core/datatypes.h>

#include <string>
#include <tuple>
#include <vector>

namespace mirheo
{

/** Read a file in .off format (ASCII) representing a triangle mesh into vertex coordinates and faces connectivity.
    This method will die if the found is not found or if the internal structure is not correct.
 */
std::tuple<std::vector<real3>, std::vector<int3>> readOff(const std::string& fileName);

/** Dump a triangle mesh with face connectivity into the file named `fileName`.
    This method writes the mesh in .off ASCII format.
    This method will die if the filename can not be written.
 */
void writeOff(const std::vector<real3>& vertices, const std::vector<int3>& faces, const std::string& fileName);

} // namespace mirheo
