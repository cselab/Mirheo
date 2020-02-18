#pragma once

#include <mpi.h>
#include <string>
#include <vector>

namespace mirheo
{

std::string getStrZeroPadded(long long i, int zeroPadding = 5);

std::vector<std::string> splitByDelim(std::string str, char delim = ',');

std::string makePath    (std::string path); ///< append '/' at the end of 'path' if needed
std::string parentPath  (std::string path);
std::string relativePath(std::string path);

/// Concat two paths A and B. Adds a '/' between A and B if A is non-empty and
/// if it doesn't already end in '/'.
std::string joinPaths(const std::string &A, const std::string &B);

bool createFoldersCollective(const MPI_Comm& comm, std::string path);

} // namespace mirheo
