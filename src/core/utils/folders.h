#pragma once

#include <mpi.h>
#include <string>
#include <vector>

std::string getStrZeroPadded(int i, int zeroPadding = 5);

std::vector<std::string> splitByDelim(std::string str, char delim = ',');

std::string makePath    (std::string path); ///< append '/' at the end of 'path' if needed
std::string parentPath  (std::string path);
std::string relativePath(std::string path);

bool createFoldersCollective(const MPI_Comm& comm, std::string path);
