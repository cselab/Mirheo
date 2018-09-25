#pragma once

#include <string>
#include <vector>
#include <mpi.h>

std::vector<std::string> splitByDelim(std::string str, char delim = ',');

std::string parentPath(std::string path);
std::string relativePath(std::string path);

bool createFoldersCollective(const MPI_Comm& comm, std::string path);
