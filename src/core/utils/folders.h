#pragma once

#include <string>
#include <vector>
#include <sstream>

#include <core/logger.h>
#include <mpi.h>

static std::vector<std::string> splitByDelim(std::string str, char delim = ',')
{
    std::stringstream sstream(str);
    std::string word;
    std::vector<std::string> splitted;

    while(std::getline(sstream, word, delim))
    {
        splitted.push_back(word);
    }

    return splitted;
}

static std::string parentPath(std::string path)
{
    return path.substr(0, path.find_last_of("/"));
}

static bool createFolders(std::string path)
{
    std::string command = "mkdir -p " + path;
    if ( system(command.c_str()) != 0 )
    {
        error("Could not create folders or files by given path '%s'", path.c_str());
        return false;
    }

    return true;
}

static bool createFoldersCollective(const MPI_Comm& comm, std::string path)
{
    bool res;
    int rank;
    MPI_Check( MPI_Comm_rank(comm, &rank) );

    if (rank == 0)
        res = createFolders(path);

    MPI_Bcast(&res, 1, MPI_C_BOOL, 0, comm);

    return res;
}
