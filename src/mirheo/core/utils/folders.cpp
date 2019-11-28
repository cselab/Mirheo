#include "folders.h"

#include <mirheo/core/logger.h>

#include <sstream>

namespace mirheo
{

std::string getStrZeroPadded(long long i, int zeroPadding)
{
    auto s = std::to_string(i);
    if (zeroPadding < static_cast<int>(s.size()))
        die("Could not create padding for i = %d", i);
    return std::string(zeroPadding - s.length(), '0') + s;
}

std::vector<std::string> splitByDelim(std::string str, char delim)
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

std::string makePath(std::string path)
{
    const size_t n = path.size();
    
    if ( n > 0 && path[n-1] != '/')
        path += '/';

    return path;
}

std::string parentPath(std::string path)
{
    return path.substr(0, path.find_last_of("/"));
}

std::string relativePath(std::string path)
{
    auto pos = path.find_last_of("/");
    if (pos == std::string::npos)
        return path;
    else
        return path.substr(pos + 1);
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

bool createFoldersCollective(const MPI_Comm& comm, std::string path)
{
    bool res;
    int rank;
    constexpr int root = 0;
    MPI_Check( MPI_Comm_rank(comm, &rank) );

    if (rank == root)
        res = createFolders(path);

    MPI_Check( MPI_Bcast(&res, 1, MPI_C_BOOL, root, comm) );

    return res;
}

} // namespace mirheo
