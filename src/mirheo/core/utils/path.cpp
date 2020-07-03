// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "path.h"

#include <mirheo/core/logger.h>

#include <sstream>

namespace mirheo
{

std::string createStrZeroPadded(long long i, int zeroPadding)
{
    auto s = std::to_string(i);
    if (zeroPadding < static_cast<int>(s.size()))
        die("Could not create padding for i = %lld", i);
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

std::string getParentPath(std::string path)
{
    auto lastSepPos = path.find_last_of("/");
    if (lastSepPos == std::string::npos)
        return "";
    return makePath(path.substr(0, lastSepPos));
}

std::string getBaseName(std::string path)
{
    auto pos = path.find_last_of("/");
    if (pos == std::string::npos)
        return path;
    else
        return path.substr(pos + 1);
}

std::string joinPaths(const std::string &A, const std::string &B) {
    std::string out;
    out.reserve(A.size() + B.size() + 1);
    if (!A.empty()) {
        out += A;
        if (A.back() != '/')
            out += '/';
    }
    out += B;
    return out;
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

std::string setExtensionOrDie(std::string path, const std::string ext)
{
    std::string fname = getBaseName(path);
    const std::string parentPath = getParentPath(path);

    auto lastDotPos = path.find_last_of(".");

    if (lastDotPos != std::string::npos) {
        const std::string currentExt(path.begin() + lastDotPos + 1, path.end());
        if (currentExt != ext) {
            die("Path '%s' has wong extension: %s instead of %s.",
                path.c_str(), currentExt.c_str(), ext.c_str());
        }
    }
    else {
        fname += '.';
        fname += ext;
    }

    return joinPaths(parentPath, fname);
}

} // namespace mirheo
