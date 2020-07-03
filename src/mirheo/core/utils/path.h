// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <mpi.h>
#include <string>
#include <vector>

namespace mirheo
{

/** \brief Create a string representing an integer with 0 padding
    \param i The integer to print (must non negative)
    \param zeroPadding The total number of characters
    \return the string representation of \p i with padded zeros

    If \p zeroPadding is too small, this method will die.
    Example: \c createStrZeroPadded(42, 5) gives "00042"
 */
std::string createStrZeroPadded(long long i, int zeroPadding = 5);

/** \brief Split a string according to a delimiter character
    \param str The input sequence of characters
    \param delim The delimiter
    \return The list of substrings (without the delimiter)

    e.g. \c splitByDelim("string_to_split", '_') -> {"string", "to", "split"}
 */
std::vector<std::string> splitByDelim(std::string str, char delim = ',');

/** \brief append '/' at the end of \p path if it doesn t have it already
    \param path The path to work with
    \return The path with a trailing separator
 */
std::string makePath(std::string path);

/** \brief Get the parent folder of the given filename
    \param path The filename containing a path
    \return The parent folder.

    If the input is a path (it ends with a '/'), the output is the same as the input.
    If the input is just a filename with no '/', this function returns an empty string.
 */
std::string getParentPath(std::string path);

/** \brief remove the path from the given filename.
    \param path The filename with full relative or absolute path
    \return the filename only without any prepended folder
 */
std::string getBaseName(std::string path);

/** \brief Concatenate two paths A and B.
    \param A first part of the full path
    \param B last part of the full path
    \return A/B
    Adds a '/' between \p A and \p B if \p A is non-empty and if it doesn't already end with '/'.
*/
std::string joinPaths(const std::string &A, const std::string &B);

/** \brief Create a folder.
    \param comm The communicator used to decide which rank creates the folder
    \param path the folder to create
    \return \c true if the operation was successful, \c false otherwise

    The operation is collective. This means that all ranks in the \p comm must call it.
    The returned value is accessible by all ranks.
 */
bool createFoldersCollective(const MPI_Comm& comm, std::string path);

/** \brief Add extension to the given path if there is no extension set.
    \param path The input filename, with or without extension.
    \param ext The extension to add or check, without the dot.
    \return \p path + `.` + \p ext if path has no extension, or path if it has the correct extension.

    This function will die if the path already contains an extension that is not the required one.
 */
std::string setExtensionOrDie(std::string path, const std::string ext);

} // namespace mirheo
