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
    Example: \c getStrZeroPadded(42, 5) gives "00042"
 */
std::string getStrZeroPadded(long long i, int zeroPadding = 5);

/** \brief Split a string according to a delimiter character
    \param str The input sequence of characters
    \param delim The delimiter
    \return The list of substrings (without the delimiter)

    e.g. splitByDelim("string_to_split", '_') -> {"string", "to", "split"}
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
std::string parentPath(std::string path);

/** \brief remove the path from the given filename.
    \param path The filename with full relative o absolute path
    \return the filename only without any path
 */
std::string getBaseName(std::string path);

/** \brief Concatenate two paths A and B. 
    \param A first part of the full path
    \param B last part of the full path
    \return A/B
    Adds a '/' between \p A and \p B if \p A is non-empty and if it doesn't already end with '/'.
*/
std::string joinPaths(const std::string &A, const std::string &B);

bool createFoldersCollective(const MPI_Comm& comm, std::string path);

} // namespace mirheo
