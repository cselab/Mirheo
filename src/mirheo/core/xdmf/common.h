#include <string>
#include <vector>
#include <sstream>
#include <iterator>
#include <algorithm>
#include <memory>

namespace mirheo
{

/** \brief Serialze an array of element
    \tparam T The element type, must have << operator overload for std::ostringstream
    \param v The array of elements
    \return The string that contains space separated elements.
 */
template<typename T>
std::string to_string(const std::vector<T, std::allocator<T>>& v)
{
    std::ostringstream stream;
    std::copy(v.begin(), v.end(), std::ostream_iterator<T>(stream, " "));
    std::string s = stream.str();
    s.erase(s.length()-1);
    
    return s;
}

} // namespace mirheo
