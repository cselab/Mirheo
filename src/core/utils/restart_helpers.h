#pragma once

#include <core/datatypes.h>

#include <string>
#include <fstream>

inline std::ostream& operator<<(std::ostream& s, const real3& v)
{
    s << v.x << " " << v.y << " " << v.z;
    return s;
}

inline std::ifstream& operator>>(std::ifstream& s, real3& v)
{
    s >> v.x >> v.y >> v.z;
    return s;
}


namespace TextIO
{
template<typename Arg>
void writeToStream(std::ofstream& fout, const Arg& arg)
{
    fout << arg << std::endl;
}

template<typename Arg, typename... Args>
void writeToStream(std::ofstream& fout, const Arg& arg, const Args&... args)
{
    fout << arg << std::endl;
    writeToStream(fout, args...);
}

template<typename... Args>
void write(std::string fname, const Args&... args)
{
    std::ofstream fout(fname);
    writeToStream(fout, args...);
}



template<typename Arg>
bool readFromStream(std::ifstream& fin, Arg& arg)
{
    return (fin >> arg).good();
}

template<typename Arg, typename... Args>
bool readFromStream(std::ifstream& fin, Arg& arg, Args&... args)
{
    return (fin >> arg).good() && readFromStream(fin, args...);
}

template<typename... Args>
bool read(std::string fname, Args&... args)
{
    std::ifstream fin(fname);
    return fin.good() && readFromStream(fin, args...);
}

} // namespace TextIO
