#include "off.h"

#include <mirheo/core/logger.h>

#include <fstream>

namespace mirheo
{

std::tuple<std::vector<real3>, std::vector<int3>> readOff(const std::string& fileName)
{
    std::vector<int3> faces;
    std::vector<real3> vertices;
    int nfaces {0}, nvertices {0}, nedges {0};
    std::ifstream fin(fileName);

    if (!fin.good())
        die("off file '%s' not found", fileName.c_str());

    debug("Reading off file '%s'", fileName.c_str());

    std::string line;
    std::getline(fin, line); // OFF header

    fin >> nvertices >> nfaces >> nedges;
    std::getline(fin, line); // Finish with this line

    vertices.reserve(nvertices);
    faces.reserve(nfaces);

    // Read the vertex coordinates
    for (int i = 0; i < nvertices; ++i)
    {
        real3 v;
        fin >> v.x >> v.y >> v.z;
        vertices.push_back(v);
    }

    // Read the connectivity data
    for (int i = 0; i < nfaces; ++i)
    {
        int number;
        int3 f;
        fin >> number;
        if (number != 3)
            die("Bad mesh file '%s' on line %d, number of face vertices is %d instead of 3",
                fileName.c_str(), 3 /* header */ + nvertices + i, number);

        fin >> f.x >> f.y >> f.z;
        faces.push_back(f);
    }
    return {std::move(vertices), std::move(faces)};
}

void writeOff(const std::vector<real3>& vertices, const std::vector<int3>& faces, const std::string& fileName)
{
    std::ofstream fout(fileName);

    if (!fout.good())
        die("could not read from off file '%s'", fileName.c_str());

    fout << "OFF\n";
    fout << vertices.size() << ' ' << faces.size() << " 0\n";

    for (const auto& r : vertices)
        fout << r.x << ' ' << r.y << ' ' << r.z << '\n';

    for (const auto& t : faces)
        fout << "3 " << t.x << ' ' << t.y << ' ' << t.z << '\n';
}

} // namespace mirheo
