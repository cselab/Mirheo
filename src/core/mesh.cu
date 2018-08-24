#include "mesh.h"

#include <fstream>
#include <unordered_map>
#include <map>
#include <vector>

#include <core/utils/cuda_common.h>

/// Read off mesh
Mesh::Mesh(std::string fname)
{
    std::ifstream fin(fname);
    if (!fin.good())
        die("Mesh file '%s' not found", fname.c_str());

    debug("Reading mesh from file '%s'", fname.c_str());

    std::string line;
    std::getline(fin, line); // OFF header

    int nedges;
    fin >> nvertices >> ntriangles >> nedges;
    std::getline(fin, line); // Finish with this line

    // Read the vertex coordinates
    vertexCoordinates.resize_anew(nvertices);
    for (int i=0; i<nvertices; i++)
        fin >> vertexCoordinates[i].x >> vertexCoordinates[i].y >> vertexCoordinates[i].z;

    // Read the connectivity data
    triangles.resize_anew(ntriangles);
    for (int i=0; i<ntriangles; i++)
    {
        int number;
        fin >> number;
        if (number != 3)
            die("Bad mesh file '%s' on line %d, number of face vertices is %d instead of 3",
                    fname.c_str(), 3 /* header */ + nvertices + i, number);

        fin >> triangles[i].x >> triangles[i].y >> triangles[i].z;

        auto check = [&] (int tr) {
            if (tr < 0 || tr >= nvertices)
                die("Bad triangle indices in mesh '%s' on line %d", fname.c_str(), 3 /* header */ + nvertices + i);
        };

        check(triangles[i].x);
        check(triangles[i].y);
        check(triangles[i].z);
    }


    vertexCoordinates.uploadToDevice(0);
    triangles.uploadToDevice(0);

    _computeMaxDegree();
}

int Mesh::getMaxDegree() const {
    if (maxDegree < 0) die("maxDegree was not computed");
    return maxDegree;
}

void Mesh::_computeMaxDegree()
{
    std::vector<int> degrees(nvertices);

    for (auto t : triangles) {
        degrees[t.x] ++;
        degrees[t.y] ++;
        degrees[t.z] ++;
    }

    maxDegree = *std::max_element(degrees.begin(), degrees.end());
    debug("max degree is %d", maxDegree);
}

MembraneMesh::MembraneMesh(std::string fname) : Mesh(fname)
{
    findAdjacent();

    initialLengths.resize_anew(nvertices * maxDegree);

    for (int i=0; i<nvertices*maxDegree; i++)
    {
        if (adjacent[i] >= 0)
            initialLengths[i] = length(vertexCoordinates[i / maxDegree] - vertexCoordinates[adjacent[i]]);
    }

    initialLengths.uploadToDevice(0);
}


void MembraneMesh::findAdjacent()
{
    std::vector< std::map<int, int> > adjacentPairs(nvertices);

    for(int i = 0; i < triangles.size(); ++i)
    {
        const int tri[3] = {triangles[i].x, triangles[i].y, triangles[i].z};

        for(int d = 0; d < 3; ++d)
            adjacentPairs[tri[d]][tri[(d + 1) % 3]] = tri[(d + 2) % 3];
    }

    degrees.resize_anew(nvertices);
    for(int i = 0; i < nvertices; ++i)
        degrees[i] = adjacentPairs[i].size();

    auto it = std::max_element(degrees.hostPtr(), degrees.hostPtr() + nvertices);
    const int curMaxDegree = *it;

    if (curMaxDegree > maxDegree)
        die("Degree of vertex %d is %d > %d (max degree supported)", (int)(it - degrees.hostPtr()), curMaxDegree, maxDegree);

    debug("Max degree of mesh vertices is %d", curMaxDegree);

    // Find first (nearest) neighbors of each vertex
    adjacent.resize_anew(ntriangles * maxDegree);
    for (int i=0; i<adjacent.size(); i++)
        adjacent[i] = -1;

    for(int v = 0; v < nvertices; ++v)
    {
        auto& l = adjacentPairs[v];

        adjacent[0 + maxDegree * v] = l.begin()->first;
        int last = adjacent[1 + maxDegree * v] = l.begin()->second;

        for(int i = 2; i < l.size(); ++i)
        {
            assert(l.find(last) != l.end());

            int tmp = adjacent[i + maxDegree * v] = l.find(last)->second;
            last = tmp;
        }
    }


    // Find distance 2 neighbors of each vertex
    adjacent_second.resize_anew(ntriangles * maxDegree);
    for (int i=0; i<adjacent_second.size(); i++)
        adjacent_second[i] = -1;

    // Get all the vertex neighbors from already compiled adjacent array
    auto extract_neighbors = [&] (const int v) {

        std::vector<int> myneighbors;
        for(int c = 0; c < maxDegree; ++c)
        {
            const int val = adjacent[c + maxDegree * v];
            if (val == -1)
                break;

            myneighbors.push_back(val);
        }

        return myneighbors;
    };

    for(int v = 0; v < nvertices; ++v)
    {
        auto myneighbors = extract_neighbors(v);

        for(int i = 0; i < myneighbors.size(); ++i)
        {
            auto s1 = extract_neighbors(myneighbors[i]);
            std::sort(s1.begin(), s1.end());

            auto s2 = extract_neighbors(myneighbors[(i + 1) % myneighbors.size()]);
            std::sort(s2.begin(), s2.end());

            std::vector<int> result(s1.size() + s2.size());

            const int nterms = std::set_intersection(s1.begin(), s1.end(), s2.begin(), s2.end(),
                    result.begin()) - result.begin();

            assert(nterms == 2);

            const int myguy = result[0] == v;

            adjacent_second[i + maxDegree * v] = result[myguy];
        }
    }


    for(int v = 0; v < nvertices; ++v)
    {
        for (int i=0; i<maxDegree; i++)
            if (adjacent[v*maxDegree + i] == -1)
            {
                adjacent[v*maxDegree + i] = adjacent[v*maxDegree];
                break;
            }

        for (int i=0; i<maxDegree; i++)
            if (adjacent_second[v*maxDegree + i] == -1)
            {
                adjacent_second[v*maxDegree + i] = adjacent_second[v*maxDegree];
                break;
            }
    }

    adjacent.uploadToDevice(0);
    adjacent_second.uploadToDevice(0);
    degrees.uploadToDevice(0);
}



