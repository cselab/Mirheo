#pragma once

#include <core/containers.h>
#include <core/utils/pytypes.h>

#include <vector_types.h>
#include <vector>

class Mesh
{
public:
    PinnedBuffer<int3> triangles;
    PinnedBuffer<real4> vertexCoordinates;

    Mesh();
    Mesh(const std::string& filename);
    Mesh(const std::vector<real3>& vertices, const std::vector<int3>& faces);

    Mesh(Mesh&&);
    Mesh& operator=(Mesh&&);

    virtual ~Mesh();

    const int& getNtriangles() const;
    const int& getNvertices() const;
    const int& getMaxDegree() const;

    PyTypes::VectorOfReal3 getVertices();
    PyTypes::VectorOfInt3  getTriangles();

protected:
    // max degree of a vertex in mesh
    int maxDegree {-1};
    void _computeMaxDegree();
    void _check() const;
    void _readOff(const std::string& fname);

protected:
    int nvertices{0}, ntriangles{0};
};



struct MeshView
{
    int nvertices, ntriangles;
    int3 *triangles;

    MeshView(const Mesh *m);
};


