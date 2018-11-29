#pragma once

#include <core/containers.h>
#include <core/datatypes.h>
#include <core/utils/pytypes.h>

class Mesh
{
protected:
    int nvertices{0}, ntriangles{0};

public:
    PinnedBuffer<int3> triangles;

    PinnedBuffer<float4> vertexCoordinates;

    Mesh() {};
    Mesh(std::string);
    Mesh(const PyTypes::VectorOfFloat3& vertices, const PyTypes::VectorOfInt3& faces);

    Mesh(Mesh&&) = default;
    Mesh& operator=(Mesh&&) = default;

    const int& getNtriangles() const;
    const int& getNvertices() const;
    const int& getMaxDegree() const;

    PyTypes::VectorOfFloat3 getVertices();
    PyTypes::VectorOfInt3  getTriangles();

protected:
    // max degree of a vertex in mesh
    int maxDegree {-1};
    void _computeMaxDegree();
    void _check() const;
    void _readOff(std::string fname);
};

class MembraneMesh : public Mesh
{
public:
    PinnedBuffer<int> adjacent, degrees;
    PinnedBuffer<float> initialLengths, initialAreas;

    MembraneMesh() {};
    MembraneMesh(std::string);
    MembraneMesh(const PyTypes::VectorOfFloat3& vertices, const PyTypes::VectorOfInt3& faces);

    MembraneMesh(MembraneMesh&&) = default;
    MembraneMesh& operator=(MembraneMesh&&) = default;

protected:
    void findAdjacent();
    void computeInitialLengths();
    void computeInitialAreas();
};



struct MeshView
{
    int nvertices, ntriangles;
    int3* triangles;

    MeshView(const Mesh* m)
    {
        nvertices = m->getNvertices();
        ntriangles = m->getNtriangles();

        triangles = m->triangles.devPtr();
    }
};

struct MembraneMeshView : public MeshView
{
    int maxDegree;

    int *adjacent, *degrees;
    float *initialLengths, *initialAreas;

    MembraneMeshView(const MembraneMesh* m) : MeshView(m)
    {
        maxDegree = m->getMaxDegree();

        adjacent        = m->adjacent.devPtr();
        degrees         = m->degrees.devPtr();
        initialLengths  = m->initialLengths.devPtr();
        initialAreas    = m->initialAreas.devPtr();
    }
};


