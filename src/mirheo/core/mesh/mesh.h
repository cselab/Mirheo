#pragma once

#include <mirheo/core/containers.h>
#include <mirheo/core/utils/common.h>
#include <mirheo/core/utils/pytypes.h>

#include <tuple>
#include <vector>
#include <vector_types.h>

namespace mirheo
{

class Mesh
{
public:
    PinnedBuffer<int3> triangles;
    PinnedBuffer<real4> vertexCoordinates;

    Mesh();
    Mesh(const std::string& fileName);
    Mesh(const std::tuple<std::vector<real3>, std::vector<int3>>& mesh);
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
    void _computeMaxDegree();
    void _check() const;

private:
    int nvertices_{0};
    int ntriangles_{0};

    // max degree of a vertex in mesh
    int maxDegree_ {-1};
};


struct MeshView
{
    int nvertices, ntriangles;
    int3 *triangles;

    MeshView(const Mesh *m);
};


template <>
struct ConfigDumper<Mesh>
{
    static Config dump(Dumper&, const Mesh&);
};

} // namespace mirheo
