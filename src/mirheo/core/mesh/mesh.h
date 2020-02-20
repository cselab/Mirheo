#pragma once

#include <mirheo/core/containers.h>
#include <mirheo/core/utils/common.h>
#include <mirheo/core/utils/pytypes.h>

#include <tuple>
#include <vector>
#include <vector_types.h>

namespace mirheo
{

class Mesh : public AutoObjectSnapshotTag
{
public:
    PinnedBuffer<int3> triangles;
    PinnedBuffer<real4> vertexCoordinates;

    Mesh();
    Mesh(const std::string& fileName);
    Mesh(const std::tuple<std::vector<real3>, std::vector<int3>>& mesh);
    Mesh(const std::vector<real3>& vertices, const std::vector<int3>& faces);

    /** \brief Construct a mesh from its snapshot.
        \param [in] loader The \c Loader object. Provides load context and unserialization functions.
        \param [in] config The mesh parameters.
     */
    Mesh(Loader& loader, const ConfigObject& config);

    Mesh(Mesh&&);
    Mesh& operator=(Mesh&&);

    virtual ~Mesh();

    const int& getNtriangles() const;
    const int& getNvertices() const;
    const int& getMaxDegree() const;

    PyTypes::VectorOfReal3 getVertices();
    PyTypes::VectorOfInt3  getTriangles();

    /** \brief Dump the mesh in an .off file, create a ConfigObject with the mesh name  and register it in the saver.
        \param [in,out] saver The \c Saver object. Provides save context and serialization functions.

        Checks that the object type is exactly \c Mesh.
      */
    virtual void saveSnapshotAndRegister(Saver& saver);

protected:
    /** \brief Implementation of the snapshot saving. Reusable by potential derived classes.
        \param [in,out] saver The \c Saver object. Provides save context and serialization functions.
        \param [in] typeName The name of the type being saved.
      */
    ConfigObject _saveSnapshot(Saver& saver, const std::string& typeName);

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

} // namespace mirheo
