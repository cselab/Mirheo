#pragma once

#include "mesh.h"

#include <mirheo/core/containers.h>

namespace mirheo
{

class MembraneMesh : public Mesh
{
public:
    MembraneMesh();

    MembraneMesh(const std::string& initialMesh);
    MembraneMesh(const std::string& initialMesh, const std::string& stressFreeMesh);

    MembraneMesh(const std::vector<real3>& vertices,
                 const std::vector<int3>& faces);
    
    MembraneMesh(const std::vector<real3>& vertices,
                 const std::vector<real3>& stressFreeVertices,
                 const std::vector<int3>& faces);

    /** \brief Construct a membrane mesh from its snapshot. Reads both initial and stress-free mesh.
        \param [in] loader The \c Loader object. Provides load context and unserialization functions.
        \param [in] config The mesh parameters.
     */
    MembraneMesh(Loader& loader, const ConfigObject& config);

    MembraneMesh(MembraneMesh&&);
    MembraneMesh& operator=(MembraneMesh&&);
    ~MembraneMesh();

    /** \brief Dump the initial and the stress-free mesh in an .off file, create a ConfigObject with the mesh name and register it in the saver.
        \param [in,out] saver The \c Saver object. Provides save context and serialization functions.

        Checks that the object type is exactly MembraneMesh.
      */
    void saveSnapshotAndRegister(Saver& saver) override;

    PinnedBuffer<int> adjacent, degrees;
    PinnedBuffer<real> initialLengths, initialAreas, initialDotProducts;

protected:
    /** \brief Implementation of the snapshot saving. Reusable by potential derived classes.
        \param [in,out] saver The \c Saver object. Provides save context and serialization functions.
        \param [in] typeName The name of the type being saved.
      */
    ConfigObject _saveSnapshot(Saver& saver, const std::string& typeName);

    void findAdjacent();

    void _computeInitialQuantities(const PinnedBuffer<real4>& vertices);
    void _computeInitialLengths(const PinnedBuffer<real4>& vertices);
    void _computeInitialAreas(const PinnedBuffer<real4>& vertices);
    void _computeInitialDotProducts(const PinnedBuffer<real4>& vertices); /// used in Lim to determine if cos(phi) < 0
};

struct MembraneMeshView : public MeshView
{
    int maxDegree;

    int *adjacent, *degrees;
    real *initialLengths, *initialAreas, *initialDotProducts;

    MembraneMeshView(const MembraneMesh *m);
};

} // namespace mirheo
