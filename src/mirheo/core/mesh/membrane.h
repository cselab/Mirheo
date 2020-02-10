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
    MembraneMesh(Undumper&, const ConfigDictionary&);

    MembraneMesh(MembraneMesh&&);
    MembraneMesh& operator=(MembraneMesh&&);
    ~MembraneMesh();

    void saveSnapshotAndRegister(Dumper&) override;

    PinnedBuffer<int> adjacent, degrees;
    PinnedBuffer<real> initialLengths, initialAreas, initialDotProducts;

protected:
    ConfigDictionary _saveSnapshot(Dumper&, const std::string& typeName);
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
