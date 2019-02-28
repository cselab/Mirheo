#pragma once

#include "mesh.h"

#include <core/containers.h>
#include <core/utils/pytypes.h>

class MembraneMesh : public Mesh
{
public:
    PinnedBuffer<int> adjacent, degrees;
    PinnedBuffer<float> initialLengths, initialAreas, initialDotProducts;

    MembraneMesh();

    MembraneMesh(std::string initialMesh);
    MembraneMesh(std::string initialMesh, std::string stressFreeMesh);

    MembraneMesh(const PyTypes::VectorOfFloat3& vertices,
                 const PyTypes::VectorOfInt3& faces);
    
    MembraneMesh(const PyTypes::VectorOfFloat3& vertices,
                 const PyTypes::VectorOfFloat3& stressFreeVertices,
                 const PyTypes::VectorOfInt3& faces);

    MembraneMesh(MembraneMesh&&);
    MembraneMesh& operator=(MembraneMesh&&);

    ~MembraneMesh();

protected:
    void findAdjacent();

    void _computeInitialQuantities(const PinnedBuffer<float4>& vertices);
    void _computeInitialLengths(const PinnedBuffer<float4>& vertices);
    void _computeInitialAreas(const PinnedBuffer<float4>& vertices);
    void _computeInitialDotProducts(const PinnedBuffer<float4>& vertices); /// used in Lim to determine if cos(phi) < 0
};

struct MembraneMeshView : public MeshView
{
    int maxDegree;

    int *adjacent, *degrees;
    float *initialLengths, *initialAreas, *initialDotProducts;

    MembraneMeshView(const MembraneMesh *m);
};

