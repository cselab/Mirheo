#pragma once

#include "mesh.h"

#include <core/containers.h>

class MembraneMesh : public Mesh
{
public:
    MembraneMesh();

    MembraneMesh(const std::string& initialMesh);
    MembraneMesh(const std::string& initialMesh, const std::string& stressFreeMesh);

    MembraneMesh(const std::vector<float3>& vertices,
                 const std::vector<int3>& faces);
    
    MembraneMesh(const std::vector<float3>& vertices,
                 const std::vector<float3>& stressFreeVertices,
                 const std::vector<int3>& faces);

    MembraneMesh(MembraneMesh&&);
    MembraneMesh& operator=(MembraneMesh&&);

    ~MembraneMesh();

    PinnedBuffer<int> adjacent, degrees;
    PinnedBuffer<float> initialLengths, initialAreas, initialDotProducts;


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

