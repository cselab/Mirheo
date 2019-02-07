#pragma once

#include "mesh.h"

#include <core/containers.h>
#include <core/utils/pytypes.h>

class MembraneMesh : public Mesh
{
public:
    PinnedBuffer<int> adjacent, degrees;
    PinnedBuffer<float> initialLengths, initialAreas;

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
    void computeInitialLengths(const PinnedBuffer<float4>& vertices);
    void computeInitialAreas(const PinnedBuffer<float4>& vertices);
};

struct MembraneMeshView : public MeshView
{
    int maxDegree;

    int *adjacent, *degrees;
    float *initialLengths, *initialAreas;

    MembraneMeshView(const MembraneMesh *m) : MeshView(m)
    {
        maxDegree = m->getMaxDegree();

        adjacent        = m->adjacent.devPtr();
        degrees         = m->degrees.devPtr();
        initialLengths  = m->initialLengths.devPtr();
        initialAreas    = m->initialAreas.devPtr();
    }
};

