#include "rigid_object_vector.h"
#include "views/rov.h"

#include <core/utils/kernel_launch.h>
#include <core/rigid_kernels/integration.h>

RigidObjectVector::RigidObjectVector(std::string name, float partMass,
                                     float3 J, const int objSize,
                                     std::shared_ptr<Mesh> mesh, const int nObjects) :
        ObjectVector( name, partMass, objSize,
                      new LocalRigidObjectVector(this, objSize, nObjects),
                      new LocalRigidObjectVector(this, objSize, 0) ),
        J(J)
{
    this->mesh = std::move(mesh);

    if (length(J) < 1e-5)
        die("Wrong momentum of inertia: [%f %f %f]", J.x, J.y, J.z);

    // rigid motion must be exchanged and shifted
    requireDataPerObject<RigidMotion>("motions", true, sizeof(RigidReal));
}

RigidObjectVector::RigidObjectVector(std::string name, float partMass,
                                     pyfloat3 J, const int objSize,
                                     std::shared_ptr<Mesh> mesh, const int nObjects) :
        RigidObjectVector( name, partMass, make_float3(J), objSize, mesh, nObjects )
{   }

PinnedBuffer<Particle>* LocalRigidObjectVector::getMeshVertices(cudaStream_t stream)
{
    auto ov = dynamic_cast<RigidObjectVector*>(pv);
    auto& mesh = ov->mesh;
    meshVertices.resize_anew(nObjects * mesh->nvertices);

    ROVview fakeView(ov, this);
    fakeView.objSize = mesh->nvertices;
    fakeView.size = mesh->nvertices * nObjects;
    fakeView.particles = reinterpret_cast<float4*>(meshVertices.devPtr());

    SAFE_KERNEL_LAUNCH(
            applyRigidMotion,
            getNblocks(fakeView.size, 128), 128, 0, stream,
            fakeView, ov->mesh->vertexCoordinates.devPtr() );

    return &meshVertices;
}

PinnedBuffer<Particle>* LocalRigidObjectVector::getOldMeshVertices(cudaStream_t stream)
{
    auto ov = dynamic_cast<RigidObjectVector*>(pv);
    auto& mesh = ov->mesh;
    meshOldVertices.resize_anew(nObjects * mesh->nvertices);

    // Overwrite particles with vertices
    // Overwrite motions with the old_motions
    ROVview fakeView(ov, this);
    fakeView.objSize = mesh->nvertices;
    fakeView.size = mesh->nvertices * nObjects;
    fakeView.particles = reinterpret_cast<float4*>(meshOldVertices.devPtr());
    fakeView.motions = extraPerObject.getData<RigidMotion>("old_motions")->devPtr();

    SAFE_KERNEL_LAUNCH(
            applyRigidMotion,
            getNblocks(fakeView.size, 128), 128, 0, stream,
            fakeView, ov->mesh->vertexCoordinates.devPtr() );

    return &meshOldVertices;
}

DeviceBuffer<Force>* LocalRigidObjectVector::getMeshForces(cudaStream_t stream)
{
    auto ov = dynamic_cast<ObjectVector*>(pv);
    meshForces.resize_anew(nObjects * ov->mesh->nvertices);

    return &meshForces;
}
