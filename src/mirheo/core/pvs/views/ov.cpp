#include "ov.h"

#include <mirheo/core/pvs/object_vector.h>

namespace mirheo
{

OVview::OVview(ObjectVector *ov, LocalObjectVector *lov) :
    PVview(ov, lov)
{
    nObjects = lov->getNumObjects();
    objSize  = ov->getObjectSize();
    objMass  = static_cast<real>(objSize) * mass;
    invObjMass = 1.0_r / objMass;

    comAndExtents = lov->dataPerObject.getData<COMandExtent>(channel_names::comExtents)->devPtr();
    ids           = lov->dataPerObject.getData<int64_t>(channel_names::globalIds)->devPtr();
}

OVviewWithAreaVolume::OVviewWithAreaVolume(ObjectVector *ov, LocalObjectVector *lov) :
    OVview(ov, lov)
{
    area_volumes = lov->dataPerObject.getData<real2>(channel_names::areaVolumes)->devPtr();
}

OVviewWithJuelicherQuants::OVviewWithJuelicherQuants(ObjectVector *ov, LocalObjectVector *lov) :
    OVviewWithAreaVolume(ov, lov)
{
    vertexAreas          = lov->dataPerParticle.getData<real>(channel_names::areas)->devPtr();
    vertexMeanCurvatures = lov->dataPerParticle.getData<real>(channel_names::meanCurvatures)->devPtr();

    lenThetaTot = lov->dataPerObject.getData<real>(channel_names::lenThetaTot)->devPtr();
}

OVviewWithNewOldVertices::OVviewWithNewOldVertices(ObjectVector *ov, LocalObjectVector *lov, cudaStream_t stream) :
    OVview(ov, lov)
{
    nvertices    = ov->mesh->getNvertices();
    vertices     = reinterpret_cast<real4*>( lov->getMeshVertices   (stream)->devPtr() );
    old_vertices = reinterpret_cast<real4*>( lov->getOldMeshVertices(stream)->devPtr() );
    vertexForces = reinterpret_cast<real4*>( lov->getMeshForces     (stream)->devPtr() );
}

} // namespace mirheo
