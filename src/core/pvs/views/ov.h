#pragma once

//#include <core/rigid_kernels/utils.h>
#include "../object_vector.h"
#include "pv.h"

/**
 * GPU-compatible struct of all the relevant data
 */
struct OVview : public PVview
{
    int nObjects {0}, objSize {0};
    real objMass {0.f}, invObjMass {0.f};

    COMandExtent *comAndExtents {nullptr};
    int64_t *ids {nullptr};

    OVview(ObjectVector *ov, LocalObjectVector *lov) :
        PVview(ov, lov)
    {
        nObjects = lov->nObjects;
        objSize  = ov->objSize;
        objMass  = objSize * mass;
        invObjMass = 1.0 / objMass;

        comAndExtents = lov->dataPerObject.getData<COMandExtent>(ChannelNames::comExtents)->devPtr();
        ids           = lov->dataPerObject.getData<int64_t>(ChannelNames::globalIds)->devPtr();
    }
};

struct OVviewWithAreaVolume : public OVview
{
    real2 *area_volumes {nullptr};

    OVviewWithAreaVolume(ObjectVector *ov, LocalObjectVector *lov) :
        OVview(ov, lov)
    {
        area_volumes = lov->dataPerObject.getData<real2>(ChannelNames::areaVolumes)->devPtr();
    }
};

struct OVviewWithJuelicherQuants : public OVviewWithAreaVolume
{
    real *vertexAreas          {nullptr};
    real *vertexMeanCurvatures {nullptr};

    real *lenThetaTot {nullptr};

    OVviewWithJuelicherQuants(ObjectVector *ov, LocalObjectVector *lov) :
        OVviewWithAreaVolume(ov, lov)
    {
        vertexAreas          = lov->dataPerParticle.getData<real>(ChannelNames::areas)->devPtr();
        vertexMeanCurvatures = lov->dataPerParticle.getData<real>(ChannelNames::meanCurvatures)->devPtr();

        lenThetaTot = lov->dataPerObject.getData<real>(ChannelNames::lenThetaTot)->devPtr();
    }
};

struct OVviewWithNewOldVertices : public OVview
{
    real4 *vertices     {nullptr};
    real4 *old_vertices {nullptr};
    real4 *vertexForces {nullptr};

    int nvertices {0};

    OVviewWithNewOldVertices(ObjectVector *ov, LocalObjectVector *lov, cudaStream_t stream) :
        OVview(ov, lov)
    {
        nvertices    = ov->mesh->getNvertices();
        vertices     = reinterpret_cast<real4*>( lov->getMeshVertices   (stream)->devPtr() );
        old_vertices = reinterpret_cast<real4*>( lov->getOldMeshVertices(stream)->devPtr() );
        vertexForces = reinterpret_cast<real4*>( lov->getMeshForces     (stream)->devPtr() );
    }
};
