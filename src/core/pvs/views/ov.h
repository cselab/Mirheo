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
    float objMass {0.f}, invObjMass {0.f};

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
    float2 *area_volumes {nullptr};

    OVviewWithAreaVolume(ObjectVector *ov, LocalObjectVector *lov) :
        OVview(ov, lov)
    {
        area_volumes = lov->dataPerObject.getData<float2>(ChannelNames::areaVolumes)->devPtr();
    }
};

struct OVviewWithJuelicherQuants : public OVviewWithAreaVolume
{
    float *vertexAreas          {nullptr};
    float *vertexMeanCurvatures {nullptr};

    float *lenThetaTot {nullptr};

    OVviewWithJuelicherQuants(ObjectVector *ov, LocalObjectVector *lov) :
        OVviewWithAreaVolume(ov, lov)
    {
        vertexAreas          = lov->dataPerParticle.getData<float>(ChannelNames::areas)->devPtr();
        vertexMeanCurvatures = lov->dataPerParticle.getData<float>(ChannelNames::meanCurvatures)->devPtr();

        lenThetaTot = lov->dataPerObject.getData<float>(ChannelNames::lenThetaTot)->devPtr();
    }
};

struct OVviewWithNewOldVertices : public OVview
{
    float4 *vertices     {nullptr};
    float4 *old_vertices {nullptr};
    float4 *vertexForces {nullptr};

    int nvertices {0};

    OVviewWithNewOldVertices(ObjectVector *ov, LocalObjectVector *lov, cudaStream_t stream) :
        OVview(ov, lov)
    {
        nvertices    = ov->mesh->getNvertices();
        vertices     = reinterpret_cast<float4*>( lov->getMeshVertices   (stream)->devPtr() );
        old_vertices = reinterpret_cast<float4*>( lov->getOldMeshVertices(stream)->devPtr() );
        vertexForces = reinterpret_cast<float4*>( lov->getMeshForces     (stream)->devPtr() );
    }
};
