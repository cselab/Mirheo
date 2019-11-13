#pragma once

#include "pv.h"

namespace mirheo
{

class ObjectVector;
class LocalObjectVector;

struct OVview : public PVview
{
    OVview(ObjectVector *ov, LocalObjectVector *lov);

    int nObjects {0}, objSize {0};
    real objMass {0._r}, invObjMass {0._r};

    COMandExtent *comAndExtents {nullptr};
    int64_t *ids {nullptr};
};

struct OVviewWithAreaVolume : public OVview
{
    OVviewWithAreaVolume(ObjectVector *ov, LocalObjectVector *lov);

    real2 *area_volumes {nullptr};
};

struct OVviewWithJuelicherQuants : public OVviewWithAreaVolume
{
    OVviewWithJuelicherQuants(ObjectVector *ov, LocalObjectVector *lov);

    real *vertexAreas          {nullptr};
    real *vertexMeanCurvatures {nullptr};

    real *lenThetaTot {nullptr};
};

struct OVviewWithNewOldVertices : public OVview
{
    OVviewWithNewOldVertices(ObjectVector *ov, LocalObjectVector *lov, cudaStream_t stream);

    real4 *vertices     {nullptr};
    real4 *old_vertices {nullptr};
    real4 *vertexForces {nullptr};

    int nvertices {0};
};

} // namespace mirheo
