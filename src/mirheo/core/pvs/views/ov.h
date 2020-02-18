#pragma once

#include "pv.h"

namespace mirheo
{

class ObjectVector;
class LocalObjectVector;

/** \brief A \c PVview with additionally basic object data
    
    Contains object ids, object extents.
 */
struct OVview : public PVview
{
    /** \brief Construct a \c OVview 
        \param [in] ov The \c ObjectVector that the view represents
        \param [in] lov The \c LocalObjectVector that the view represents
    */
    OVview(ObjectVector *ov, LocalObjectVector *lov);

    int nObjects {0}; ///< number of objects
    int objSize {0}; ///< number of particles per object
    real objMass {0._r}; ///< mass of one object
    real invObjMass {0._r}; ///< 1 / objMass

    COMandExtent *comAndExtents {nullptr}; ///< center of mass and extents of the objects
    int64_t *ids {nullptr}; ///< global ids of objects
};

/// \brief A \c OVview with additionally area and volume information
struct OVviewWithAreaVolume : public OVview
{
    /** \brief Construct a \c OVviewWithAreaVolume
        \param [in] ov The \c ObjectVector that the view represents
        \param [in] lov The \c LocalObjectVector that the view represents

        \rst
        .. warning::
            The ov must hold a areaVolumes channel.
        \endrst
    */
    OVviewWithAreaVolume(ObjectVector *ov, LocalObjectVector *lov);

    real2 *area_volumes {nullptr}; ///< area and volume per object
};

/// \brief A \c OVviewWithAreaVolume with additional curvature related quantities
struct OVviewWithJuelicherQuants : public OVviewWithAreaVolume
{
    /** \brief Construct a \c OVviewWithJuelicherQuants 
        \param [in] ov The \c ObjectVector that the view represents
        \param [in] lov The \c LocalObjectVector that the view represents

        \rst
        .. warning::
            The ov must hold areaVolumes and lenThetaTot object channels and vertex areas, meanCurvatures particle channels.
        \endrst
    */
    OVviewWithJuelicherQuants(ObjectVector *ov, LocalObjectVector *lov);

    real *vertexAreas          {nullptr}; ///< area per vertex (defined on a triangle mesh)
    real *vertexMeanCurvatures {nullptr}; ///< mean curvature vertex (defined on a triangle mesh)

    real *lenThetaTot {nullptr}; ///< helper quantity to compute Juelicher bending energy
};

/// \brief A \c OVview with additionally vertices information
struct OVviewWithNewOldVertices : public OVview
{
    /** \brief Construct a \c OVviewWithNewOldVertices
        \param [in] ov The \c ObjectVector that the view represents
        \param [in] lov The \c LocalObjectVector that the view represents
        \param [in] stream Stream used to create mesh vertices if not already present
    */
    OVviewWithNewOldVertices(ObjectVector *ov, LocalObjectVector *lov, cudaStream_t stream);

    real4 *vertices     {nullptr}; ///< vertex positions
    real4 *old_vertices {nullptr}; ///< vertex positions at previous time step
    real4 *vertexForces {nullptr}; ///< vertex forces

    int nvertices {0}; ///< number of vertices
};

} // namespace mirheo
