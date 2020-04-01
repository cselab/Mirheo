#pragma once

namespace mirheo
{
class ObjectVector;
class LocalObjectVector;

/** \brief compute center of mass and bounding box of objects
    \param [in,out] ov The parent of lov
    \param [in,out] lov The LocalObjectVector that contains the objects. Will contain the ComAndExtents data per object on return.
    \param [in] stream The execution stream.
 */
void computeComExtents(ObjectVector *ov, LocalObjectVector *lov, cudaStream_t stream);

} // namespace mirheo
