#pragma once

namespace mirheo
{
class ObjectVector;
class LocalObjectVector;

void computeComExtents(ObjectVector *ov, LocalObjectVector *lov, cudaStream_t stream);

} // namespace mirheo
