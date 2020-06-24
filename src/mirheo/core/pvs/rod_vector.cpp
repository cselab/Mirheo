#include "rod_vector.h"
#include "views/rv.h"

#include <mirheo/core/utils/quaternion.h>
#include <mirheo/core/utils/kernel_launch.h>

namespace mirheo
{

inline constexpr int getNumParts(int nSegments)
{
    return 5 * nSegments + 1;
}

inline constexpr int getNumSegments(int np)
{
    return (np - 1) / 5;
}

LocalRodVector::LocalRodVector(ParticleVector *pv, int objSize, int nObjects) :
    LocalObjectVector(pv, objSize, nObjects)
{
    resize_anew(objSize * nObjects);
}

LocalRodVector::~LocalRodVector() = default;

void LocalRodVector::resize(int np, cudaStream_t stream)
{
    LocalObjectVector::resize(np, stream);

    int totNumBisegments = (getNumSegmentsPerRod() - 1) * getNumObjects();

    dataPerBisegment.resize(totNumBisegments, stream);
}

void LocalRodVector::resize_anew(int np)
{
    LocalObjectVector::resize_anew(np);

    const int totNumBisegments = (getNumSegmentsPerRod() - 1) * getNumObjects();

    dataPerBisegment.resize_anew(totNumBisegments);
}

int LocalRodVector::getNumSegmentsPerRod() const
{
    return getNumSegments(getObjectSize());
}

RodVector::RodVector(const MirState *state, const std::string& name, real mass, int nSegments, int nObjects) :
    ObjectVector( state, name, mass, getNumParts(nSegments),
                  std::make_unique<LocalRodVector>(this, getNumParts(nSegments), nObjects),
                  std::make_unique<LocalRodVector>(this, getNumParts(nSegments), 0) )
{}

RodVector::~RodVector() = default;

} // namespace mirheo
