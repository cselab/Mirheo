#pragma once

#include "object_belonging.h"

namespace mirheo
{

/// \brief Check in/out status of particles against a RodObjectVector.
class RodBelongingChecker : public ObjectVectorBelongingChecker
{
public:
    /** \brief Construct a RodBelongingChecker object.
        \param [in] state Simulation state.
        \param [in] name Name of the bouncer.
        \param [in] radius The radius of the rod. Must be positive.
     */
    RodBelongingChecker(const MirState *state, const std::string& name, real radius);

protected:
    void _tagInner(ParticleVector *pv, CellList *cl, cudaStream_t stream) override;

private:
    real radius_;
};

} // namespace mirheo
