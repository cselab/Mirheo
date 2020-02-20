#pragma once

#include "interface.h"
#include <mirheo/core/containers.h>

namespace mirheo
{

/// \brief Represent the in/out status of a single particle.
enum class BelongingTags
{
    Outside = 0, ///< The particle is outside.
    Inside  = 1  ///< The particle is inside.
};

/// \brief ObjectBelongingChecker base implementation.
class ObjectVectorBelongingChecker : public ObjectBelongingChecker
{
public:
    /** \brief Construct a ObjectVectorBelongingChecker object.
        \param [in] state Simulation state.
        \param [in] name Name of the bouncer.
     */
    ObjectVectorBelongingChecker(const MirState *state, const std::string& name);
    ~ObjectVectorBelongingChecker() override;

    void splitByBelonging(ParticleVector *src, ParticleVector *pvIn, ParticleVector *pvOut, cudaStream_t stream) override;
    void checkInner(ParticleVector *pv, CellList *cl, cudaStream_t stream) override;
    void setup(ObjectVector *ov) override;

    std::vector<std::string> getChannelsToBeExchanged() const override;
    ObjectVector* getObjectVector() override;

protected:
    /** \brief Compute the in/out status of particles against the registered ObjectVector.
        \param [in] pv Particles to be tagged.
        \param [in] cl Cell lists of pv.
        \param [in] stream Stream on which to execute.

        The in/out status will be stored in \ref tags_.
     */
    virtual void _tagInner(ParticleVector *pv, CellList *cl, cudaStream_t stream) = 0;

protected:
    ObjectVector *ov_; ///< The registered ObjectVector.

    DeviceBuffer<BelongingTags> tags_; ///< Work space to store the in/out status of each input particle.
    PinnedBuffer<int> nInside_{1};  ///< Number of particles inside the registered ObjectVector.
    PinnedBuffer<int> nOutside_{1}; ///< Number of particles outside the registered ObjectVector.
};

} // namespace mirheo
