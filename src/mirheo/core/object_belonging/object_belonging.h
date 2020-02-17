#pragma once

#include "interface.h"
#include <mirheo/core/containers.h>

namespace mirheo
{

enum class BelongingTags
{
    Outside = 0, Inside = 1
};

class ObjectVectorBelongingChecker : public ObjectBelongingChecker
{
public:
    ObjectVectorBelongingChecker(const MirState *state, const std::string& name);
    ~ObjectVectorBelongingChecker() override;

    /**
     * Particle with tags == BelongingTags::Outside  will be copied to pvOut
     *                    == BelongingTags::Inside   will be copied to pvIn
     * Other particles are DROPPED (boundary particles)
     */
    void splitByBelonging(ParticleVector *src, ParticleVector *pvIn, ParticleVector *pvOut, cudaStream_t stream) override;
    void checkInner(ParticleVector *pv, CellList *cl, cudaStream_t stream) override;
    void setup(ObjectVector *ov) override;

    std::vector<std::string> getChannelsToBeExchanged() const override;
    ObjectVector* getObjectVector() override;
    
protected:
    ObjectVector *ov_;

    DeviceBuffer<BelongingTags> tags_;
    PinnedBuffer<int> nInside_{1}, nOutside_{1};

    virtual void tagInner(ParticleVector *pv, CellList *cl, cudaStream_t stream) = 0;
};

} // namespace mirheo
