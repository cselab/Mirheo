#pragma once

#include "interface.h"
#include "kernels/api.h"

#include <random>

namespace mirheo
{

/**
 * Implements bounce-back from analytical shapes
 */
template <class Shape>
class BounceFromRigidShape : public Bouncer
{
public:

    BounceFromRigidShape(const MirState *state, const std::string& name, VarBounceKernel varBounceKernel);
    ~BounceFromRigidShape();

    void setup(ObjectVector *ov) override;

    void setPrerequisites(ParticleVector *pv) override;
    std::vector<std::string> getChannelsToBeExchanged() const override;
    std::vector<std::string> getChannelsToBeSentBack() const override;
    
protected:

    void exec(ParticleVector *pv, CellList *cl, ParticleVectorLocality locality, cudaStream_t stream) override;

    VarBounceKernel varBounceKernel_;
    std::mt19937 rng_ {42L};
};

} // namespace mirheo
