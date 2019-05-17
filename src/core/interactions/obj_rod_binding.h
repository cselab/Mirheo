#pragma once

#include "interface.h"

class RigidObjectVector;
class RodVector;

class ObjectRodBindingInteraction : public Interaction
{
public:
    ObjectRodBindingInteraction(const YmrState *state, std::string name,
                                float torque, float3 relAnchor, float kBound);

    ~ObjectRodBindingInteraction();
    
    void setPrerequisites(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2) override;
    void local(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2, cudaStream_t stream) override;
    void halo (ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2, cudaStream_t stream) override;

protected:

    void _local(RigidObjectVector *rov, RodVector *rv, cudaStream_t stream) const;
    void  _halo(RigidObjectVector *rov, RodVector *rv, cudaStream_t stream) const;
protected:

    float torque; // torque magnitude to apply to the rod from the object
    float3 relAnchor; // relative position with respect to object of attachement point
    float kBound;
};
