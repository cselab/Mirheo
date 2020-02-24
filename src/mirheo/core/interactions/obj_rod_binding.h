#pragma once

#include "interface.h"

namespace mirheo
{

class RigidObjectVector;
class RodVector;

/** Compute elastic interaction used to attach a rod to a RigidObjectVector entity.
 */
class ObjectRodBindingInteraction : public Interaction
{
public:
    /** Construct an ObjectRodBindingInteraction interaction
        \param [in] state The global state of the system
        \param [in] name The name of the interaction
        \param [in] torque Torque applied from the rigid objet to the rod
        \param [in] relAnchor position of attachement with respect to the rigid object
        \param [in] kBound The elastic constant for the binding
     */
    ObjectRodBindingInteraction(const MirState *state, std::string name,
                                real torque, real3 relAnchor, real kBound);

    ~ObjectRodBindingInteraction();
    
    void setPrerequisites(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2) override;
    void local(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2, cudaStream_t stream) override;
    void halo (ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2, cudaStream_t stream) override;

private:

    void _local(RigidObjectVector *rov, RodVector *rv, cudaStream_t stream) const;
    void  _halo(RigidObjectVector *rov, RodVector *rv, cudaStream_t stream) const;
private:

    real torque_; // torque magnitude to apply to the rod from the object
    real3 relAnchor_; // relative position with respect to object of attachement point
    real kBound_;
};

} // namespace mirheo
