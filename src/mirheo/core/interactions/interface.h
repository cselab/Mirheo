#pragma once

#include <mirheo/core/mirheo_object.h>

#include <cuda_runtime.h>
#include <functional>
#include <memory>
#include <mpi.h>
#include <vector>

class CellList;
class ParticleVector;

/**
 * Interface for classes computing particle interactions.
 *
 * At the moment cut-off radius is the part of the interface,
 * so every interaction will require cell-list creation.
 * The cut-off raduis has to be removed later from the interface,
 * such that only certain interactions require cell-lists.
 */
class Interaction : public MirSimulationObject
{
public:
    /// monitor activity of a channel
    using ActivePredicate = std::function<bool()>;

    /**
     * describe the activity of a channel in an interaction
     */
    struct InteractionChannel
    {
        std::string name;
        ActivePredicate active;
    };

    enum class Stage {Intermediate, Final};
    
    Interaction(const MirState *state, std::string name, real rc);

    virtual ~Interaction();

    /**
     * Ask ParticleVectors which the class will be working with to have specific properties
     * Default: ask nothing
     * Called from Simulation right after setup
     */
    virtual void setPrerequisites(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2);
    
    /**
     * Interface to compute local interactions.
     * For now order of \e pv1 and \e pv2 is important for computational reasons,
     * this may be changed later on so that the best order is chosen automatically.
     *
     * @param pv1 first interacting ParticleVector
     * @param pv2 second interacting ParticleVector. If it is the same as
     *            the \p pv1, self interactions will be computed
     * @param cl1 cell-list built for the appropriate cut-off raduis #rc for \p pv1
     * @param cl2 cell-list built for the appropriate cut-off raduis #rc for \p pv2
     */
    virtual void local(ParticleVector *pv1, ParticleVector *pv2,
                       CellList *cl1, CellList *cl2, cudaStream_t stream) = 0;

    /**
     * Interface to compute halo interactions. It principle it has to compute
     * pv1->halo() \<--\> pv2->local() and pv2->halo() \<--\> pv1->local().
     * See InteractionPair for more details
     *
     * @param pv1 first interacting ParticleVector
     * @param pv2 second interacting ParticleVector
     * @param cl1 cell-list built for the appropriate cut-off raduis #rc for \p pv1
     * @param cl2 cell-list built for the appropriate cut-off raduis #rc for \p pv2
     */
    virtual void halo(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1,
                      CellList *cl2, cudaStream_t stream) = 0;


    /**
     * true if the interaction is a self interaction for objects
     * this is useful to know if we need exchange / cell-lists for 
     * that interaction
     */
    virtual bool isSelfObjectInteraction() const;

    virtual Stage getStage() const {return Stage::Final;}
    
    /**
     * describe which channels are required as input
     * default: nothing
     */
    virtual std::vector<InteractionChannel> getInputChannels() const;

    /**
     * describe which channels are output
     * default: forces, always active
     */
    virtual std::vector<InteractionChannel> getOutputChannels() const;


    void checkpoint(MPI_Comm comm, const std::string& path, int checkpointId) override;
    void restart   (MPI_Comm comm, const std::string& path) override;
    
    static const ActivePredicate alwaysActive;
    
public:
    /// Cut-off raduis
    real rc;

protected:

    std::unique_ptr<Interaction> impl; // concrete implementation of interactions
};
