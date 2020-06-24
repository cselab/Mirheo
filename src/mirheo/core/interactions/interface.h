// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/mirheo_object.h>

#include <cuda_runtime.h>
#include <functional>
#include <memory>
#include <mpi.h>
#include <vector>

namespace mirheo
{

class CellList;
class ParticleVector;

/** \brief Compute forces from particle interactions.

    We distinguish two kinds of interactions (see Stage enum type):
    1. Intermediate ones, that do not compute any force, but compute intermediate quantities (e.g. densities in SDPD).
    2. Final ones, that compute forces (and possibly other quantities, e.g. stresses).
 */
class Interaction : public MirSimulationObject
{
public:
    /** \brief Used to specify if a channel is active or not.

        If a channel is inactive, the Interaction object can
        tell the simulation via this function object that the
        conserned channel does not need to be exchanged.

        Typically, this can store the simulation state and be
        active only at given time intervals. The most common
        case is to be always active.
     */
    using ActivePredicate = std::function<bool()>;

    /// \brief A simple structure used to describe which  channels are active.
    struct InteractionChannel
    {
        std::string name; ///< the name of the channel
        ActivePredicate active; ///< the activity of the channel
    };

    /// Describes the stage of an interaction
    enum class Stage {Intermediate, Final};

    /** \brief Constructs a \c Interaction object
        \param [in] state The global state of the system
        \param [in] name The name of the interaction
     */
    Interaction(const MirState *state, std::string name);

    /** \brief Constructs a \c Interaction object from a snapshot.
        \param [in] state The global state of the system
        \param [in] loader The \c Loader object. Provides load context and unserialization functions.
        \param [in] config The parameters of the interaction.
     */
    Interaction(const MirState *state, Loader& loader, const ConfigObject& config);

    virtual ~Interaction();

    /** \brief Add needed properties to the given ParticleVectors for future interactions.
        \param [in] pv1 One ParticleVector of the interaction
        \param [in] pv2 The other ParticleVector of that will interact
        \param [in] cl1 CellList of pv1
        \param [in] cl2 CellList of pv2

        Must be called before any other method of this class.
     */
    virtual void setPrerequisites(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2);

    /** \brief Compute interactions between bulk particles.
        \param [in,out] pv1 first interacting ParticleVector
        \param [in,out] pv2 second interacting ParticleVector. If it is the same as
               the pv1, self interactions will be computed.
        \param [in] cl1 cell-list built for the appropriate cut-off radius for pv1
        \param [in] cl2 cell-list built for the appropriate cut-off radius for pv2
        \param [in] stream Execution stream

        The result of the interaction is **added** to the corresponding channel of the ParticleVector.
        The order of pv1 and pv2 may change the performance of the interactions.
     */
    virtual void local(ParticleVector *pv1, ParticleVector *pv2,
                       CellList *cl1, CellList *cl2, cudaStream_t stream) = 0;

    /** \brief Compute interactions between bulk particles and halo particles.
        \param [in,out] pv1 first interacting ParticleVector
        \param [in,out] pv2 second interacting ParticleVector. If it is the same as
               the pv1, self interactions will be computed.
        \param [in] cl1 cell-list built for the appropriate cut-off radius for pv1
        \param [in] cl2 cell-list built for the appropriate cut-off radius for pv2
        \param [in] stream Execution stream

        The result of the interaction is **added** to the corresponding channel of the ParticleVector.
        In general, the following interactions will be computed:
        pv1->halo() \<--\> pv2->local() and pv2->halo() \<--\> pv1->local().
     */
    virtual void halo(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1,
                      CellList *cl2, cudaStream_t stream) = 0;


    /** \return boolean describing if the interaction is an internal interaction.

        This is useful to know if we need exchange / cell-lists for  that interaction.
        Example: membrane interactions are internal, all particles of a membrane are always
        on the same rank thus it does not need halo particles.
     */
    virtual bool isSelfObjectInteraction() const;

    /// returns the Stage corresponding of this interaction.
    virtual Stage getStage() const {return Stage::Final;}

    /** Returns which channels are required as input.
        We consider that positions and velocities are always available;
        Only other channels must be specified here.
     */
    virtual std::vector<InteractionChannel> getInputChannels() const;

    /// Returns which channels are output by the interactions.
    virtual std::vector<InteractionChannel> getOutputChannels() const;

    /// \return the cut-off radius of the interaction
    virtual real getCutoffRadius() const;

    /// a predicate that always returns true.
    static const ActivePredicate alwaysActive;

protected:
    /// Base snapshot function for interactions, sets the category to "Interaction".
    ConfigObject _saveSnapshot(Saver& saver, const std::string &typeName);
};

} // namespace mirheo
