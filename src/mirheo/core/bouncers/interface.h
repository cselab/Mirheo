#pragma once

#include <mirheo/core/mirheo_object.h>

#include <cuda_runtime.h>
#include <mpi.h>
#include <string>
#include <vector>

namespace mirheo
{

class CellList;
class ParticleVector;
class ObjectVector;
enum class ParticleVectorLocality;

/** \brief Avoid penetration of particles inside onjects

    Interface class for Bouncers.
    Bouncers are responsible to reflect particles on the surface of the attached object.
    Each \c Bouncer class needs to attach exactly one \c ObjectVector before performing the bounce.
 */
class Bouncer : public MirSimulationObject
{
public:
    /** \brief Base \c Bouncer constructor.
        \param [in] state Simulation state.
        \param [in] name Name of the bouncer.
     */
    Bouncer(const MirState *state, std::string name);
    virtual ~Bouncer();

    /** \brief Second initialization stage
        \param [in] ov The \c ObjectVector to attach to that \c Bouncer.

        This method must be called before calling any other method of this class.
     */
    virtual void setup(ObjectVector *ov);

    /**
       \return The attached \c ObjectVector
     */
    ObjectVector* getObjectVector();
    
    /** \brief Setup prerequisites to a given \c ParticleVector
        \param [in,out] pv The \c ParticleVector that will be bounced

        Add additional properties to a \c ParticleVector to make it compatible with the exec() method.
        The default implementation does not add any properties.
     */
    virtual void setPrerequisites(ParticleVector *pv);

    /** \brief Perform the reflection of local particles onto the **local** attached objects surface.
        \param [in,out] pv The \c ParticleVector that will be bounced
        \param [in] cl The \c CellList attached to \p pv
        \param [in] stream The cuda stream used for execution
     */
    void bounceLocal(ParticleVector *pv, CellList *cl, cudaStream_t stream);

    /** \brief Perform the reflection of local particles onto the **halo** attached objects surface.
        \param [in,out] pv The \c ParticleVector that will be bounced
        \param [in] cl The \c CellList attached to \p pv
        \param [in] stream The cuda stream used for execution
     */
    void bounceHalo (ParticleVector *pv, CellList *cl, cudaStream_t stream);

    /**
        \return list of channel names of the attached object needed before bouncing
     */
    virtual std::vector<std::string> getChannelsToBeExchanged() const = 0;

    /**
        \return list of channel names of the attached object that need to be exchanged after bouncing
     */
    virtual std::vector<std::string> getChannelsToBeSentBack() const;

protected:
    ObjectVector *ov_;  ///< Attached \c ObjectVector. The particles will be bounced against its surface

    /** \brief Driver to execute bouncing
        \param [in,out] pv The \c ParticleVector whose particles will be bounced from \ref ov_.
        \param [in] cl The \c CellList associated to \p pv.
        \param [in] locality State if the bouncing is performed against **local** or **halo** objects.
        \param stream The cuda stream used for execution.

        \rst
        .. note::
            Particles from \p pv (that actually will be bounced back) are always local

        .. note::
            This method will generally also modify \ref ov_ (it might add forces and torque to it)
        \endrst
    */
    virtual void exec(ParticleVector *pv, CellList *cl, ParticleVectorLocality locality, cudaStream_t stream) = 0;
};

} // namespace mirheo
