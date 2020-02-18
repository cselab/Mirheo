#pragma once

#include <mirheo/core/mirheo_object.h>

#include <cuda_runtime.h>
#include <mpi.h>
#include <string>
#include <vector>

namespace mirheo
{

class ParticleVector;
class ObjectVector;
class CellList;

/** \brief Mark or split particles which are inside of a given \c ObjectVector.
    
    The user must call setup() exactly once before any call of checkInner() or splitByBelonging(). 
 */
class ObjectBelongingChecker : public MirSimulationObject
{
public:
    /** \brief Construct a \c ObjectBelongingChecker object.
        \param [in] state Simulation state.
        \param [in] name Name of the bouncer.
     */
    ObjectBelongingChecker(const MirState *state, const std::string& name);
    virtual ~ObjectBelongingChecker();

    /** \brief Split a \c ParticleVector into inside and outside particles.
        \param [in] src The particles to split.
        \param [in,out] pvIn Buffer that will contain the inside particles.
        \param [in,out] pvOut Buffer that will contain the outside particles.
        \param [in] stream Stream used for the execution.

        The `pvIn` and `pvOut` \c ParticleVector can be set to `nullptr`, in which case they will be ignored.
        If `pvIn` and `src` point to the same object, `pvIn` will contain only inside particles of `src` in the end.
        Otherwise, `pvIn` will contain its original particles additionally to the inside particles of `src`.
        If `pvOut` and `src` point to the same object, `pvOut` will contain only outside particles of `src` in the end.
        Otherwise, `pvOut` will contain its original particles additionally to the outside particles of `src`.

        This method will die if the type of `pvIn`, `pvOut` and `src` have a different type.

        Must be called after setup().
     */
    virtual void splitByBelonging(ParticleVector *src, ParticleVector *pvIn, ParticleVector *pvOut, cudaStream_t stream) = 0;

    /** \brief Prints number of inside and outside particles in the log as a `Info` entry.
        \param [in] pv The particles to check.
        \param [in] cl Cell lists of pv.
        \param stream Stream used for execution.

        Additionally, this will compute the inside/outside tags of the particles and store it inside this object instance.

        Must be called after setup().
     */
    virtual void checkInner(ParticleVector *pv, CellList *cl, cudaStream_t stream) = 0;

    /** \brief Register the \c ObjectVector that defines inside and outside.
        \param [in] ov The \c ObjectVector to register.
    */
    virtual void setup(ObjectVector *ov) = 0;


    /// \brief Return the channels of the registered \c ObjectVector to be exchanged before splitting.
    virtual std::vector<std::string> getChannelsToBeExchanged() const;

    /// \brief Return the registered \c ObjectVector.
    virtual ObjectVector* getObjectVector() = 0;
};

} // namespace mirheo
