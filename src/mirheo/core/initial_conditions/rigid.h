#pragma once

#include "interface.h"

#include <mirheo/core/datatypes.h>

#include <string>
#include <vector>
#include <vector_types.h>

namespace mirheo
{

/** \brief Initialize RigidObjectVector objects
    \ingroup ICs
    
    Initialize rigid objects from center of mass positions, orientations and frozen particles.
*/
class RigidIC : public InitialConditions
{
public:
    /** \brief Construct a RigidIC object
        \param [in] comQ List of (position, orientation) corresponding to each object.
        The size of the list is the number of rigid objects that will be initialized.
        \param [in] xyzfname The name of a file in xyz format. 
        It contains the list of coordinates of the frozen particles (in the object frame of reference).

        This method will die if the file does not exist.
     */
    RigidIC(const std::vector<ComQ>& comQ, const std::string& xyzfname);

    /** \brief Construct a RigidIC object
        \param [in] comQ List of (position, orientation) corresponding to each object.
        The size of the list is the number of rigid objects that will be initialized.
        \param [in] coords List of positions of the frozen particles of one object, in the object frame of reference.
     */
    RigidIC(const std::vector<ComQ>& comQ, const std::vector<real3>& coords);

    /** \brief Construct a RigidIC object
        \param [in] comQ List of (position, orientation) corresponding to each object.
        The size of the list is the number of rigid objects that will be initialized.
        \param [in] coords List of positions of the frozen particles of one object, in the object frame of reference.
        \param [in] comVelocities List of velocities of the velocities of the objects center of mass.
        Must have the same size as \p comQ.
     */
    RigidIC(const std::vector<ComQ>& comQ, const std::vector<real3>& coords,
            const std::vector<real3>& comVelocities);

    ~RigidIC();

    void exec(const MPI_Comm& comm, ParticleVector *pv, cudaStream_t stream) override;

private:
    std::vector<ComQ> comQ_;
    std::vector<real3> coords_;
    std::vector<real3> comVelocities_;
};

} // namespace mirheo
