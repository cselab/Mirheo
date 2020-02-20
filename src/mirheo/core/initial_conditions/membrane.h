#pragma once

#include "interface.h"

#include <mirheo/core/datatypes.h>
#include <mirheo/core/domain.h>

#include <mpi.h>
#include <string>
#include <vector>
#include <vector_types.h>

namespace mirheo
{

class ParticleVector;

/** \brief Initialize MembraneVector objects
    
    Initialize membrane objects from center of mass positions and orientations.
*/
class MembraneIC : public InitialConditions
{
public:
    /** \brief Construct a MembraneIC object
        \param [in] comQ List of (position, orientation) corresponding to each object.
        The size of the list is the number of membrane objects that will be initialized.
        \param [in] globalScale scale the membranes by this scale when placing the initial vertices.
    */
    MembraneIC(const std::vector<ComQ>& comQ, real globalScale = 1.0);
    ~MembraneIC();

    void exec(const MPI_Comm& comm, ParticleVector *pv, cudaStream_t stream) override;

protected:
    /** \brief create a list which contains the indices of all membranes in the current subdomain.
        \param [in] domain Domain information

        The indices correspond to the indices of the comQ_ member variable.
    */
    std::vector<int> createMap(DomainInfo domain) const;
    
private:
    std::vector<ComQ> comQ_;
    real globalScale_;
};

} // namespace mirheo
