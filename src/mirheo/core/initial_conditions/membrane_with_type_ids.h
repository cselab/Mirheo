#pragma once

#include "membrane.h"

namespace mirheo
{

/** \brief Initialize MembraneVector objects with a typeId
    \ingroup ICs
    
    \see MembraneIC
    Attach an additional typeId field to each membrane.
    This is useful to have different membrane forces without having many MembraneVector objects.
*/
class MembraneWithTypeIdsIC : public MembraneIC
{
public:
    /** \brief Construct a MembraneWithTypeIdsIC object
        \param [in] comQ List of (position, orientation) corresponding to each object.
        The size of the list is the number of membrane objects that will be initialized.
        \param [in] typeIds List of type Ids. must have the same size as \p comQ.
        \param [in] globalScale scale the membranes by this scale when placing the initial vertices.
    */
    MembraneWithTypeIdsIC(const std::vector<ComQ>& comQ, const std::vector<int>& typeIds, real globalScale = 1.0);
    ~MembraneWithTypeIdsIC();

    void exec(const MPI_Comm& comm, ParticleVector *pv, cudaStream_t stream) override;

private:
    std::vector<int> typeIds_;
};

} // namespace mirheo
