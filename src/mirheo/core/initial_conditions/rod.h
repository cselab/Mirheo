#pragma once

#include "interface.h"

#include <mirheo/core/datatypes.h>

#include <functional>
#include <string>
#include <vector>
#include <vector_types.h>

namespace mirheo
{

/** \brief Initialize \c RodVector objects
    \ingroup ICs
    
    All rods will have the same torsion and centerline in there frame of reference.
    Each rod has a specific center of mass and orientation. 
*/
class RodIC : public InitialConditions
{
public:
    static const real Default; ///< default real value, used to pass default parameters
    static const real3 DefaultFrame;  ///< default orientation, used to pass default parameters
    
    using MappingFunc3D = std::function<real3(real)>; ///< a map from [0,1] to R^3
    using MappingFunc1D = std::function<real(real)>;  ///< a map from [0,1] to R


    /** \brief Construct a \c RodIC object
        \param [in] comQ list of center of mass and orientation of each rod. 
        This will determine the number of rods. 
        The rods with center of mass outside of the domain will be discarded.
        \param centerLine Function describing the centerline in the frame of reference of the rod
        \param torsion Function describing the torsion along the centerline.
        \param a The width of the rod (the cross particles are separated by \p a).
        \param initialMaterialFrame If set, this describes the orientation  of the local material frame at the start of the rod (in the object frame of reference).
        If not set, this is chosen arbitrarily.
    */
    RodIC(const std::vector<ComQ>& comQ, MappingFunc3D centerLine, MappingFunc1D torsion, real a,
          real3 initialMaterialFrame = DefaultFrame);
    ~RodIC();
    
    void exec(const MPI_Comm& comm, ParticleVector *pv, cudaStream_t stream) override;    

private:
    std::vector<ComQ> comQ_;
    MappingFunc3D centerLine_;
    MappingFunc1D torsion_;
    real3 initialMaterialFrame_;
    real a_;
};

} // namespace mirheo
