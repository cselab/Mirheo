#pragma once

#include <mirheo/core/utils/helper_math.h>
#include <mirheo/core/utils/cpu_gpu_defines.h>

#include <mpi.h>
#include <vector_types.h>

namespace mirheo
{

/** \brief Describes the simulation global and local domains, with a mapping between the two.

    The simulation domain is a rectangular box.
    It is splitted into smaller rectangles, one by simulation rank.
    Each of these subdomains have a local system of coordinates, centered at the center of these rectangular boxes.
    The global system of coordinate has the lowest corner of the domain at (0,0,0).
 */
struct DomainInfo
{
    real3 globalSize;  ///< Size of the whole simulation domain
    real3 globalStart; ///< coordinates of the lower corner of the local domain, in global coordinates
    real3 localSize;   ///< size of the sub domain in the current rank.

    /** \brief Convert local coordinates to global coordinates
        \param [in] x The local coordinates in the current subdomain 
        \return The position \p x expressed in global coordinates
     */
    inline __HD__ real3 local2global(real3 x) const
    {
        return x + globalStart + 0.5_r * localSize;
    }

    /** \brief Convert global coordinates to local coordinates
        \param [in] x The global coordinates in the simulation domain 
        \return The position \p x expressed in local coordinates
     */
    inline __HD__ real3 global2local(real3 x) const
    {
        return x - globalStart - 0.5_r * localSize;
    }

    /** \brief Checks if the global coordinates \p xg are inside the current subdomain
        \param [in] xg The global coordinates in the simulation domain 
        \return \c true if \p xg is inside the current subdomain, \c false otherwise
     */
    template <typename RealType3>
    inline __HD__ bool inSubDomain(RealType3 xg) const
    {
        return (globalStart.x <= xg.x) && (xg.x < (globalStart.x + localSize.x))
            && (globalStart.y <= xg.y) && (xg.y < (globalStart.y + localSize.y))
            && (globalStart.z <= xg.z) && (xg.z < (globalStart.z + localSize.z));
    }    
};

/** \brief Construct a DomainInfo 
    \param [in] cartComm A cartesian MPI communicator of the simulation
    \param [in] globalSize The size of the whole simulation domain
    \return The DomainInfo
 */
DomainInfo createDomainInfo(MPI_Comm cartComm, real3 globalSize);

} // namespace mirheo
