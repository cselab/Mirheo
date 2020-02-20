#pragma once

#include "domain.h"
#include "utils/common.h"

#include <memory>
#include <mpi.h>
#include <string>

namespace mirheo
{

/** \brief Global quantities accessible by all simulation objects in Mirheo
 */
class MirState
{
public:
    using TimeType = double; ///< type used to store time information
    using StepType = long long; ///< type to store time step information

    /** \brief Construct a MirState object
        \param [in] domain The DomainInfo of the simulation
        \param [in] dt Simulation time step
        \param [in] state If not \nullptr, will set the current time info from snapshot info
    */
    MirState(DomainInfo domain, real dt, const ConfigValue *state = nullptr);

    /// copy constructor
    MirState(const MirState&);
    /// assignment operator
    MirState& operator=(MirState other);

    virtual ~MirState();

    /// swap the content of this object with that of \p other
    void swap(MirState& other);

    /** Save internal state to file
        \param [in] comm MPI comm of the simulation 
        \param [in] path The directory in which to save the file
     */
    void checkpoint(MPI_Comm comm, std::string path);

    /** Load internal state from file
        \param [in] comm MPI comm of the simulation 
        \param [in] path The directory from which to load the file
     */
    void restart(MPI_Comm comm, std::string path);

public:
    DomainInfo domain; ///< Global DomainInfo

    real dt; ///< time step
    TimeType currentTime; ///< Current simulation time
    StepType currentStep; ///< Current simulation step
};

template <>
struct TypeLoadSave<MirState>
{
    static ConfigValue save(Saver&, MirState& state);
};

} // namespace mirheo
