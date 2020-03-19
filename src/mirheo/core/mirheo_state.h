#pragma once

#include "domain.h"
#include "utils/common.h"

#include <memory>
#include <mpi.h>
#include <string>

namespace mirheo
{

/** \brief Unit conversion between Mirheo and SI units.

    The conversion factors are optional. Mirheo may run without having this information.
  */
class UnitConversion {
public:
    /// Default constructor, used when conversion factors are not known.
    UnitConversion() noexcept :
        toMeters_{0}, toSeconds_{0}, toKilograms_{0}
    {}

    /// Construct with conversion factors from Mirheo units to SI units.
    UnitConversion(real l2m, real t2s, real m2kg) noexcept :
        toMeters_{l2m}, toSeconds_{t2s}, toKilograms_{m2kg}
    {}

    /// Returns true if the unit conversion factors are known.
    bool isSet() const noexcept
    {
        return toMeters_ != 0 && toSeconds_ != 0 && toKilograms_ != 0;
    }

    /// Convert a length value given in mirheo units to meters.
    real toMeters(real length) const noexcept
    {
        return length * toMeters_;
    }
    /// Convert a duration value given in mirheo units to seconds.
    real toSeconds(real time) const noexcept
    {
        return time * toSeconds_;
    }
    /// Convert a mass value given in mirheo units to kilograms.
    real toKilograms(real mass) const noexcept
    {
        return mass * toKilograms_;
    }

    /// Convert a value in joules to corresponding mirheo units.
    real joulesToMirheo(real energy) const noexcept
    {
        return energy * (toSeconds_ * toSeconds_)
             / (toKilograms_ * toMeters_ * toMeters_);
    }

private:
    friend TypeLoadSave<UnitConversion>;

    real toMeters_;    ///< 1 mirLength == toMeters * m
    real toSeconds_;   ///< 1 mirTime   == toSeconds * s
    real toKilograms_; ///< 1 mirMass   == toKilograms * kg
};

/// Specialization for `UnitConversion`.
template <>
struct TypeLoadSave<UnitConversion>
{
    /// Save UnitConversion as a ConfigValue.
    static ConfigValue save(Saver&, const UnitConversion& value);

    /// Context-free parsing. Create a UnitConversion from a ConfigValue.
    static UnitConversion parse(const ConfigValue& config);
};

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
        \param [in] state If not \c nullptr, will set the current time info from snapshot info
    */
    MirState(DomainInfo domain, real dt, UnitConversion units, const ConfigValue *state = nullptr);


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
    UnitConversion units; ///< Conversion between Mirheo and SI units (optional).
};

/// template specialization struct to implement snapshot save
template <>
struct TypeLoadSave<MirState>
{
    /// save a MirState object to snapshot object
    static ConfigValue save(Saver&, MirState& state);
};

} // namespace mirheo
