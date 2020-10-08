// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "mirheo_state.h"

#include <mirheo/core/logger.h>
#include <mirheo/core/utils/restart_helpers.h>
#include <mirheo/core/utils/config.h>

namespace mirheo
{

static const std::string fname = "state.mirheo";

ConfigValue TypeLoadSave<UnitConversion>::save(Saver& saver, const UnitConversion& units)
{
    return ConfigValue::Object{
        {"toMeters",    saver(units.toMeters_)},
        {"toSeconds",   saver(units.toSeconds_)},
        {"toKilograms", saver(units.toKilograms_)},
    };
}

UnitConversion TypeLoadSave<UnitConversion>::parse(const ConfigValue& config) {
    return UnitConversion{config["toMeters"], config["toSeconds"], config["toKilograms"]};
}

MirState::MirState(DomainInfo domain_, real dt, UnitConversion units_,
                   const ConfigValue *state) :
    domain(domain_),
    currentTime(0),
    currentStep(0),
    units(units_),
    dt_(dt)
{
    if (state) {
        currentTime = (*state)["currentTime"];
        currentStep = (*state)["currentStep"];
    }
}

static bool isMasterRank(MPI_Comm comm)
{
    int rank;
    MPI_Check( MPI_Comm_rank(comm, &rank) );
    return rank == 0;
}

void MirState::checkpoint(MPI_Comm comm, std::string folder)
{
    if (!isMasterRank(comm))
        return;

    real3 gsz, gst, lsz;
    gsz = domain.globalSize;
    gst = domain.globalStart;
    lsz = domain.localSize;

    text_IO::write(folder + fname,
                  gsz.x, gsz.y, gsz.z,
                  gst.x, gst.y, gst.z,
                  lsz.x, lsz.y, lsz.z,
                  dt_, currentTime, currentStep);
}

void MirState::restart(MPI_Comm comm, std::string folder)
{
    if (!isMasterRank(comm))
        return;

    real3 gsz, gst, lsz;
    auto filename = folder + fname;
    auto good = text_IO::read(filename,
                             gsz.x, gsz.y, gsz.z,
                             gst.x, gst.y, gst.z,
                             lsz.x, lsz.y, lsz.z,
                             dt_, currentTime, currentStep);

    if (!good) die("failed to read '%s'\n", filename.c_str());

    domain.globalSize  = gsz;
    domain.globalStart = gst;
    domain.localSize   = lsz;
}

void MirState::_dieInvalidDt [[noreturn]]() const {
    die("Time step dt not available. Using dt is valid only during Mirheo::run().");
}

ConfigValue TypeLoadSave<MirState>::save(Saver& saver, MirState& state)
{
    return ConfigValue::Object{
        {"__category",        saver("MirState")},
        {"__type",            saver("MirState")},
        {"domainGlobalStart", saver(state.domain.globalStart)},
        {"domainGlobalSize",  saver(state.domain.globalSize)},
        {"dt",                saver(state.dt_)},
        {"currentTime",       saver(state.currentTime)},
        {"currentStep",       saver(state.currentStep)},
        {"units",             saver(state.units)},
    };
}

} // namespace mirheo
