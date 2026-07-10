// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "copy_pv.h"

#include <mirheo/core/pvs/particle_vector.h>
#include <mirheo/core/pvs/views/pv.h>
#include <mirheo/core/simulation.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/utils/kernel_launch.h>

namespace mirheo
{


CopyPVPlugin::CopyPVPlugin(const MirState *state, const std::string& name, const std::string& pvTargetName, const std::string& pvSourceName) :
    SimulationPlugin(state, name),
    pvTargetName_(pvTargetName),
    pvSourceName_(pvSourceName)
{}

void CopyPVPlugin::setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(simulation, comm, interComm);

    pvTarget_ = simulation->getPVbyNameOrDie(pvTargetName_);
    pvSource_ = simulation->getPVbyNameOrDie(pvSourceName_);

    pvTarget_->local()->dataPerParticle.copyChannelMap(pvSource_->local()->dataPerParticle);
}

void CopyPVPlugin::beforeCellLists(__UNUSED cudaStream_t stream)
{
    *pvTarget_->local() = *pvSource_->local();    //This implementation is not the best ever, as it reallocates the buffers. We can optimize it later if needed.
}

} // namespace mirheo
