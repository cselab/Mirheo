// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "membrane_extra_force.h"

#include <mirheo/core/pvs/membrane_vector.h>
#include <mirheo/core/pvs/views/ov.h>
#include <mirheo/core/simulation.h>
#include <mirheo/core/snapshot.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/utils/folders.h>
#include <mirheo/core/utils/kernel_launch.h>

#include <fstream>

namespace mirheo
{

namespace membrane_extra_forces_kernels
{
__global__ void addForce(OVview view, const Force *forces)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= view.size) return;

    int locId = gid % view.objSize;

    view.forces[gid] += forces[locId].toReal4();
}
} // namespace membrane_extra_forces_kernels

/// Read forces from the file written by `writeForces`.
static std::vector<real3> readForces(const std::string& filename)
{
    std::ifstream f(filename);
    if (!f.good())
        die("Error opening file %s.\n", filename.c_str());
    size_t size;
    f >> size;
    std::vector<real3> forces(size);
    for (real3 &force : forces)
        f >> force.x >> force.y >> force.z;
    return forces;
}

/// Store the forces in a file.
static void writeForces(const std::string& filename, const Force *forces, size_t size)
{
    // Minimum number of digits required to losslessly store a floating point number.
    constexpr int digits = std::numeric_limits<decltype(forces[0].f.x)>::max_digits10;
    FileWrapper f(filename, "w");
    fprintf(f.get(), "%zu\n", size);
    for (size_t i = 0; i < size; ++i) {
        fprintf(f.get(), "%.*g %.*g %.*g\n",
                digits, forces[i].f.x,
                digits, forces[i].f.y,
                digits, forces[i].f.z);
    }
}

MembraneExtraForcePlugin::MembraneExtraForcePlugin(const MirState *state, std::string name, std::string pvName, const std::vector<real3>& forces) :
    SimulationPlugin(state, name),
    pvName_(pvName),
    forces_(forces.size())
{
    HostBuffer<Force> hostForces(forces.size());

    for (size_t i = 0; i < forces.size(); ++i)
        hostForces[i].f = forces[i];

    forces_.copy(hostForces, defaultStream);
}

MembraneExtraForcePlugin::MembraneExtraForcePlugin(
        const MirState *state, Loader& loader, const ConfigObject& config) :
    MembraneExtraForcePlugin{
            state, config["name"].getString(), config["pvName"].getString(),
            readForces(joinPaths(loader.getContext().getPath(), config["name"] + ".dat"))}
{}

void MembraneExtraForcePlugin::setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(simulation, comm, interComm);

    auto pvPtr = simulation->getPVbyNameOrDie(pvName_);
    if ( !(pv_ = dynamic_cast<MembraneVector*>(pvPtr)) )
        die("MembraneExtraForcePlugin '%s' expects a MembraneVector (given '%s')",
            getCName(), pvName_.c_str());
}

void MembraneExtraForcePlugin::beforeForces(cudaStream_t stream)
{
    OVview view(pv_, pv_->local());
    const int nthreads = 128;

    SAFE_KERNEL_LAUNCH(
        membrane_extra_forces_kernels::addForce,
        getNblocks(view.size, nthreads), nthreads, 0, stream,
        view, forces_.devPtr() );
}

void MembraneExtraForcePlugin::saveSnapshotAndRegister(Saver& saver)
{
    saver.registerObject<MembraneExtraForcePlugin>(
            this, _saveSnapshot(saver, "MembraneExtraForcePlugin"));
}

ConfigObject MembraneExtraForcePlugin::_saveSnapshot(Saver& saver, const std::string& typeName)
{
    HostBuffer<Force> hostForces;
    hostForces.copy(forces_, defaultStream);
    writeForces(joinPaths(saver.getContext().path, getName() + ".dat"),
                hostForces.data(), hostForces.size());

    ConfigObject config = SimulationPlugin::_saveSnapshot(saver, typeName);
    config.emplace("pvName", saver(pvName_));
    return config;
}

} // namespace mirheo
