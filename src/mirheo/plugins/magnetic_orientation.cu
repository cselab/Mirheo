#include "magnetic_orientation.h"

#include <mirheo/core/pvs/rigid_object_vector.h>
#include <mirheo/core/pvs/views/rov.h>
#include <mirheo/core/simulation.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/utils/kernel_launch.h>
#include <mirheo/core/utils/quaternion.h>

namespace mirheo
{

namespace MagneticOrientationPluginKernels
{
__global__ void applyMagneticField(ROVview view, real3 B, real3 M)
{
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= view.nObjects) return;

    const auto q = static_cast<Quaternion<real>>(view.motions[gid].q);

    M = q.rotate(M);

    const real3 T = cross(M, B);
    
    atomicAdd(&view.motions[gid].torque.x, static_cast<RigidReal>(T.x));
    atomicAdd(&view.motions[gid].torque.y, static_cast<RigidReal>(T.y));
    atomicAdd(&view.motions[gid].torque.z, static_cast<RigidReal>(T.z));
}
} // namespace MagneticOrientationPluginKernels

MagneticOrientationPlugin::MagneticOrientationPlugin(const MirState *state, std::string name, std::string rovName,
                                                     real3 moment, UniformMagneticFunc magneticFunction) :
    SimulationPlugin(state, name),
    rovName(rovName),
    moment(moment),
    magneticFunction(magneticFunction)
{}

void MagneticOrientationPlugin::setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(simulation, comm, interComm);

    rov = dynamic_cast<RigidObjectVector*>( simulation->getOVbyNameOrDie(rovName) );
    if (rov == nullptr)
        die("Need rigid object vector to interact with magnetic field, plugin '%s', OV name '%s'",
            name.c_str(), rovName.c_str());
}

void MagneticOrientationPlugin::beforeForces(cudaStream_t stream)
{
    ROVview view(rov, rov->local());
    const int nthreads = 128;

    const auto t = state->currentTime;
    const auto B = magneticFunction(t);
    
    SAFE_KERNEL_LAUNCH(
            MagneticOrientationPluginKernels::applyMagneticField,
            getNblocks(view.size, nthreads), nthreads, 0, stream,
            view, B, moment);
}

} // namespace mirheo
