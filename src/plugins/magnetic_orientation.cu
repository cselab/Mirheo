#include "magnetic_orientation.h"

#include <core/utils/kernel_launch.h>
#include <core/pvs/rigid_object_vector.h>
#include <core/pvs/views/rov.h>
#include <core/simulation.h>


#include <core/utils/cuda_common.h>
#include <core/rigid_kernels/quaternion.h>

__global__ void applyMagneticField(ROVview view, float3 B, float3 M)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= view.nObjects) return;

    auto q = view.motions[gid].q;

    M = rotate(M, q);

    float3 T = cross(M, B);
    
    view.motions[gid].torque += T;
}


MagneticOrientationPlugin::MagneticOrientationPlugin(const YmrState *state, std::string name, std::string rovName,
                                                     float3 moment, UniformMagneticFunc magneticFunction) :
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

    auto t = state->currentTime;
    float3 B = magneticFunction(t);
    
    SAFE_KERNEL_LAUNCH(
            applyMagneticField,
            getNblocks(view.size, nthreads), nthreads, 0, stream,
            view, B, moment);
}

