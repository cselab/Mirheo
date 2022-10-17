// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "magnetic_dipole_interactions.h"

#include <mirheo/core/pvs/rigid_object_vector.h>
#include <mirheo/core/pvs/views/rov.h>
#include <mirheo/core/simulation.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/utils/kernel_launch.h>
#include <mirheo/core/utils/quaternion.h>
#include <mirheo/core/utils/mpi_types.h>

#include <mpi.h>

namespace mirheo {
namespace magnetic_dipole_interactions_plugin_kernels {

__global__ void collectRigidInfo(const DomainInfo domain, const ROVview view, real4 *rigidPosQuat)
{
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= view.nObjects) return;

    const auto m = view.motions[gid];

    const real3 r = domain.local2global(make_real3(m.r.x, m.r.y, m.r.z));

    rigidPosQuat[2*gid + 0] = make_real4(r.x, r.y, r.z, 0.0_r);
    rigidPosQuat[2*gid + 1] = make_real4(m.q.w, m.q.x, m.q.y, m.q.z);
}

/** Compute the difference between two points.
    If periodic, returns the smallest difference between two points in the periodic domain.
 */
__device__ inline real3 difference(real3 a, real3 b, real3 L, bool periodic)
{
    real3 d = a - b;

    if (periodic)
    {
        const real3 h = 0.5_r * L;

        if (d.x < -h.x) d.x += L.x;
        if (d.x >  h.x) d.x -= L.x;

        if (d.y < -h.y) d.y += L.y;
        if (d.y >  h.y) d.y -= L.y;

        if (d.z < -h.z) d.z += L.z;
        if (d.z >  h.z) d.z -= L.z;
    }

    return d;
}

__global__ void computeInteractions(const DomainInfo domain, ROVview view,
                                    int numSources,  const real4 *rigidPosQuat,
                                    real mu0_4pi, real3 moment, bool periodic)
{
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= view.nObjects) return;

    const auto m = view.motions[gid];
    const real3 rdst = domain.local2global(make_real3(m.r.x, m.r.y, m.r.z));
    const auto qdst = static_cast<Quaternion<real>>(m.q);
    const auto mdst = qdst.rotate(moment);

    auto force = make_real3(0);
    auto torque = make_real3(0);

    for (int i = 0; i < numSources; ++i)
    {
        const real4 rsrc4 = rigidPosQuat[2*i];
        const real3 rsrc = make_real3(rsrc4.x, rsrc4.y, rsrc4.z);
        const auto qsrc = Quaternion<real>::createFromComponents(rigidPosQuat[2*i+1]);
        const auto msrc = qsrc.rotate(moment);

        const real3 dr = difference(rdst, rsrc, domain.globalSize, periodic);
        const real r2 = dot(dr, dr);

        if (r2 < 1e-6_r)
            continue;


        const real r = math::sqrt(r2);
        const real3 er = dr / r;

        const real forceFactor = 3 * mu0_4pi / (r2 * r2);
        const real torqueFactor = mu0_4pi / (r2 * r);

        const real3 erXmdst = cross(er, mdst);
        const real3 erXmsrc = cross(er, msrc);

        force += forceFactor * (cross(erXmdst, msrc) + cross(erXmsrc, mdst)
                                - 2 * dot(mdst, msrc) * er + 5 * dot(erXmdst, erXmsrc) * er);

        torque += torqueFactor * (3 * dot(mdst, er) * cross(msrc, er) - cross(mdst, msrc));
    }

    atomicAdd(&view.motions[gid].torque.x, static_cast<RigidReal>(torque.x));
    atomicAdd(&view.motions[gid].torque.y, static_cast<RigidReal>(torque.y));
    atomicAdd(&view.motions[gid].torque.z, static_cast<RigidReal>(torque.z));

    atomicAdd(&view.motions[gid].force.x, static_cast<RigidReal>(force.x));
    atomicAdd(&view.motions[gid].force.y, static_cast<RigidReal>(force.y));
    atomicAdd(&view.motions[gid].force.z, static_cast<RigidReal>(force.z));
}
} // namespace magnetic_dipole_interactions_plugin_kernels

MagneticDipoleInteractionsPlugin::MagneticDipoleInteractionsPlugin(const MirState *state, std::string name,
                                                                   std::string rovName,
                                                                   real3 moment, real mu0, bool periodic) :
    SimulationPlugin(state, name),
    rovName_(rovName),
    moment_(moment),
    mu0_(mu0),
    periodic_(periodic)
{}

void MagneticDipoleInteractionsPlugin::setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(simulation, comm, interComm);

    rov_ = dynamic_cast<RigidObjectVector*>( simulation->getOVbyNameOrDie(rovName_) );
    if (rov_ == nullptr)
        die("Need rigid object vector to interact with magnetic field, plugin '%s', OV name '%s'",
            getCName(), rovName_.c_str());

    recvCounts_.resize(nranks_);
    recvDispls_.resize(nranks_ + 1);
}

void MagneticDipoleInteractionsPlugin::beforeCellLists(cudaStream_t stream)
{
    const ROVview view(rov_, rov_->local());
    const int nthreads = 64;

    // share the number of objects to all ranks
    MPI_Request reqNumObjects;
    MPI_Check(MPI_Iallgather(&view.nObjects, 1, MPI_INT,
                             recvCounts_.data(), 1, MPI_INT,
                             comm_, &reqNumObjects));

    // gather the rigid objects positions and orientations
    const auto domain = getState()->domain;
    sendRigidPosQuat_.resize_anew(2 * view.nObjects);

    SAFE_KERNEL_LAUNCH(
        magnetic_dipole_interactions_plugin_kernels::collectRigidInfo,
        getNblocks(view.nObjects, nthreads), nthreads, 0, stream,
        domain, view, sendRigidPosQuat_.devPtr());

    sendRigidPosQuat_.downloadFromDevice(stream, ContainersSynch::Synch);

    MPI_Wait(&reqNumObjects, MPI_STATUS_IGNORE);

    for (auto& v : recvCounts_)
        v *= 2 * sizeof(real4) / sizeof(real);

    recvDispls_[0] = 0;
    for (size_t i = 0; i < recvCounts_.size(); ++i)
        recvDispls_[i+1] = recvDispls_[i] + recvCounts_[i];

    const int totNumObjects = recvDispls_[nranks_] / (2 * sizeof(real4) / sizeof(real));

    recvRigidPosQuat_.resize_anew(2 * totNumObjects);

    // send the object data to all ranks

    const auto dataType = getMPIFloatType<real>();

    MPI_Check(MPI_Iallgatherv(sendRigidPosQuat_.data(), sizeof(real4) / sizeof(real) * sendRigidPosQuat_.size(), dataType,
                              recvRigidPosQuat_.data(), recvCounts_.data(), recvDispls_.data(), dataType,
                              comm_, &reqObjInfo_));

}

void MagneticDipoleInteractionsPlugin::beforeForces(cudaStream_t stream)
{
    MPI_Wait(&reqObjInfo_, MPI_STATUS_IGNORE);
    recvRigidPosQuat_.uploadToDevice(stream);

    const int totNumObjects = recvRigidPosQuat_.size() / 2;

    ROVview view(rov_, rov_->local());
    const int nthreads = 128;

    const auto domain = getState()->domain;

    SAFE_KERNEL_LAUNCH(
            magnetic_dipole_interactions_plugin_kernels::computeInteractions,
            getNblocks(view.nObjects, nthreads), nthreads, 0, stream,
            domain, view, totNumObjects, recvRigidPosQuat_.devPtr(),
            mu0_ / (4 * M_PI), moment_,
            periodic_);
}

} // namespace mirheo
