// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/containers.h>
#include <mirheo/core/plugins.h>

#include <string>
#include <vector>

namespace mirheo
{

class RigidObjectVector;

/** Compute the magnetic dipole-dipole forces and torques induced by the
    interactions between rigid objects that have a magnetic moment.
 */
class MagneticDipoleInteractionsPlugin : public SimulationPlugin
{
public:

    /** Create a MagneticDipoleInteractionsPlugin object.
        \param [in] state The global state of the simulation.
        \param [in] name The name of the plugin.
        \param [in] rovName The name of the RigidObjectVector interacting.
        \param [in] moment The constant magnetic moment of one object, in its frame of reference.
        \param [in] mu0 The magnetic permeability of the medium.
     */
     MagneticDipoleInteractionsPlugin(const MirState *state, std::string name,
                                      std::string rovName, real3 moment, real mu0);

    void setup(Simulation* simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;

    void beforeCellLists(cudaStream_t stream) override;
    void beforeForces(cudaStream_t stream) override;

    bool needPostproc() override { return false; }

private:
    std::string rovName_;
    RigidObjectVector *rov_;
    real3 moment_;
    real mu0_;

    PinnedBuffer<real4> sendRigidPosQuat_;
    PinnedBuffer<real4> recvRigidPosQuat_;
    std::vector<int> recvCounts_;
    std::vector<int> recvDispls_;

    MPI_Request reqObjInfo_;
};

} // namespace mirheo
