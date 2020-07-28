// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/containers.h>
#include <mirheo/core/plugins.h>

#include <memory>
#include <string>

namespace mirheo
{

class ParticleVector;
class ParticlePacker;

/** Transfer particles from one ParticleVector to another when they cross a given plane.
 */
class ExchangePVSFluxPlanePlugin : public SimulationPlugin
{
public:
    /** Create a \c ExchangePVSFluxPlanePlugin object.
        \param [in] state The global state of the simulation.
        \param [in] name The name of the plugin.
        \param [in] pv1Name The name of the source ParticleVector. Only particles from this ParticleVector are transfered.
        \param [in] pv2Name The name of the destination ParticleVector.
        \param [in] plane Coefficients of the plane to be crossed, (a, b, c, d).

        The particle has crossed the plane if a *x + b * y + c * z + d goes from negative to positive.
     */
    ExchangePVSFluxPlanePlugin(const MirState *state, std::string name, std::string pv1Name, std::string pv2Name, real4 plane);
    ~ExchangePVSFluxPlanePlugin();

    void setup(Simulation* simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;
    void beforeCellLists(cudaStream_t stream) override;

    bool needPostproc() override { return false; }

private:
    std::string pv1Name_, pv2Name_;
    ParticleVector *pv1_, *pv2_;
    real4 plane_;

    PinnedBuffer<int> numberCrossedParticles_;
    std::unique_ptr<ParticlePacker> extra1_, extra2_;
};

} // namespace mirheo
