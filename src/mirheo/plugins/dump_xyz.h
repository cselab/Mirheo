// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/containers.h>
#include <mirheo/core/datatypes.h>
#include <mirheo/core/plugins.h>

#include <vector>

namespace mirheo
{

class ParticleVector;
class CellList;


/** Send particle positions to \c XYZDumper.
*/
class XYZPlugin : public SimulationPlugin
{
public:
    /** Create a \c XYZPlugin object.
        \param [in] state The global state of the simulation.
        \param [in] name The name of the plugin.
        \param [in] pvName The name of the ParticleVector to dump.
        \param [in] dumpEvery Send the data to the postprocess side every this number of steps.
     */
    XYZPlugin(const MirState *state, std::string name, std::string pvName, int dumpEvery);

    void setup(Simulation* simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;

    void beforeForces(cudaStream_t stream) override;
    void serializeAndSend(cudaStream_t stream) override;

    bool needPostproc() override { return true; }

private:
    std::string pvName_;
    int dumpEvery_;
    std::vector<char> sendBuffer_;
    ParticleVector *pv_;
    HostBuffer<real4> positions_;
};

/** Postprocess side of \c XYZPlugin.
    Dump the particle positions to simple .xyz format.
*/
class XYZDumper : public PostprocessPlugin
{
public:
    /** Create a \c XYZDumper object.
        \param [in] name The name of the plugin.
        \param [in] path Data will be dumped to `pathXXXXX.xyz`.
     */
    XYZDumper(std::string name, std::string path);
    ~XYZDumper();

    void deserialize() override;
    void setup(const MPI_Comm& comm, const MPI_Comm& interComm) override;

private:
    std::string path_;
    bool activated_ {true};
    std::vector<real4> pos_;
};

} // namespace mirheo
