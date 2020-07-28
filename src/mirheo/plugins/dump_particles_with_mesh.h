// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include "dump_particles.h"

namespace mirheo
{

/** Send particle data to ParticleWithMeshSenderPlugin.
    Does the same as ParticleSenderPlugin with additional Mesh connectivity information.
    This is compatible only with ObjectVector.
*/
class ParticleWithMeshSenderPlugin : public ParticleSenderPlugin
{
public:
    /** Create a ParticleWithMeshSenderPlugin object.
        \param [in] state The global state of the simulation.
        \param [in] name The name of the plugin.
        \param [in] pvName The name of the ParticleVector to dump.
        \param [in] dumpEvery Send the data to the postprocess side every this number of steps.
        \param [in] channelNames The list of channels to send, additionally to the default positions, velocities and global ids.
     */
    ParticleWithMeshSenderPlugin(const MirState *state, std::string name, std::string pvName, int dumpEvery,
                                 const std::vector<std::string>& channelNames);

    void setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;
    void handshake() override;
};


/** Postprocess side of ParticleWithMeshSenderPlugin.
    Dump particles data with connectivity to xmf + hdf5 format.
*/
class ParticleWithMeshDumperPlugin : public ParticleDumperPlugin
{
public:
    /** Create a ParticleWithMeshDumperPlugin object.
        \param [in] name The name of the plugin.
        \param [in] path Data will be dumped to `pathXXXXX.[xmf,h5]`.
    */
    ParticleWithMeshDumperPlugin(std::string name, std::string path);

    void handshake() override;
    void deserialize() override;

private:
    void _prepareConnectivity(int totNVertices);

private:
    std::shared_ptr<std::vector<int3>> allTriangles_;

    int nvertices_;
    std::vector<int3> triangles_;
};

} // namespace mirheo
