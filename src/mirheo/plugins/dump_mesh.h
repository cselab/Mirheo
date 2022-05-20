// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/containers.h>
#include <mirheo/core/datatypes.h>
#include <mirheo/core/plugins.h>

#include <vector>

namespace mirheo
{

class ParticleVector;
class ObjectVector;
class CellList;

/** Send mesh information of an object for dump to MeshDumper postprocess plugin.
 */
class MeshPlugin : public SimulationPlugin
{
public:
    /** Create a MeshPlugin object.
        \param [in] state The global state of the simulation.
        \param [in] name The name of the plugin.
        \param [in] ovName The name of the ObjectVector that has a mesh to dump.
        \param [in] dumpEvery Will dump the mesh every this number of timesteps.
     */
    MeshPlugin(const MirState *state, std::string name, std::string ovName, int dumpEvery);

    void setup(Simulation* simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;

    void beforeForces(cudaStream_t stream) override;
    void serializeAndSend(cudaStream_t stream) override;

    bool needPostproc() override { return true; }

private:
    std::string ovName_;
    int dumpEvery_;

    std::vector<char> sendBuffer_;
    std::vector<real3> vertices_;
    PinnedBuffer<real4>* srcVerts_;

    ObjectVector *ov_;
};


/** Postprocess side of MeshPlugin.
    Receives mesh info and dump it to ply format.
*/
class MeshDumper : public PostprocessPlugin
{
public:
    /** Create a MeshDumper object.
        \param [in] name The name of the plugin.
        \param [in] path The files will be dumped to `path-XXXXX.ply`.
     */
    MeshDumper(std::string name, std::string path);

    ~MeshDumper();

    void deserialize() override;
    void setup(const MPI_Comm& comm, const MPI_Comm& interComm) override;

private:
    std::string path_;

    bool activated_{true};

    std::vector<int3> connectivity_;
    std::vector<real3> vertices_;
};

} // namespace mirheo
