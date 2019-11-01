#include "dump_xyz.h"
#include "utils/simple_serializer.h"
#include "utils/time_stamp.h"
#include "utils/xyz.h"

#include <mirheo/core/pvs/particle_vector.h>
#include <mirheo/core/simulation.h>
#include <mirheo/core/utils/folders.h>

XYZPlugin::XYZPlugin(const MirState *state, std::string name, std::string pvName, int dumpEvery) :
    SimulationPlugin(state, name), pvName(pvName),
    dumpEvery(dumpEvery)
{}

void XYZPlugin::setup(Simulation* simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(simulation, comm, interComm);

    pv = simulation->getPVbyNameOrDie(pvName);

    info("Plugin %s initialized for the following particle vector: %s", name.c_str(), pvName.c_str());
}

void XYZPlugin::beforeForces(cudaStream_t stream)
{
    if (!isTimeEvery(state, dumpEvery)) return;

    positions.copy(pv->local()->positions(), stream);
}

void XYZPlugin::serializeAndSend(__UNUSED cudaStream_t stream)
{
    if (!isTimeEvery(state, dumpEvery)) return;

    debug2("Plugin %s is sending now data", name.c_str());

    for (auto& r : positions)
    {
        auto r3 = make_real3(r);
        r3 = state->domain.local2global(r3);
        r.x = r3.x; r.y = r3.y; r.z = r3.z;
    }

    MirState::StepType timeStamp = getTimeStamp(state, dumpEvery);
    
    waitPrevSend();
    SimpleSerializer::serialize(sendBuffer, timeStamp, pv->name, positions);
    send(sendBuffer);
}

//=================================================================================

XYZDumper::XYZDumper(std::string name, std::string path) :
    PostprocessPlugin(name),
    path(makePath(path))
{}

void XYZDumper::setup(const MPI_Comm& comm, const MPI_Comm& interComm)
{
    PostprocessPlugin::setup(comm, interComm);
    activated = createFoldersCollective(comm, path);
}

void XYZDumper::deserialize()
{
    std::string pvName;
    MirState::StepType timeStamp;
    
    SimpleSerializer::deserialize(data, timeStamp, pvName, pos);

    std::string currentFname = path + pvName + "_" + getStrZeroPadded(timeStamp) + ".xyz";

    if (activated)
        writeXYZ(comm, currentFname, pos.data(), pos.size());
}



