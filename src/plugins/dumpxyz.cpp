#include "dumpxyz.h"
#include "utils/simple_serializer.h"
#include "utils/time_stamp.h"
#include "utils/xyz.h"

#include <core/pvs/particle_vector.h>
#include <core/simulation.h>
#include <core/utils/folders.h>

XYZPlugin::XYZPlugin(const YmrState *state, std::string name, std::string pvName, int dumpEvery) :
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

void XYZPlugin::serializeAndSend(cudaStream_t stream)
{
    if (!isTimeEvery(state, dumpEvery)) return;

    debug2("Plugin %s is sending now data", name.c_str());

    for (auto& r : positions)
    {
        auto r3 = make_float3(r);
        r3 = state->domain.local2global(r3);
        r.x = r3.x; r.y = r3.y; r.z = r3.z;
    }

    waitPrevSend();
    SimpleSerializer::serialize(sendBuffer, pv->name, positions);
    send(sendBuffer);
}

//=================================================================================

XYZDumper::XYZDumper(std::string name, std::string path) :
        PostprocessPlugin(name),
        path(path)
{}

void XYZDumper::setup(const MPI_Comm& comm, const MPI_Comm& interComm)
{
    PostprocessPlugin::setup(comm, interComm);
    activated = createFoldersCollective(comm, path);
}

void XYZDumper::deserialize(MPI_Status& stat)
{
    std::string pvName;

    SimpleSerializer::deserialize(data, pvName, pos);

    std::string tstr = std::to_string(timeStamp++);
    std::string currentFname = path + "/" + pvName + "_" + std::string(5 - tstr.length(), '0') + tstr + ".xyz";

    if (activated)
        writeXYZ(comm, currentFname, pos.data(), pos.size());
}



