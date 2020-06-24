#include "dump_xyz.h"
#include "utils/simple_serializer.h"
#include "utils/time_stamp.h"
#include "utils/xyz.h"

#include <mirheo/core/pvs/particle_vector.h>
#include <mirheo/core/simulation.h>
#include <mirheo/core/utils/folders.h>

namespace mirheo
{

XYZPlugin::XYZPlugin(const MirState *state, std::string name, std::string pvName, int dumpEvery) :
    SimulationPlugin(state, name),
    pvName_(pvName),
    dumpEvery_(dumpEvery)
{}

void XYZPlugin::setup(Simulation* simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(simulation, comm, interComm);

    pv_ = simulation->getPVbyNameOrDie(pvName_);

    info("Plugin %s initialized for the following particle vector: %s", getCName(), pvName_.c_str());
}

void XYZPlugin::beforeForces(cudaStream_t stream)
{
    if (!isTimeEvery(getState(), dumpEvery_)) return;

    positions_.copy(pv_->local()->positions(), stream);
}

void XYZPlugin::serializeAndSend(__UNUSED cudaStream_t stream)
{
    if (!isTimeEvery(getState(), dumpEvery_)) return;

    debug2("Plugin %s is sending now data", getCName());

    for (auto& r : positions_)
    {
        auto r3 = make_real3(r);
        r3 = getState()->domain.local2global(r3);
        r.x = r3.x; r.y = r3.y; r.z = r3.z;
    }

    MirState::StepType timeStamp = getTimeStamp(getState(), dumpEvery_);

    _waitPrevSend();
    SimpleSerializer::serialize(sendBuffer_, timeStamp, pv_->getName(), positions_);
    _send(sendBuffer_);
}



XYZDumper::XYZDumper(std::string name, std::string path) :
    PostprocessPlugin(name),
    path_(makePath(path))
{}

XYZDumper::~XYZDumper() = default;

void XYZDumper::setup(const MPI_Comm& comm, const MPI_Comm& interComm)
{
    PostprocessPlugin::setup(comm, interComm);
    activated_ = createFoldersCollective(comm, path_);
}

void XYZDumper::deserialize()
{
    std::string pvName;
    MirState::StepType timeStamp;

    SimpleSerializer::deserialize(data_, timeStamp, pvName, pos_);

    std::string currentFname = path_ + pvName + "_" + createStrZeroPadded(timeStamp) + ".xyz";

    if (activated_)
        writeXYZ(comm_, currentFname, pos_.data(), static_cast<int>(pos_.size()));
}

} // namespace mirheo
