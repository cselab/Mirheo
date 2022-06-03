// Copyright 2022 ETH Zurich. All Rights Reserved.
#include "dump_polylines.h"
#include "utils/simple_serializer.h"

#include <mirheo/core/pvs/chain_vector.h>
#include <mirheo/core/simulation.h>
#include <mirheo/core/utils/path.h>

namespace mirheo {

ParticleWithPolylinesSenderPlugin::ParticleWithPolylinesSenderPlugin(const MirState *state, std::string name,
                                                                     std::string pvName, int dumpEvery,
                                                                     const std::vector<std::string>& channelNames) :
    ParticleSenderPlugin(state, name, pvName, dumpEvery, channelNames)
{}

void ParticleWithPolylinesSenderPlugin::setup(Simulation *simulation,
                                              const MPI_Comm& comm,
                                              const MPI_Comm& interComm)
{
    SimulationPlugin::setup(simulation, comm, interComm);

    pv_ = simulation->getOVbyNameOrDie(pvName_);

    if (nullptr == dynamic_cast<ChainVector*>(pv_))
        die("%s expects a chain vector, got %s", getCName(), pvName_.c_str());

    info("Plugin %s initialized for the following chain vector: %s", getCName(), pvName_.c_str());
}

void ParticleWithPolylinesSenderPlugin::handshake()
{
    ParticleSenderPlugin::handshake();

    const int chainSize = static_cast<ChainVector*>(pv_)->getObjectSize();

    _waitPrevSend();
    debug("handshake for plugin '%s': sending polylines info", getCName());
    SimpleSerializer::serialize(sendBuffer_, chainSize);
    _send(sendBuffer_);
}




ParticleWithPolylinesDumperPlugin::ParticleWithPolylinesDumperPlugin(std::string name, std::string path) :
    ParticleDumperPlugin(name, path),
    allPolylines_(std::make_shared<std::vector<int>>()),
    chainSize_{-1}
{}

void ParticleWithPolylinesDumperPlugin::handshake()
{
    ParticleDumperPlugin::handshake();

    auto req = waitData();
    MPI_Check( MPI_Wait(&req, MPI_STATUS_IGNORE) );
    recv();

    SimpleSerializer::deserialize(data_, chainSize_);
    debug("handshake for plugin '%s': polylines have %d vertices", getCName(), chainSize_);
}

void ParticleWithPolylinesDumperPlugin::_prepareConnectivity(int totNVertices)
{
    if (totNVertices % chainSize_ != 0)
        die("plugin '%s' expecting a multiple of %d vertices, got %d", getCName(), chainSize_, totNVertices);

    const int nobjects = totNVertices / chainSize_;
    int offset = 0;

    MPI_Check( MPI_Exscan(&nobjects, &offset, 1, MPI_INT, MPI_SUM, comm_) );

    allPolylines_->resize(nobjects * chainSize_);

    for (int i = 0; i < nobjects; ++i)
    {
        const int start = chainSize_ * (offset + i);
        for (int j = 0; j < chainSize_; ++j)
        {
            const int id = i * chainSize_ + j;
            allPolylines_->data()[id] = start + j;
        }
    }
}

void ParticleWithPolylinesDumperPlugin::deserialize()
{
    debug2("Plugin '%s' will dump right now", getCName());

    MirState::TimeType time;
    MirState::StepType timeStamp;
    _recvAndUnpack(time, timeStamp);

    const int totNVertices = static_cast<int>(positions_->size());

    _prepareConnectivity(totNVertices);

    const std::string fname = path_ + createStrZeroPadded(timeStamp, zeroPadding_);

    const XDMF::PolylineMeshGrid grid(positions_, allPolylines_, chainSize_, comm_);
    XDMF::write(fname, &grid, channels_, time, comm_);
}

} // namespace mirheo
