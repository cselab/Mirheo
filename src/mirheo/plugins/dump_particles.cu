// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "dump_particles.h"
#include "utils/simple_serializer.h"
#include "utils/time_stamp.h"

#include <mirheo/core/pvs/rod_vector.h>
#include <mirheo/core/simulation.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/utils/path.h>
#include <mirheo/core/utils/kernel_launch.h>
#include <mirheo/core/xdmf/type_map.h>

namespace mirheo
{

namespace dump_particles_kernels
{

template <typename T>
__global__ void copyObjectDataToParticles(int objSize, int nObjects, const T *srcObjData, T *dstParticleData)
{
    const int pid   = threadIdx.x + blockIdx.x * blockDim.x;
    const int objId = pid / objSize;

    if (objId >= nObjects) return;

    dstParticleData[pid] = srcObjData[objId];
}

template <typename T>
__global__ void copyRodDataToParticles(int numBiSegmentsPerObject, int objSize, int nObjects, const T *rodData, T *particleData)
{
    constexpr int stride = 5;
    const int pid = threadIdx.x + blockIdx.x * blockDim.x;

    const int objId        = pid / objSize;
    const int localPartId  = pid % objSize;
    const int localBisegId = math::min(localPartId / stride, numBiSegmentsPerObject); // min because of last particle

    const int bid = objId * numBiSegmentsPerObject + localBisegId;

    if (objId < nObjects)
        particleData[pid] = rodData[bid];
}

} // namespace dump_particles_kernels


ParticleSenderPlugin::ParticleSenderPlugin(const MirState *state, std::string name, std::string pvName, int dumpEvery,
                                           const std::vector<std::string>& channelNames) :
    SimulationPlugin(state, name),
    pvName_(pvName),
    dumpEvery_(dumpEvery),
    channelNames_(channelNames)
{
    channelData_.resize(channelNames_.size());
}
ParticleSenderPlugin::~ParticleSenderPlugin() = default;

void ParticleSenderPlugin::setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(simulation, comm, interComm);

    pv_ = simulation->getPVbyNameOrDie(pvName_);

    info("Plugin %s initialized for the following particle vector: %s", getCName(), pvName_.c_str());
}

void ParticleSenderPlugin::handshake()
{
    std::vector<XDMF::Channel::DataForm> dataForms;
    std::vector<XDMF::Channel::NumberType> numberTypes;
    std::vector<std::string> typeDescriptorsStr;

    auto pushChannelInfos = [&dataForms, &numberTypes, &typeDescriptorsStr](const DataManager::ChannelDescription& desc)
    {
        std::visit([&dataForms, &numberTypes, &typeDescriptorsStr](auto pinnedBufferPtr)
        {
            using T = typename std::remove_pointer<decltype(pinnedBufferPtr)>::type::value_type;
            dataForms         .push_back(XDMF::getDataForm  <T>());
            numberTypes       .push_back(XDMF::getNumberType<T>());
            typeDescriptorsStr.push_back(typeDescriptorToString(DataTypeWrapper<T>{}));
        }, desc.varDataPtr);
    };

    auto ov = dynamic_cast<ObjectVector*>(pv_);
    auto rv = dynamic_cast<RodVector*>(pv_);

    for (const auto& name : channelNames_)
    {
        if (pv_->local()->dataPerParticle.checkChannelExists(name))
        {
            const auto& desc = pv_->local()->dataPerParticle.getChannelDescOrDie(name);
            pushChannelInfos(desc);
        }
        else if (ov != nullptr && ov->local()->dataPerObject.checkChannelExists(name))
        {
            const auto& desc = ov->local()->dataPerObject.getChannelDescOrDie(name);
            pushChannelInfos(desc);
        }
        else if (rv != nullptr && rv->local()->dataPerBisegment.checkChannelExists(name))
        {
            const auto& desc = rv->local()->dataPerBisegment.getChannelDescOrDie(name);
            pushChannelInfos(desc);
        }
        else
        {
            die("Channel not found: '%s' in particle vector '%s'",
                getCName(), pv_->getCName());
        }
    }

    _waitPrevSend();
    SimpleSerializer::serialize(sendBuffer_, channelNames_, dataForms, numberTypes, typeDescriptorsStr);
    _send(sendBuffer_);
}

static inline void copyData(ParticleVector *pv, const std::string& channelName, HostBuffer<char>& dst, cudaStream_t stream)
{
    auto srcContainer = pv->local()->dataPerParticle.getGenericData(channelName);
    dst.genericCopy(srcContainer, stream);
}

static inline void copyData(ObjectVector *ov, const std::string& channelName, HostBuffer<char>& dst, DeviceBuffer<char>& workSpace, cudaStream_t stream)
{
    auto lov = ov->local();

    const auto& srcDesc = lov->dataPerObject.getChannelDescOrDie(channelName);

    const int objSize  = lov->getObjectSize();
    const int nObjects = lov->getNumObjects();

    std::visit([&](auto srcBufferPtr)
    {
        using T = typename std::remove_pointer<decltype(srcBufferPtr)>::type::value_type;

        constexpr int nthreads = 128;
        const int nParts = objSize * nObjects;
        const int nblocks = getNblocks(nParts, nthreads);

        workSpace.resize_anew(nParts * sizeof(T));

        SAFE_KERNEL_LAUNCH(
            dump_particles_kernels::copyObjectDataToParticles,
            nblocks, nthreads, 0, stream,
            objSize, nObjects, srcBufferPtr->devPtr(),
            reinterpret_cast<T*>(workSpace.devPtr()));
    }, srcDesc.varDataPtr);

    dst.genericCopy(&workSpace, stream);
}

static inline void copyData(RodVector *rv, const std::string& channelName, HostBuffer<char>& dst, DeviceBuffer<char>& workSpace, cudaStream_t stream)
{
    auto lrv = rv->local();

    const auto& srcDesc = lrv->dataPerBisegment.getChannelDescOrDie(channelName);

    const int objSize  = lrv->getObjectSize();
    const int nObjects = lrv->getNumObjects();
    const int numBiSegmentsPerObject = lrv->getNumSegmentsPerRod() - 1;

    std::visit([&](auto srcBufferPtr)
    {
        using T = typename std::remove_pointer<decltype(srcBufferPtr)>::type::value_type;

        constexpr int nthreads = 128;
        const int nParts = objSize * nObjects;
        const int nblocks = getNblocks(nParts, nthreads);

        workSpace.resize_anew(nParts * sizeof(T));

        SAFE_KERNEL_LAUNCH(
            dump_particles_kernels::copyRodDataToParticles,
            nblocks, nthreads, 0, stream,
            numBiSegmentsPerObject, objSize, nObjects, srcBufferPtr->devPtr(),
            reinterpret_cast<T*>(workSpace.devPtr()));
    }, srcDesc.varDataPtr);

    dst.genericCopy(&workSpace, stream);
}

void ParticleSenderPlugin::beforeForces(cudaStream_t stream)
{
    if (!isTimeEvery(getState(), dumpEvery_)) return;

    positions_ .genericCopy(&pv_->local()->positions() , stream);
    velocities_.genericCopy(&pv_->local()->velocities(), stream);

    auto ov = dynamic_cast<ObjectVector*>(pv_);
    auto rv = dynamic_cast<RodVector*>(pv_);

    for (size_t i = 0; i < channelNames_.size(); ++i)
    {
        auto name = channelNames_[i];

        if (pv_->local()->dataPerParticle.checkChannelExists(name))
        {
            copyData(pv_, name, channelData_[i], stream);
        }
        else if (ov != nullptr && ov->local()->dataPerObject.checkChannelExists(name))
        {
            copyData(ov, name, channelData_[i], workSpace_, stream);
        }
        else if (rv != nullptr && rv->local()->dataPerBisegment.checkChannelExists(name))
        {
            copyData(rv, name, channelData_[i], workSpace_, stream);
        }
        else
        {
            die("Channel not found: '%s' in particle vector '%s'",
                getCName(), pv_->getCName());
        }
    }
}

void ParticleSenderPlugin::serializeAndSend(__UNUSED cudaStream_t stream)
{
    if (!isTimeEvery(getState(), dumpEvery_)) return;

    debug2("Plugin %s is sending now data", getCName());

    for (auto& p : positions_)
    {
        auto r = getState()->domain.local2global(make_real3(p));
        p.x = r.x; p.y = r.y; p.z = r.z;
    }

    const MirState::StepType timeStamp = getTimeStamp(getState(), dumpEvery_);

    debug2("Plugin %s is packing now data consisting of %zu particles",
           getCName(), positions_.size());
    _waitPrevSend();
    SimpleSerializer::serialize(sendBuffer_, timeStamp, getState()->currentTime, positions_, velocities_, channelData_);
    _send(sendBuffer_);
}



ParticleDumperPlugin::ParticleDumperPlugin(std::string name, std::string path) :
    PostprocessPlugin(name),
    path_(path),
    positions_(std::make_shared<std::vector<real3>>())
{}

ParticleDumperPlugin::~ParticleDumperPlugin() = default;

void ParticleDumperPlugin::handshake()
{
    auto req = waitData();
    MPI_Check( MPI_Wait(&req, MPI_STATUS_IGNORE) );
    recv();

    std::vector<std::string> names;
    std::vector<XDMF::Channel::DataForm> dataForms;
    std::vector<XDMF::Channel::NumberType> numberTypes;
    std::vector<std::string> typeDescriptorsStr;

    SimpleSerializer::deserialize(data_, names, dataForms, numberTypes, typeDescriptorsStr);

    auto initChannel = [] (const std::string& name, XDMF::Channel::DataForm dataForm,
                           XDMF::Channel::NumberType numberType, TypeDescriptor datatype,
                           XDMF::Channel::NeedShift needShift = XDMF::Channel::NeedShift::False)
    {
        return XDMF::Channel{name, nullptr, dataForm, numberType, datatype, needShift};
    };

    // Velocity and id are special channels which are always present
    std::string allNames = "'velocity', 'id'";
    channels_.clear();
    channels_.push_back(initChannel("velocity", XDMF::Channel::DataForm::Vector, XDMF::getNumberType<real>(), DataTypeWrapper<real>()));
    channels_.push_back(initChannel("id",       XDMF::Channel::DataForm::Scalar, XDMF::Channel::NumberType::Int64, DataTypeWrapper<int64_t>()));

    for (size_t i = 0; i < names.size(); ++i)
    {
        const std::string& name = names[i];
        const auto dataForm   = dataForms[i];
        const auto numberType = numberTypes[i];
        const auto dataType   = stringToTypeDescriptor(typeDescriptorsStr[i]);

        const auto channel = initChannel(name, dataForm, numberType, dataType);

        channels_.push_back(channel);
        allNames += ", '" + name + "'";
    }

    // Create the required folder
    createFoldersCollective(comm_, getParentPath(path_));

    debug2("Plugin '%s' was set up to dump channels %s. Path is %s",
           getCName(), allNames.c_str(), path_.c_str());
}

static void unpackParticles(const std::vector<real4> &pos4, const std::vector<real4> &vel4,
                            std::vector<real3> &pos, std::vector<real3> &vel, std::vector<int64_t> &ids)
{
    const size_t n = pos4.size();
    pos.resize(n);
    vel.resize(n);
    ids.resize(n);

    for (size_t i = 0; i < n; ++i)
    {
        auto p = Particle(pos4[i], vel4[i]);
        pos[i] = p.r;
        vel[i] = p.u;
        ids[i] = p.getId();
    }
}

void ParticleDumperPlugin::_recvAndUnpack(MirState::TimeType &time, MirState::StepType& timeStamp)
{
    int c = 0;
    SimpleSerializer::deserialize(data_, timeStamp, time, pos4_, vel4_, channelData_);

    unpackParticles(pos4_, vel4_, *positions_, velocities_, ids_);

    channels_[c++].data = velocities_.data();
    channels_[c++].data = ids_.data();

    for (auto& cd : channelData_)
        channels_[c++].data = cd.data();
}

void ParticleDumperPlugin::deserialize()
{
    debug2("Plugin '%s' will dump right now", getCName());

    MirState::TimeType time;
    MirState::StepType timeStamp;
    _recvAndUnpack(time, timeStamp);

    std::string fname = path_ + createStrZeroPadded(timeStamp, zeroPadding_);

    XDMF::VertexGrid grid(positions_, comm_);
    XDMF::write(fname, &grid, channels_, time, comm_);
}

} // namespace mirheo
