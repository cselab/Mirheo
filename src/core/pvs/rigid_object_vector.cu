#include "restart/helpers.h"
#include "checkpoint/helpers.h"
#include "rigid_object_vector.h"
#include "views/rov.h"

#include <core/rigid_kernels/integration.h>
#include <core/utils/folders.h>
#include <core/utils/kernel_launch.h>
#include <core/xdmf/type_map.h>
#include <core/xdmf/xdmf.h>

constexpr const char *RestartROVIdentifier = "ROV";
constexpr const char *RestartIPIdentifier = "ROV.TEMPLATE";

LocalRigidObjectVector::LocalRigidObjectVector(ParticleVector* pv, int objSize, int nObjects) :
    LocalObjectVector(pv, objSize, nObjects)
{}

PinnedBuffer<float4>* LocalRigidObjectVector::getMeshVertices(cudaStream_t stream)
{
    auto ov = dynamic_cast<RigidObjectVector*>(pv);
    auto& mesh = ov->mesh;
    meshVertices.resize_anew(nObjects * mesh->getNvertices());

    ROVview fakeView(ov, this);
    fakeView.objSize   = mesh->getNvertices();
    fakeView.size      = mesh->getNvertices() * nObjects;
    fakeView.positions = meshVertices.devPtr();

    const int nthreads = 128;
    
    SAFE_KERNEL_LAUNCH(
            RigidIntegrationKernels::applyRigidMotion
                <RigidIntegrationKernels::ApplyRigidMotion::PositionsOnly>,
            getNblocks(fakeView.size, nthreads), nthreads, 0, stream,
            fakeView, ov->mesh->vertexCoordinates.devPtr() );

    return &meshVertices;
}

PinnedBuffer<float4>* LocalRigidObjectVector::getOldMeshVertices(cudaStream_t stream)
{
    auto ov = dynamic_cast<RigidObjectVector*>(pv);
    auto& mesh = ov->mesh;
    meshOldVertices.resize_anew(nObjects * mesh->getNvertices());

    // Overwrite particles with vertices
    // Overwrite motions with the old_motions
    ROVview fakeView(ov, this);
    fakeView.objSize   = mesh->getNvertices();
    fakeView.size      = mesh->getNvertices() * nObjects;
    fakeView.positions = meshOldVertices.devPtr();
    fakeView.motions   = dataPerObject.getData<RigidMotion>(ChannelNames::oldMotions)->devPtr();

    const int nthreads = 128;
    
    SAFE_KERNEL_LAUNCH(
            RigidIntegrationKernels::applyRigidMotion
                <RigidIntegrationKernels::ApplyRigidMotion::PositionsOnly>,
            getNblocks(fakeView.size, nthreads), nthreads, 0, stream,
            fakeView, ov->mesh->vertexCoordinates.devPtr() );

    return &meshOldVertices;
}

PinnedBuffer<Force>* LocalRigidObjectVector::getMeshForces(cudaStream_t stream)
{
    auto ov = dynamic_cast<ObjectVector*>(pv);
    meshForces.resize_anew(nObjects * ov->mesh->getNvertices());
    return &meshForces;
}




RigidObjectVector::RigidObjectVector(const MirState *state, std::string name, float partMass,
                                     float3 J, const int objSize,
                                     std::shared_ptr<Mesh> mesh, const int nObjects) :
    ObjectVector( state, name, partMass, objSize,
                  std::make_unique<LocalRigidObjectVector>(this, objSize, nObjects),
                  std::make_unique<LocalRigidObjectVector>(this, objSize, 0) ),
    J(J)
{
    this->mesh = std::move(mesh);

    if (length(J) < 1e-5)
        die("Wrong momentum of inertia: [%f %f %f]", J.x, J.y, J.z);

    if (J.x < 0 || J.y < 0 || J.z < 0)
        die("Inertia tensor must be positive; got [%f %f %f]", J.x, J.y, J.z);


    // rigid motion must be exchanged and shifted
    requireDataPerObject<RigidMotion>(ChannelNames::motions,
                                      DataManager::PersistenceMode::Active,
                                      DataManager::ShiftMode::Active);

    requireDataPerObject<RigidMotion>(ChannelNames::oldMotions,
                                      DataManager::PersistenceMode::None);
}

RigidObjectVector::RigidObjectVector(const MirState *state, std::string name, float partMass,
                                     PyTypes::float3 J, const int objSize,
                                     std::shared_ptr<Mesh> mesh, const int nObjects) :
    RigidObjectVector( state, name, partMass, make_float3(J), objSize, mesh, nObjects )
{}

RigidObjectVector::~RigidObjectVector() = default;

static void writeInitialPositions(MPI_Comm comm, const std::string& filename,
                                  const PinnedBuffer<float4>& positions)
{
    int rank;
    MPI_Check( MPI_Comm_rank(comm, &rank) );
    if (rank != 0) return;

    FileWrapper f;
    f.open(filename, "wb");
    fwrite(positions.data(), sizeof(positions[0]), positions.size(), f.get());
}

static PinnedBuffer<float4> readInitialPositions(MPI_Comm comm, const std::string& filename,
                                                 int objSize)
{
    PinnedBuffer<float4> positions(objSize);
    int rank;
    MPI_Check( MPI_Comm_rank(comm, &rank) );
    constexpr int root = 0;

    if (rank == root)
    {
        FileWrapper f;
        f.open(filename, "rb");
        fread(positions.data(), sizeof(positions[0]), objSize, f.get());
    }
    MPI_Check(MPI_Bcast(positions.data(), objSize * sizeof(positions[0]),
                        MPI_BYTE, root, comm));

    positions.uploadToDevice(defaultStream);
    return positions;
}
                                  

void RigidObjectVector::_checkpointObjectData(MPI_Comm comm, std::string path, int checkpointId)
{
    CUDA_Check( cudaDeviceSynchronize() );

    auto filename = createCheckpointNameWithId(path, RestartROVIdentifier, "", checkpointId);
    info("Checkpoint for rigid object vector '%s', writing to file %s",
         name.c_str(), filename.c_str());

    auto motions = local()->dataPerObject.getData<RigidMotion>(ChannelNames::motions);

    motions->downloadFromDevice(defaultStream, ContainersSynch::Synch);
    
    auto positions = std::make_shared<std::vector<float3>>();
    std::vector<RigidReal4> quaternion;
    std::vector<RigidReal3> vel, omega, force, torque;
    
    std::tie(*positions, quaternion, vel, omega, force, torque)
        = CheckpointHelpers::splitAndShiftMotions(state->domain, *motions);

    XDMF::VertexGrid grid(positions, comm);    

    auto rigidType = XDMF::getNumberType<RigidReal>();

    const std::set<std::string> blackList {ChannelNames::motions};
    
    auto channels = CheckpointHelpers::extractShiftPersistentData(state->domain,
                                                                  local()->dataPerObject,
                                                                  blackList);

    channels.emplace_back(ChannelNames::XDMF::Motions::quaternion, quaternion .data(),
                          XDMF::Channel::DataForm::Quaternion,
                          rigidType, DataTypeWrapper<RigidReal4>());
    
    channels.emplace_back(ChannelNames::XDMF::Motions::velocity,   vel.data(),
                          XDMF::Channel::DataForm::Vector,
                          rigidType, DataTypeWrapper<RigidReal3>());

    channels.emplace_back(ChannelNames::XDMF::Motions::omega,      omega.data(),
                          XDMF::Channel::DataForm::Vector,
                          rigidType, DataTypeWrapper<RigidReal3>());
    
    channels.emplace_back(ChannelNames::XDMF::Motions::force,      force.data(),
                          XDMF::Channel::DataForm::Vector,
                          rigidType, DataTypeWrapper<RigidReal3>());
    
    channels.emplace_back(ChannelNames::XDMF::Motions::torque,     torque.data(),
                          XDMF::Channel::DataForm::Vector,
                          rigidType, DataTypeWrapper<RigidReal3>());
    
    XDMF::write(filename, &grid, channels, comm);

    createCheckpointSymlink(comm, path, RestartROVIdentifier, "xmf", checkpointId);

    filename = createCheckpointNameWithId(path, RestartIPIdentifier, "coords", checkpointId);
    writeInitialPositions(comm, filename, initialPositions);
    createCheckpointSymlink(comm, path, RestartIPIdentifier, "coords", checkpointId);

    debug("Checkpoint for object vector '%s' successfully written", name.c_str());
}

void RigidObjectVector::_restartObjectData(MPI_Comm comm, std::string path,
                                           const RigidObjectVector::ExchMapSize& ms)
{
    using namespace RestartHelpers;
    constexpr int objChunkSize = 1; // only one datum per object
    CUDA_Check( cudaDeviceSynchronize() );

    auto filename = createCheckpointName(path, RestartROVIdentifier, "xmf");
    info("Restarting rigid object vector %s from file %s", name.c_str(), filename.c_str());

    auto listData = readData(filename, comm, objChunkSize);

    namespace ChNames = ChannelNames::XDMF;
    auto pos        = extractChannel<float3>     (ChNames::position,            listData);
    auto quaternion = extractChannel<RigidReal4> (ChNames::Motions::quaternion, listData);
    auto vel        = extractChannel<RigidReal3> (ChNames::Motions::velocity,   listData);
    auto omega      = extractChannel<RigidReal3> (ChNames::Motions::omega,      listData);
    auto force      = extractChannel<RigidReal3> (ChNames::Motions::force,      listData);
    auto torque     = extractChannel<RigidReal3> (ChNames::Motions::torque,     listData);

    auto motions = combineMotions(pos, quaternion, vel, omega, force, torque);
    
    auto& dataPerObject = local()->dataPerObject;
    dataPerObject.resize_anew(ms.newSize);

    exchangeData    (comm, ms.map, motions,  objChunkSize);
    exchangeListData(comm, ms.map, listData, objChunkSize);

    shiftElementsGlobal2Local(motions, state->domain);

    auto& dstMotions = *dataPerObject.getData<RigidMotion>(ChannelNames::motions);

    std::copy(motions.begin(), motions.end(), dstMotions.begin());
    dstMotions.uploadToDevice(defaultStream);

    copyAndShiftListData(state->domain, listData, dataPerObject);


    filename = createCheckpointName(path, RestartIPIdentifier, "coords");
    initialPositions = readInitialPositions(comm, filename, objSize);

    info("Successfully read object infos of '%s'", name.c_str());
}
