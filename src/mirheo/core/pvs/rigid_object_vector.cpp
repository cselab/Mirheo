// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "restart/helpers.h"
#include "checkpoint/helpers.h"
#include "rigid_object_vector.h"
#include "views/rov.h"

#include <mirheo/core/rigid/operations.h>
#include <mirheo/core/snapshot.h>
#include <mirheo/core/utils/mpi_types.h>
#include <mirheo/core/utils/path.h>
#include <mirheo/core/xdmf/type_map.h>
#include <mirheo/core/xdmf/xdmf.h>

namespace mirheo
{

constexpr const char *RestartROVIdentifier = "ROV";
constexpr const char *RestartIPIdentifier = "ROV.TEMPLATE";

LocalRigidObjectVector::LocalRigidObjectVector(ParticleVector* pv, int objSize, int nObjects) :
    LocalObjectVector(pv, objSize, nObjects)
{}

PinnedBuffer<real4>* LocalRigidObjectVector::getMeshVertices(cudaStream_t stream)
{
    auto ov = dynamic_cast<RigidObjectVector*>(parent());
    auto& mesh = ov->mesh;
    meshVertices_.resize_anew(getNumObjects() * mesh->getNvertices());

    ROVview fakeView(ov, this);
    fakeView.objSize   = mesh->getNvertices();
    fakeView.size      = mesh->getNvertices() * getNumObjects();
    fakeView.positions = meshVertices_.devPtr();

    rigid_operations::applyRigidMotion(fakeView, ov->mesh->getVertices(),
                                      rigid_operations::ApplyTo::PositionsOnly, stream);

    return &meshVertices_;
}

PinnedBuffer<real4>* LocalRigidObjectVector::getOldMeshVertices(cudaStream_t stream)
{
    auto ov = dynamic_cast<RigidObjectVector*>(parent());
    auto& mesh = ov->mesh;
    meshOldVertices_.resize_anew(getNumObjects() * mesh->getNvertices());

    // Overwrite particles with vertices
    // Overwrite motions with the old_motions
    ROVview fakeView(ov, this);
    fakeView.objSize   = mesh->getNvertices();
    fakeView.size      = mesh->getNvertices() * getNumObjects();
    fakeView.positions = meshOldVertices_.devPtr();
    fakeView.motions   = dataPerObject.getData<RigidMotion>(channel_names::oldMotions)->devPtr();

    rigid_operations::applyRigidMotion(fakeView, ov->mesh->getVertices(),
                                      rigid_operations::ApplyTo::PositionsOnly, stream);

    return &meshOldVertices_;
}

PinnedBuffer<Force>* LocalRigidObjectVector::getMeshForces(__UNUSED cudaStream_t stream)
{
    auto ov = dynamic_cast<ObjectVector*>(parent());
    meshForces_.resize_anew(getNumObjects() * ov->mesh->getNvertices());
    return &meshForces_;
}

void LocalRigidObjectVector::clearRigidForces(cudaStream_t stream)
{
    ROVview view(static_cast<RigidObjectVector*>(parent()), this);
    rigid_operations::clearRigidForcesFromMotions(view, stream);
}




RigidObjectVector::RigidObjectVector(const MirState *state, const std::string& name, real partMass,
                                     real3 J, const int objSize,
                                     std::shared_ptr<Mesh> mesh_, const int nObjects) :
    ObjectVector( state, name, partMass, objSize,
                  std::make_unique<LocalRigidObjectVector>(this, objSize, nObjects),
                  std::make_unique<LocalRigidObjectVector>(this, objSize, 0) ),
    J_(J)
{
    mesh = std::move(mesh_);

    if (length(J) < 1e-5)
        die("Wrong momentum of inertia: [%f %f %f]", J.x, J.y, J.z);

    if (J.x < 0 || J.y < 0 || J.z < 0)
        die("Inertia tensor must be positive; got [%f %f %f]", J.x, J.y, J.z);


    // rigid motion must be exchanged and shifted
    requireDataPerObject<RigidMotion>(channel_names::motions,
                                      DataManager::PersistenceMode::Active,
                                      DataManager::ShiftMode::Active);

    requireDataPerObject<RigidMotion>(channel_names::oldMotions,
                                      DataManager::PersistenceMode::None);
}

RigidObjectVector::~RigidObjectVector() = default;

static void writeInitialPositions(MPI_Comm comm, const std::string& filename,
                                  const PinnedBuffer<real4>& positions)
{
    int rank;
    MPI_Check( MPI_Comm_rank(comm, &rank) );
    if (rank != 0) return;

    FileWrapper f;
    f.open(filename, "wb");
    fwrite(positions.data(), sizeof(positions[0]), positions.size(), f.get());
}

static PinnedBuffer<real4> readInitialPositions(MPI_Comm comm, const std::string& filename,
                                                 int objSize)
{
    PinnedBuffer<real4> positions(objSize);
    int rank;
    MPI_Check( MPI_Comm_rank(comm, &rank) );
    constexpr int root = 0;
    constexpr int nRealsPerPosition = sizeof(positions[0]) / sizeof(real);

    if (rank == root)
    {
        FileWrapper f;
        f.open(filename, "rb");
        f.fread(positions.data(), sizeof(positions[0]), objSize);
    }
    MPI_Check(MPI_Bcast(positions.data(), objSize * nRealsPerPosition, getMPIFloatType<real>(), root, comm));

    positions.uploadToDevice(defaultStream);
    return positions;
}


void RigidObjectVector::_snapshotObjectData(MPI_Comm comm, const std::string& xdmfFilename,
                                            const std::string& ipFilename)
{
    CUDA_Check( cudaDeviceSynchronize() );

    info("Checkpoint for rigid object vector '%s', writing to file %s",
         getCName(), xdmfFilename.c_str());

    auto motions = local()->dataPerObject.getData<RigidMotion>(channel_names::motions);

    motions->downloadFromDevice(defaultStream, ContainersSynch::Synch);

    auto positions = std::make_shared<std::vector<real3>>();
    std::vector<RigidReal4> quaternion;
    std::vector<RigidReal3> vel, omega, force, torque;

    std::tie(*positions, quaternion, vel, omega, force, torque)
        = checkpoint_helpers::splitAndShiftMotions(getState()->domain, *motions);

    XDMF::VertexGrid grid(positions, comm);

    auto rigidType = XDMF::getNumberType<RigidReal>();

    const std::set<std::string> blackList {channel_names::motions};

    auto channels = checkpoint_helpers::extractShiftPersistentData(getState()->domain,
                                                                   local()->dataPerObject,
                                                                   blackList);

    channels.push_back(XDMF::Channel{channel_names::XDMF::motions::quaternion, quaternion .data(),
                                         XDMF::Channel::DataForm::Quaternion,
                                         rigidType, DataTypeWrapper<RigidReal4>(),
                                         XDMF::Channel::NeedShift::False});

    channels.push_back(XDMF::Channel{channel_names::XDMF::motions::velocity,   vel.data(),
                                         XDMF::Channel::DataForm::Vector,
                                         rigidType, DataTypeWrapper<RigidReal3>(),
                                         XDMF::Channel::NeedShift::False});

    channels.push_back(XDMF::Channel{channel_names::XDMF::motions::omega,      omega.data(),
                                         XDMF::Channel::DataForm::Vector,
                                         rigidType, DataTypeWrapper<RigidReal3>(),
                                         XDMF::Channel::NeedShift::False});

    channels.push_back(XDMF::Channel{channel_names::XDMF::motions::force,      force.data(),
                                         XDMF::Channel::DataForm::Vector,
                                         rigidType, DataTypeWrapper<RigidReal3>(),
                                         XDMF::Channel::NeedShift::False});

    channels.push_back(XDMF::Channel{channel_names::XDMF::motions::torque,     torque.data(),
                                         XDMF::Channel::DataForm::Vector,
                                         rigidType, DataTypeWrapper<RigidReal3>(),
                                         XDMF::Channel::NeedShift::False});

    XDMF::write(xdmfFilename, &grid, channels, comm);

    writeInitialPositions(comm, ipFilename, initialPositions);

    debug("Checkpoint for rigid object vector '%s' successfully written", getCName());
}

void RigidObjectVector::saveSnapshotAndRegister(Saver& saver)
{
    saver.registerObject<RigidObjectVector>(this, _saveSnapshot(saver, "RigidObjectVector"));
}

ConfigObject RigidObjectVector::_saveSnapshot(Saver &saver, const std::string& typeName)
{
    die("RigitObjectVector::_saveSnapshot not tested.");
    // The filename does not include the extension.
    std::string xdmfFilename = joinPaths(saver.getContext().path, getName() + "." + RestartROVIdentifier);
    std::string ipFilename   = joinPaths(saver.getContext().path, getName() + "." + RestartIPIdentifier);
    _snapshotObjectData(saver.getContext().groupComm, xdmfFilename, ipFilename);

    ConfigObject config = ObjectVector::_saveSnapshot(saver, typeName);
    config.emplace("J", saver(J_));
    // `initialPositions` is stored in `_snapshotObjectData`.
    return config;
}

void RigidObjectVector::_checkpointObjectData(MPI_Comm comm, const std::string& path, int checkpointId)
{
    auto xdmfFilename = createCheckpointNameWithId(path, RestartROVIdentifier, "", checkpointId);
    auto ipFilename   = createCheckpointNameWithId(path, RestartIPIdentifier, "coords", checkpointId);
    _snapshotObjectData(comm, xdmfFilename, ipFilename);
    createCheckpointSymlink(comm, path, RestartROVIdentifier, "xmf", checkpointId);
    createCheckpointSymlink(comm, path, RestartIPIdentifier, "coords", checkpointId);
    debug("Symbolic links for rigid object vector '%s' created", getCName());
}

void RigidObjectVector::_restartObjectData(MPI_Comm comm, const std::string& path,
                                           const RigidObjectVector::ExchMapSize& ms)
{
    constexpr int objChunkSize = 1; // only one datum per object
    CUDA_Check( cudaDeviceSynchronize() );

    auto filename = createCheckpointName(path, RestartROVIdentifier, "xmf");
    info("Restarting rigid object vector %s from file %s", getCName(), filename.c_str());

    auto listData = restart_helpers::readData(filename, comm, objChunkSize);

    namespace ChNames = channel_names::XDMF;
    auto pos        = restart_helpers::extractChannel<real3>      (ChNames::position,            listData);
    auto quaternion = restart_helpers::extractChannel<RigidReal4> (ChNames::motions::quaternion, listData);
    auto vel        = restart_helpers::extractChannel<RigidReal3> (ChNames::motions::velocity,   listData);
    auto omega      = restart_helpers::extractChannel<RigidReal3> (ChNames::motions::omega,      listData);
    auto force      = restart_helpers::extractChannel<RigidReal3> (ChNames::motions::force,      listData);
    auto torque     = restart_helpers::extractChannel<RigidReal3> (ChNames::motions::torque,     listData);

    auto motions = restart_helpers::combineMotions(pos, quaternion, vel, omega, force, torque);

    restart_helpers::exchangeData    (comm, ms.map, motions,  objChunkSize);
    restart_helpers::exchangeListData(comm, ms.map, listData, objChunkSize);

    requireExtraDataPerObject(listData, this);

    auto& dataPerObject = local()->dataPerObject;
    dataPerObject.resize_anew(ms.newSize);
    restart_helpers::copyAndShiftListData(getState()->domain, listData, dataPerObject);

    restart_helpers::shiftElementsGlobal2Local(motions, getState()->domain);

    auto& dstMotions = *dataPerObject.getData<RigidMotion>(channel_names::motions);

    std::copy(motions.begin(), motions.end(), dstMotions.begin());
    dstMotions.uploadToDevice(defaultStream);


    filename = createCheckpointName(path, RestartIPIdentifier, "coords");
    initialPositions = readInitialPositions(comm, filename, getObjectSize());

    info("Successfully read object infos of '%s'", getCName());
}

} // namespace mirheo
