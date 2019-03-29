#include "restart_helpers.h"
#include "rigid_object_vector.h"
#include "views/rov.h"

#include <core/rigid_kernels/integration.h>
#include <core/utils/folders.h>
#include <core/utils/kernel_launch.h>
#include <core/xdmf/type_map.h>
#include <core/xdmf/xdmf.h>

RigidObjectVector::RigidObjectVector(const YmrState *state, std::string name, float partMass,
                                     float3 J, const int objSize,
                                     std::shared_ptr<Mesh> mesh, const int nObjects) :
    ObjectVector( state, name, partMass, objSize,
                  new LocalRigidObjectVector(this, objSize, nObjects),
                  new LocalRigidObjectVector(this, objSize, 0) ),
    J(J)
{
    this->mesh = std::move(mesh);

    if (length(J) < 1e-5)
        die("Wrong momentum of inertia: [%f %f %f]", J.x, J.y, J.z);

    if (J.x < 0 || J.y < 0 || J.z < 0)
        die("Inertia tensor must be positive; got [%f %f %f]", J.x, J.y, J.z);


    // rigid motion must be exchanged and shifted
    requireDataPerObject<RigidMotion>(ChannelNames::motions,
                                      ExtraDataManager::PersistenceMode::Persistent,
                                      sizeof(RigidReal));

    requireDataPerObject<RigidMotion>(ChannelNames::oldMotions,
                                      ExtraDataManager::PersistenceMode::None);
}

RigidObjectVector::RigidObjectVector(const YmrState *state, std::string name, float partMass,
                                     PyTypes::float3 J, const int objSize,
                                     std::shared_ptr<Mesh> mesh, const int nObjects) :
    RigidObjectVector( state, name, partMass, make_float3(J), objSize, mesh, nObjects )
{}

PinnedBuffer<Particle>* LocalRigidObjectVector::getMeshVertices(cudaStream_t stream)
{
    auto ov = dynamic_cast<RigidObjectVector*>(pv);
    auto& mesh = ov->mesh;
    meshVertices.resize_anew(nObjects * mesh->getNvertices());

    ROVview fakeView(ov, this);
    fakeView.objSize = mesh->getNvertices();
    fakeView.size = mesh->getNvertices() * nObjects;
    fakeView.particles = reinterpret_cast<float4*>(meshVertices.devPtr());

    SAFE_KERNEL_LAUNCH(
            RigidIntegrationKernels::applyRigidMotion,
            getNblocks(fakeView.size, 128), 128, 0, stream,
            fakeView, ov->mesh->vertexCoordinates.devPtr() );

    return &meshVertices;
}

PinnedBuffer<Particle>* LocalRigidObjectVector::getOldMeshVertices(cudaStream_t stream)
{
    auto ov = dynamic_cast<RigidObjectVector*>(pv);
    auto& mesh = ov->mesh;
    meshOldVertices.resize_anew(nObjects * mesh->getNvertices());

    // Overwrite particles with vertices
    // Overwrite motions with the old_motions
    ROVview fakeView(ov, this);
    fakeView.objSize = mesh->getNvertices();
    fakeView.size = mesh->getNvertices() * nObjects;
    fakeView.particles = reinterpret_cast<float4*>(meshOldVertices.devPtr());
    fakeView.motions = extraPerObject.getData<RigidMotion>(ChannelNames::oldMotions)->devPtr();

    SAFE_KERNEL_LAUNCH(
            RigidIntegrationKernels::applyRigidMotion,
            getNblocks(fakeView.size, 128), 128, 0, stream,
            fakeView, ov->mesh->vertexCoordinates.devPtr() );

    return &meshOldVertices;
}

DeviceBuffer<Force>* LocalRigidObjectVector::getMeshForces(cudaStream_t stream)
{
    auto ov = dynamic_cast<ObjectVector*>(pv);
    meshForces.resize_anew(nObjects * ov->mesh->getNvertices());

    return &meshForces;
}

// TODO refactor this

static void splitMotions(DomainInfo domain, const PinnedBuffer<RigidMotion>& motions,
                         std::vector<float> &pos, std::vector<RigidReal4> &quaternion,
                         std::vector<RigidReal3> &vel, std::vector<RigidReal3> &omega,
                         std::vector<RigidReal3> &force, std::vector<RigidReal3> &torque)
{
    int n = motions.size();
    pos  .resize(3*n); quaternion.resize(n);
    vel  .resize(n);        omega.resize(n);
    force.resize(n);       torque.resize(n);

    float3 *pos3 = (float3*) pos.data();
    
    for (int i = 0; i < n; ++i) {
        auto m = motions[i];
        pos3[i] = domain.local2global(make_float3(m.r));
        quaternion[i] = m.q;
        vel[i] = m.vel;
        omega[i] = m.omega;
        force[i] = m.force;
        torque[i] = m.torque;
    }
}

void RigidObjectVector::_checkpointObjectData(MPI_Comm comm, std::string path)
{
    CUDA_Check( cudaDeviceSynchronize() );

    auto filename = createCheckpointNameWithId(path, "ROV", "");
    info("Checkpoint for rigid object vector '%s', writing to file %s", name.c_str(), filename.c_str());

    auto motions = local()->extraPerObject.getData<RigidMotion>(ChannelNames::motions);

    motions->downloadFromDevice(0, ContainersSynch::Synch);
    
    auto positions = std::make_shared<std::vector<float>>();
    std::vector<RigidReal4> quaternion;
    std::vector<RigidReal3> vel, omega, force, torque;
    
    splitMotions(state->domain, *motions, *positions, quaternion, vel, omega, force, torque);

    XDMF::VertexGrid grid(positions, comm);    

    auto rigidType = XDMF::getNumberType<RigidReal>();

    std::vector<XDMF::Channel> channels = {
        XDMF::Channel( "quaternion", quaternion .data(), XDMF::Channel::DataForm::Quaternion, rigidType, typeTokenize<RigidReal4>() ),
        XDMF::Channel( "velocity",   vel        .data(), XDMF::Channel::DataForm::Vector,     rigidType, typeTokenize<RigidReal3>() ),
        XDMF::Channel( "omega",      omega      .data(), XDMF::Channel::DataForm::Vector,     rigidType, typeTokenize<RigidReal3>() ),
        XDMF::Channel( "force",      force      .data(), XDMF::Channel::DataForm::Vector,     rigidType, typeTokenize<RigidReal3>() ),
        XDMF::Channel( "torque",     torque     .data(), XDMF::Channel::DataForm::Vector,     rigidType, typeTokenize<RigidReal3>() )
    };         

    _extractPersistentExtraObjectData(channels, /* blacklist */ {ChannelNames::motions} );
    
    XDMF::write(filename, &grid, channels, comm);

    createCheckpointSymlink(comm, path, "ROV", "xmf");

    debug("Checkpoint for object vector '%s' successfully written", name.c_str());
}

static void shiftCoordinates(const DomainInfo& domain, std::vector<RigidMotion>& motions)
{
    for (auto& m : motions)
        m.r = make_rigidReal3( domain.global2local(make_float3(m.r)) );
}

void RigidObjectVector::_restartObjectData(MPI_Comm comm, std::string path, const std::vector<int>& map)
{
    CUDA_Check( cudaDeviceSynchronize() );

    auto filename = createCheckpointName(path, "ROV", "xmf");
    info("Restarting rigid object vector %s from file %s", name.c_str(), filename.c_str());

    XDMF::readRigidObjectData(filename, comm, this);

    auto loc_ids     = local()->extraPerObject.getData<int>(ChannelNames::globalIds);
    auto loc_motions = local()->extraPerObject.getData<RigidMotion>(ChannelNames::motions);
    
    std::vector<int>             ids(loc_ids->size());
    std::vector<RigidMotion> motions(loc_motions->size());
    
    std::copy(loc_ids    ->begin(), loc_ids    ->end(), ids.begin());
    std::copy(loc_motions->begin(), loc_motions->end(), motions.begin());
    
    RestartHelpers::exchangeData(comm, map, ids, 1);
    RestartHelpers::exchangeData(comm, map, motions, 1);

    shiftCoordinates(state->domain, motions);
    
    loc_ids->resize_anew(ids.size());
    loc_motions->resize_anew(motions.size());

    std::copy(ids.begin(), ids.end(), loc_ids->begin());
    std::copy(motions.begin(), motions.end(), loc_motions->begin());

    loc_ids->uploadToDevice(0);
    loc_motions->uploadToDevice(0);
    CUDA_Check( cudaDeviceSynchronize() );

    info("Successfully read %d object infos", loc_motions->size());
}
