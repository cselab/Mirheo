#pragma once

#include "interface.h"

#include "add_force.h"
#include "add_torque.h"
#include "anchor_particle.h"
#include "average_flow.h"
#include "average_relative_flow.h"
#include "channel_dumper.h"
#include "outlet.h"
#include "density_control.h"
#include "displacement.h"
#include "dump_mesh.h"
#include "dump_obj_stats.h"
#include "dump_particles.h"
#include "dump_particles_rod.h"
#include "dump_particles_with_mesh.h"
#include "dump_xyz.h"
#include "exchange_pvs_flux_plane.h"
#include "force_saver.h"
#include "impose_profile.h"
#include "impose_velocity.h"
#include "magnetic_orientation.h"
#include "membrane_extra_force.h"
#include "object_portal.h"
#include "object_to_particles.h"
#include "particle_channel_saver.h"
#include "particle_checker.h"
#include "particle_drag.h"
#include "particle_portal.h"
#include "pin_object.h"
#include "pin_rod_extremity.h"
#include "radial_velocity_control.h"
#include "stats.h"
#include "temperaturize.h"
#include "velocity_control.h"
#include "velocity_inlet.h"
#include "virial_pressure.h"
#include "wall_force_collector.h"
#include "wall_repulsion.h"

#include <mirheo/core/pvs/object_vector.h>
#include <mirheo/core/pvs/rod_vector.h>
#include <mirheo/core/pvs/particle_vector.h>
#include <mirheo/core/walls/interface.h>

#include <memory>

namespace PluginFactory
{
template <typename T1, typename T2>
using pair_shared = std::pair<std::shared_ptr<T1>, std::shared_ptr<T2>>;
    
static void extractChannelsInfos(const std::vector< std::pair<std::string, std::string> >& channels,
                                 std::vector<std::string>& names, std::vector<Average3D::ChannelType>& types)
{
    for (auto& p : channels) {
        names.push_back(p.first);
        std::string typeStr = p.second;

        if      (typeStr == "scalar")             types.push_back(Average3D::ChannelType::Scalar);
        else if (typeStr == "vector")             types.push_back(Average3D::ChannelType::Vector_real3);
        else if (typeStr == "vector_from_float4") types.push_back(Average3D::ChannelType::Vector_real4);
        else if (typeStr == "tensor6")            types.push_back(Average3D::ChannelType::Tensor6);
        else die("Unable to get parse channel type '%s'", typeStr.c_str());
    }
}

static void extractPVsNames(const std::vector<ParticleVector*>& pvs, std::vector<std::string>& pvNames)
{
    pvNames.reserve(pvs.size());
    for (auto &pv : pvs)
        pvNames.push_back(pv->name);
}

static void extractChannelInfos(const std::vector< std::pair<std::string, std::string> >& channels,
                                std::vector<std::string>& names, std::vector<ParticleSenderPlugin::ChannelType>& types)
{
    for (auto& p : channels) {
        names.push_back(p.first);
        std::string typeStr = p.second;

        if      (typeStr == "scalar")    types.push_back(ParticleSenderPlugin::ChannelType::Scalar);
        else if (typeStr == "vector")    types.push_back(ParticleSenderPlugin::ChannelType::Vector);
        else if (typeStr == "tensor6")   types.push_back(ParticleSenderPlugin::ChannelType::Tensor6);
        else die("Unable to get parse channel type '%s'", typeStr.c_str());
    }
}

    

    
inline pair_shared< AddForcePlugin, PostprocessPlugin >
createAddForcePlugin(bool computeTask, const MirState *state, std::string name, ParticleVector *pv, real3 force)
{
    auto simPl = computeTask ? std::make_shared<AddForcePlugin> (state, name, pv->name, force) : nullptr;
    return { simPl, nullptr };
}

inline pair_shared< AddTorquePlugin, PostprocessPlugin >
createAddTorquePlugin(bool computeTask, const MirState *state, std::string name, ParticleVector *pv, real3 torque)
{
    auto simPl = computeTask ? std::make_shared<AddTorquePlugin> (state, name, pv->name, torque) : nullptr;
    return { simPl, nullptr };
}

inline pair_shared< AnchorParticlesPlugin, AnchorParticlesStatsPlugin >
createAnchorParticlesPlugin(bool computeTask, const MirState *state, std::string name, ParticleVector *pv,
                            std::function<std::vector<real3>(real)> positions,
                            std::function<std::vector<real3>(real)> velocities,
                            std::vector<int> pids, int reportEvery, const std::string& path)
{
    auto simPl = computeTask ?
        std::make_shared<AnchorParticlesPlugin> (state, name, pv->name,
                                                 positions, velocities,
                                                 pids, reportEvery)
        : nullptr;

    auto postPl = computeTask ?
        nullptr :
        std::make_shared<AnchorParticlesStatsPlugin> (name, path);
    
    return { simPl, postPl };
}

inline pair_shared< DensityControlPlugin, PostprocessDensityControl >
createDensityControlPlugin(bool computeTask, const MirState *state, std::string name, std::string fname, std::vector<ParticleVector*> pvs,
                           real targetDensity, std::function<real(real3)> region, real3 resolution,
                           real levelLo, real levelHi, real levelSpace, real Kp, real Ki, real Kd,
                           int tuneEvery, int dumpEvery, int sampleEvery)
{
    std::vector<std::string> pvNames;

    if (computeTask) extractPVsNames(pvs, pvNames);
    
    auto simPl = computeTask ?
        std::make_shared<DensityControlPlugin> (state, name, pvNames, targetDensity,
                                                region, resolution, levelLo, levelHi, levelSpace,
                                                Kp, Ki, Kd, tuneEvery, dumpEvery, sampleEvery) :
        nullptr;

    auto postPl = computeTask ?
        nullptr :
        std::make_shared<PostprocessDensityControl> (name, fname);
    
    return { simPl, postPl };
}

inline pair_shared< DensityOutletPlugin, PostprocessPlugin >
createDensityOutletPlugin(bool computeTask, const MirState *state, std::string name, std::vector<ParticleVector*> pvs,
                          real numberDensity, std::function<real(real3)> region, real3 resolution)
{
    std::vector<std::string> pvNames;

    if (computeTask) extractPVsNames(pvs, pvNames);
    
    auto simPl = computeTask ?
        std::make_shared<DensityOutletPlugin> (state, name, pvNames, numberDensity, region, resolution)
        : nullptr;
    return { simPl, nullptr };
}

inline pair_shared< PlaneOutletPlugin, PostprocessPlugin >
createPlaneOutletPlugin(bool computeTask, const MirState *state, std::string name,
                        std::vector<ParticleVector*> pvs, real4 plane)
{
    if (!computeTask)
        return { nullptr, nullptr };

    std::vector<std::string> pvNames;
    extractPVsNames(pvs, pvNames);

    return { std::make_shared<PlaneOutletPlugin> (state, name, std::move(pvNames), plane), nullptr };
}

inline pair_shared< RateOutletPlugin, PostprocessPlugin >
createRateOutletPlugin(bool computeTask, const MirState *state, std::string name, std::vector<ParticleVector*> pvs,
                       real rate, std::function<real(real3)> region, real3 resolution)
{
    std::vector<std::string> pvNames;

    if (computeTask) extractPVsNames(pvs, pvNames);
    
    auto simPl = computeTask ?
        std::make_shared<RateOutletPlugin> (state, name, pvNames, rate, region, resolution)
        : nullptr;
    return { simPl, nullptr };
}

inline pair_shared< Average3D, UniformCartesianDumper >
createDumpAveragePlugin(bool computeTask, const MirState *state, std::string name, std::vector<ParticleVector*> pvs,
                        int sampleEvery, int dumpEvery, real3 binSize,
                        std::vector< std::pair<std::string, std::string> > channels,
                        std::string path)
{
    std::vector<std::string> names, pvNames;
    std::vector<Average3D::ChannelType> types;

    extractChannelsInfos(channels, names, types);
        
    if (computeTask) extractPVsNames(pvs, pvNames);
        
    auto simPl  = computeTask ?
        std::make_shared<Average3D> (state, name, pvNames, names, types, sampleEvery, dumpEvery, binSize) :
        nullptr;

    auto postPl = computeTask ? nullptr : std::make_shared<UniformCartesianDumper> (name, path);

    return { simPl, postPl };
}

inline pair_shared< AverageRelative3D, UniformCartesianDumper >
createDumpAverageRelativePlugin(bool computeTask, const MirState *state, std::string name, std::vector<ParticleVector*> pvs,
                                ObjectVector* relativeToOV, int relativeToId,
                                int sampleEvery, int dumpEvery, real3 binSize,
                                std::vector< std::pair<std::string, std::string> > channels,
                                std::string path)
{
    std::vector<std::string> names, pvNames;
    std::vector<Average3D::ChannelType> types;

    extractChannelsInfos(channels, names, types);

    if (computeTask) extractPVsNames(pvs, pvNames);
    
    auto simPl  = computeTask ?
        std::make_shared<AverageRelative3D> (state, name, pvNames,
                                             names, types, sampleEvery, dumpEvery,
                                             binSize, relativeToOV->name, relativeToId) :
        nullptr;

    auto postPl = computeTask ? nullptr : std::make_shared<UniformCartesianDumper> (name, path);

    return { simPl, postPl };
}

inline pair_shared< MeshPlugin, MeshDumper >
createDumpMeshPlugin(bool computeTask, const MirState *state, std::string name, ObjectVector* ov, int dumpEvery, std::string path)
{
    auto simPl  = computeTask ? std::make_shared<MeshPlugin> (state, name, ov->name, dumpEvery) : nullptr;
    auto postPl = computeTask ? nullptr : std::make_shared<MeshDumper> (name, path);

    return { simPl, postPl };
}

inline pair_shared< ParticleSenderPlugin, ParticleDumperPlugin >
createDumpParticlesPlugin(bool computeTask, const MirState *state, std::string name, ParticleVector *pv, int dumpEvery,
                          std::vector< std::pair<std::string, std::string> > channels, std::string path)
{
    std::vector<std::string> names;
    std::vector<ParticleSenderPlugin::ChannelType> types;

    extractChannelInfos(channels, names, types);
        
    auto simPl  = computeTask ? std::make_shared<ParticleSenderPlugin> (state, name, pv->name, dumpEvery, names, types) : nullptr;
    auto postPl = computeTask ? nullptr : std::make_shared<ParticleDumperPlugin> (name, path);

    return { simPl, postPl };
}

inline pair_shared< ParticleWithRodQuantitiesSenderPlugin, ParticleDumperPlugin >
createDumpParticlesWithRodDataPlugin(bool computeTask, const MirState *state, std::string name, ParticleVector *pv, int dumpEvery,
                                     std::vector< std::pair<std::string, std::string> > channels, std::string path)
{
    std::vector<std::string> names;
    std::vector<ParticleSenderPlugin::ChannelType> types;

    extractChannelInfos(channels, names, types);
        
    auto simPl  = computeTask ? std::make_shared<ParticleWithRodQuantitiesSenderPlugin> (state, name, pv->name, dumpEvery, names, types) : nullptr;
    auto postPl = computeTask ? nullptr : std::make_shared<ParticleDumperPlugin> (name, path);

    return { simPl, postPl };
}

inline pair_shared< ParticleWithMeshSenderPlugin, ParticleWithMeshDumperPlugin >
createDumpParticlesWithMeshPlugin(bool computeTask, const MirState *state, std::string name, ObjectVector *ov, int dumpEvery,
                                  std::vector< std::pair<std::string, std::string> > channels, std::string path)
{
    std::vector<std::string> names;
    std::vector<ParticleSenderPlugin::ChannelType> types;

    extractChannelInfos(channels, names, types);
        
    auto simPl  = computeTask ? std::make_shared<ParticleWithMeshSenderPlugin> (state, name, ov->name, dumpEvery, names, types) : nullptr;
    auto postPl = computeTask ? nullptr : std::make_shared<ParticleWithMeshDumperPlugin> (name, path);

    return { simPl, postPl };
}

inline pair_shared< XYZPlugin, XYZDumper >
createDumpXYZPlugin(bool computeTask, const MirState *state, std::string name, ParticleVector* pv, int dumpEvery, std::string path)
{
    auto simPl  = computeTask ? std::make_shared<XYZPlugin> (state, name, pv->name, dumpEvery) : nullptr;
    auto postPl = computeTask ? nullptr : std::make_shared<XYZDumper> (name, path);

    return { simPl, postPl };
}

inline pair_shared< ObjStatsPlugin, ObjStatsDumper >
createDumpObjStats(bool computeTask, const MirState *state, std::string name, ObjectVector* ov, int dumpEvery, std::string path)
{
    auto simPl  = computeTask ? std::make_shared<ObjStatsPlugin> (state, name, ov->name, dumpEvery) : nullptr;
    auto postPl = computeTask ? nullptr : std::make_shared<ObjStatsDumper> (name, path);

    return { simPl, postPl };
}

inline pair_shared< ExchangePVSFluxPlanePlugin, PostprocessPlugin >
createExchangePVSFluxPlanePlugin(bool computeTask, const MirState *state, std::string name, ParticleVector *pv1, ParticleVector *pv2, real4 plane)
{
    auto simPl = computeTask ?
        std::make_shared<ExchangePVSFluxPlanePlugin> (state, name, pv1->name, pv2->name, plane) : nullptr;
        
    return { simPl, nullptr };    
}

inline pair_shared< ForceSaverPlugin, PostprocessPlugin >
createForceSaverPlugin(bool computeTask,  const MirState *state, std::string name, ParticleVector *pv)
{
    auto simPl = computeTask ? std::make_shared<ForceSaverPlugin> (state, name, pv->name) : nullptr;
    return { simPl, nullptr };
}

inline pair_shared< ImposeProfilePlugin, PostprocessPlugin >
createImposeProfilePlugin(bool computeTask,  const MirState *state, std::string name, ParticleVector* pv, 
                          real3 low, real3 high, real3 velocity, real kBT)
{
    auto simPl = computeTask ?
        std::make_shared<ImposeProfilePlugin> (state, name, pv->name, low, high, velocity, kBT) :
        nullptr;
            
    return { simPl, nullptr };
}

inline pair_shared< ImposeVelocityPlugin, PostprocessPlugin >
createImposeVelocityPlugin(bool computeTask,  const MirState *state, std::string name,
                           std::vector<ParticleVector*> pvs, int every,
                           real3 low, real3 high, real3 velocity)
{
    std::vector<std::string> pvNames;
    if (computeTask) extractPVsNames(pvs, pvNames);
            
    auto simPl = computeTask ?
        std::make_shared<ImposeVelocityPlugin> (state, name, pvNames, low, high, velocity, every) :
        nullptr;
                                    
    return { simPl, nullptr };
}

inline pair_shared< MagneticOrientationPlugin, PostprocessPlugin >
createMagneticOrientationPlugin(bool computeTask, const MirState *state, std::string name, RigidObjectVector *rov, real3 moment,
                                std::function<real3(real)> magneticFunction)
{
    auto simPl = computeTask ?
        std::make_shared<MagneticOrientationPlugin>(state, name, rov->name, moment, magneticFunction)
        : nullptr;

    return { simPl, nullptr };
}

inline pair_shared< MembraneExtraForcePlugin, PostprocessPlugin >
createMembraneExtraForcePlugin(bool computeTask, const MirState *state, std::string name, ParticleVector *pv, const std::vector<real3>& forces)
{
    auto simPl = computeTask ?
        std::make_shared<MembraneExtraForcePlugin> (state, name, pv->name, forces) : nullptr;

    return { simPl, nullptr };
}

inline pair_shared< ObjectPortalDestination, PostprocessPlugin >
createObjectPortalDestination(bool computeTask, const MirState *state, std::string name,
                              ObjectVector *ov, real3 src, real3 dst, real3 size,
                              int tag, long interCommPtr)
{
    if (!computeTask)
        return { nullptr, nullptr };

    MPI_Comm interComm = *((MPI_Comm *)interCommPtr);
    auto simPl = std::make_shared<ObjectPortalDestination> (
            state, name, ov->name, src, dst, size, tag, interComm);
    return { std::move(simPl), nullptr };
}

inline pair_shared< ObjectPortalSource, PostprocessPlugin >
createObjectPortalSource(bool computeTask, const MirState *state, std::string name,
                         ObjectVector *ov, real3 src, real3 dst, real3 size, real4 plane,
                         int tag, long interCommPtr)
{
    if (!computeTask)
        return { nullptr, nullptr };

    MPI_Comm interComm = *((MPI_Comm *)interCommPtr);
    auto simPl = std::make_shared<ObjectPortalSource> (
            state, name, ov->name, src, dst, size, plane, tag, interComm);
    return { std::move(simPl), nullptr };
}

inline pair_shared< ObjectToParticlesPlugin, PostprocessPlugin >
createObjectToParticlesPlugin(bool computeTask, const MirState *state, std::string name,
                              ObjectVector *ov, ParticleVector *pv, real4 plane)
{
    if (!computeTask)
        return { nullptr, nullptr };

    return { std::make_shared<ObjectToParticlesPlugin> (state, name, ov->name, pv->name, plane), nullptr };
}

inline pair_shared< ParticleChannelSaverPlugin, PostprocessPlugin >
createParticleChannelSaverPlugin(bool computeTask, const MirState *state, std::string name, ParticleVector *pv,
                                 std::string channelName, std::string savedName)
{
    auto simPl = computeTask ? std::make_shared<ParticleChannelSaverPlugin> (state, name, pv->name, channelName, savedName) : nullptr;
    return { simPl, nullptr };
}

inline pair_shared< ParticleCheckerPlugin, PostprocessPlugin >
createParticleCheckerPlugin(bool computeTask, const MirState *state, std::string name, int checkEvery)
{
    auto simPl = computeTask ? std::make_shared<ParticleCheckerPlugin> (state, name, checkEvery) : nullptr;
    return { simPl, nullptr };
}

inline pair_shared< ParticleDisplacementPlugin, PostprocessPlugin >
createParticleDisplacementPlugin(bool computeTask, const MirState *state, std::string name, ParticleVector *pv, int updateEvery)
{
    auto simPl = computeTask ?
        std::make_shared<ParticleDisplacementPlugin> (state, name, pv->name, updateEvery) :
        nullptr;
    return { simPl, nullptr };
}

inline pair_shared< ParticleDragPlugin, PostprocessPlugin >
createParticleDragPlugin(bool computeTask, const MirState *state, std::string name, ParticleVector *pv, real drag)
{
    auto simPl = computeTask ?
        std::make_shared<ParticleDragPlugin> (state, name, pv->name, drag) :
        nullptr;
    return { simPl, nullptr };
}

inline pair_shared< ParticlePortalDestination, PostprocessPlugin >
createParticlePortalDestination(bool computeTask, const MirState *state, std::string name, ParticleVector *pv,
                                real3 src, real3 dst, real3 size, int tag, long comm_ptr)
{
    MPI_Comm comm = *((MPI_Comm *)comm_ptr);
    auto simPl = computeTask ? std::make_shared<ParticlePortalDestination> (
            state, name, pv->name, src, dst, size, tag, comm) : nullptr;
    return { std::move(simPl), nullptr };
}

inline pair_shared< ParticlePortalSource, PostprocessPlugin >
createParticlePortalSource(bool computeTask, const MirState *state, std::string name, ParticleVector *pv,
                                real3 src, real3 dst, real3 size, int tag, long comm_ptr)
{
    MPI_Comm comm = *((MPI_Comm *)comm_ptr);
    auto simPl = computeTask ? std::make_shared<ParticlePortalSource> (
            state, name, pv->name, src, dst, size, tag, comm) : nullptr;
    return { std::move(simPl), nullptr };
}

inline pair_shared< PinObjectPlugin, ReportPinObjectPlugin >
createPinObjPlugin(bool computeTask, const MirState *state, std::string name, ObjectVector *ov,
                   int dumpEvery, std::string path, real3 velocity, real3 omega)
{
    auto simPl  = computeTask ? std::make_shared<PinObjectPlugin> (state, name, ov->name, velocity, omega, dumpEvery) : 
        nullptr;
    auto postPl = computeTask ? nullptr : std::make_shared<ReportPinObjectPlugin> (name, path);

    return { simPl, postPl };
}

inline pair_shared< PinRodExtremityPlugin, PostprocessPlugin >
createPinRodExtremityPlugin(bool computeTask, const MirState *state, std::string name, RodVector *rv, int segmentId,
                            real fmagn, real3 targetDirection)
{
    auto simPl  = computeTask ?
        std::make_shared<PinRodExtremityPlugin> (state, name, rv->name, segmentId, fmagn, targetDirection) : 
        nullptr;

    return { simPl, nullptr };
}

inline pair_shared< SimulationVelocityControl, PostprocessVelocityControl >
createVelocityControlPlugin(bool computeTask, const MirState *state, std::string name, std::string filename, std::vector<ParticleVector*> pvs,
                            real3 low, real3 high, int sampleEvery, int tuneEvery, int dumpEvery, real3 targetVel, real Kp, real Ki, real Kd)
{
    std::vector<std::string> pvNames;
    if (computeTask) extractPVsNames(pvs, pvNames);
        
    auto simPl = computeTask ?
        std::make_shared<SimulationVelocityControl>(state, name, pvNames, low, high,
                                                    sampleEvery, tuneEvery, dumpEvery,
                                                    targetVel, Kp, Ki, Kd) :
        nullptr;

    auto postPl = computeTask ?
        nullptr :
        std::make_shared<PostprocessVelocityControl> (name, filename);

    return { simPl, postPl };
}

inline pair_shared< SimulationRadialVelocityControl, PostprocessRadialVelocityControl >
createRadialVelocityControlPlugin(bool computeTask, const MirState *state, std::string name, std::string filename, std::vector<ParticleVector*> pvs,
                                  real minRadius, real maxRadius, int sampleEvery, int tuneEvery, int dumpEvery,
                                  real3 center, real targetVel, real Kp, real Ki, real Kd)
{
    std::vector<std::string> pvNames;
    if (computeTask) extractPVsNames(pvs, pvNames);
        
    auto simPl = computeTask ?
        std::make_shared<SimulationRadialVelocityControl>(state, name, pvNames, minRadius, maxRadius, 
                                                          sampleEvery, tuneEvery, dumpEvery,
                                                          center, targetVel, Kp, Ki, Kd) :
        nullptr;

    auto postPl = computeTask ?
        nullptr :
        std::make_shared<PostprocessRadialVelocityControl> (name, filename);

    return { simPl, postPl };
}

inline pair_shared< SimulationStats, PostprocessStats >
createStatsPlugin(bool computeTask, const MirState *state, std::string name, std::string filename, int every)
{
    auto simPl  = computeTask ? std::make_shared<SimulationStats> (state, name, every) : nullptr;
    auto postPl = computeTask ? nullptr : std::make_shared<PostprocessStats> (name, filename);

    return { simPl, postPl };
}

inline pair_shared< TemperaturizePlugin, PostprocessPlugin >
createTemperaturizePlugin(bool computeTask, const MirState *state, std::string name, ParticleVector* pv, real kBT, bool keepVelocity)
{
    auto simPl = computeTask ? std::make_shared<TemperaturizePlugin> (state, name, pv->name, kBT, keepVelocity) : nullptr;
    return { simPl, nullptr };
}

inline pair_shared< VirialPressurePlugin, VirialPressureDumper >
createVirialPressurePlugin(bool computeTask, const MirState *state, std::string name, ParticleVector *pv,
                           std::function<real(real3)> region, real3 h, int dumpEvery, std::string path)
{
    auto simPl  = computeTask ? std::make_shared<VirialPressurePlugin> (state, name, pv->name, region, h, dumpEvery)
        : nullptr;
    auto postPl = computeTask ? nullptr : std::make_shared<VirialPressureDumper> (name, path);
    return { simPl, postPl };
}

inline pair_shared< VelocityInletPlugin, PostprocessPlugin >
createVelocityInletPlugin(bool computeTask, const MirState *state, std::string name, ParticleVector *pv,
                          std::function< real(real3)> implicitSurface,
                          std::function<real3(real3)> velocityField,
                          real3 resolution, real numberDensity, real kBT)
{
    auto simPl  = computeTask ?
        std::make_shared<VelocityInletPlugin> (state, name, pv->name,
                                               implicitSurface, velocityField,
                                               make_real3(resolution),
                                               numberDensity, kBT)
        : nullptr;

    return { simPl, nullptr };
}
    
inline pair_shared< WallRepulsionPlugin, PostprocessPlugin >
createWallRepulsionPlugin(bool computeTask, const MirState *state, std::string name, ParticleVector* pv, Wall* wall,
                          real C, real h, real maxForce)
{
    auto simPl = computeTask ? std::make_shared<WallRepulsionPlugin> (state, name, pv->name, wall->name, C, h, maxForce) : nullptr;
    return { simPl, nullptr };
}

inline pair_shared< WallForceCollectorPlugin, WallForceDumperPlugin >
createWallForceCollectorPlugin(bool computeTask, const MirState *state, std::string name, Wall *wall, ParticleVector* pvFrozen,
                               int sampleEvery, int dumpEvery, std::string filename)
{
    auto simPl = computeTask ?
        std::make_shared<WallForceCollectorPlugin> (state, name, wall->name, pvFrozen->name, sampleEvery, dumpEvery) :
        nullptr;

    auto postPl = computeTask ?
        nullptr :
        std::make_shared<WallForceDumperPlugin> (name, filename);
        
    return { simPl, postPl };
}
} // namespace PluginFactory
