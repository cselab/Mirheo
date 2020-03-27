#include "factory.h"

#include "add_force.h"
#include "add_torque.h"
#include "anchor_particle.h"
#include "average_flow.h"
#include "average_relative_flow.h"
#include "berendsen_thermostat.h"
#include "channel_dumper.h"
#include "outlet.h"
#include "density_control.h"
#include "displacement.h"
#include "dump_mesh.h"
#include "dump_obj_stats.h"
#include "dump_particles.h"
#include "dump_particles_with_mesh.h"
#include "dump_xyz.h"
#include "exchange_pvs_flux_plane.h"
#include "force_saver.h"
#include "impose_profile.h"
#include "impose_velocity.h"
#include "magnetic_orientation.h"
#include "membrane_extra_force.h"
#include "particle_channel_saver.h"
#include "particle_checker.h"
#include "particle_drag.h"
#include "pin_object.h"
#include "pin_rod_extremity.h"
#include "stats.h"
#include "temperaturize.h"
#include "velocity_control.h"
#include "velocity_inlet.h"
#include "virial_pressure.h"
#include "wall_force_collector.h"
#include "wall_repulsion.h"

namespace mirheo
{
namespace plugin_factory
{

static std::vector<std::string> extractPVNames(const std::vector<ParticleVector*>& pvs)
{
    std::vector<std::string> pvNames;
    pvNames.reserve(pvs.size());
    for (auto &pv : pvs)
        pvNames.push_back(pv->getName());
    return pvNames;
}

PairPlugin createAddForcePlugin(bool computeTask, const MirState *state, std::string name, ParticleVector *pv, real3 force)
{
    auto simPl = computeTask ? std::make_shared<AddForcePlugin> (state, name, pv->getName(), force) : nullptr;
    return { simPl, nullptr };
}

PairPlugin createAddTorquePlugin(bool computeTask, const MirState *state, std::string name, ParticleVector *pv, real3 torque)
{
    auto simPl = computeTask ? std::make_shared<AddTorquePlugin> (state, name, pv->getName(), torque) : nullptr;
    return { simPl, nullptr };
}

PairPlugin createAnchorParticlesPlugin(bool computeTask, const MirState *state, std::string name, ParticleVector *pv,
                                       std::function<std::vector<real3>(real)> positions,
                                       std::function<std::vector<real3>(real)> velocities,
                                       std::vector<int> pids, int reportEvery, const std::string& path)
{
    auto simPl = computeTask ?
        std::make_shared<AnchorParticlesPlugin> (state, name, pv->getName(),
                                                 positions, velocities,
                                                 pids, reportEvery)
        : nullptr;

    auto postPl = computeTask ?
        nullptr :
        std::make_shared<AnchorParticlesStatsPlugin> (name, path);
    
    return { simPl, postPl };
}

PairPlugin createBerendsenThermostatPlugin(
        bool computeTask, const MirState *state, std::string name,
        const std::vector<ParticleVector *> &pvs, real tau, real T, real kBT, bool increaseIfLower)
{
    if (kBT == 0 && T == 0)
        throw std::invalid_argument("At least one of `kBT` and `T` must be set.");
    if (kBT != 0 && T != 0)
        throw std::invalid_argument("Cannot set both `kBT` and `T`.");
    if (computeTask && T != 0) {
        if (!state->units.isSet())
            throw std::invalid_argument("Cannot use `T` without unit conversion factors.");
        kBT = T * state->units.joulesToMirheo(1.380649e-23_r); // kB
    }
    return {
        computeTask ? std::make_shared<BerendsenThermostatPlugin>(
                state, name, extractPVNames(pvs), kBT, tau, increaseIfLower) : nullptr,
        nullptr
    };
}

PairPlugin createDensityControlPlugin(bool computeTask, const MirState *state, std::string name, std::string fname, std::vector<ParticleVector*> pvs,
                                      real targetDensity, std::function<real(real3)> region, real3 resolution,
                                      real levelLo, real levelHi, real levelSpace, real Kp, real Ki, real Kd,
                                      int tuneEvery, int dumpEvery, int sampleEvery)
{
    auto simPl = computeTask ?
        std::make_shared<DensityControlPlugin> (state, name, extractPVNames(pvs), targetDensity,
                                                region, resolution, levelLo, levelHi, levelSpace,
                                                Kp, Ki, Kd, tuneEvery, dumpEvery, sampleEvery) :
        nullptr;

    auto postPl = computeTask ?
        nullptr :
        std::make_shared<PostprocessDensityControl> (name, fname);
    
    return { simPl, postPl };
}

PairPlugin createDensityOutletPlugin(bool computeTask, const MirState *state, std::string name, std::vector<ParticleVector*> pvs,
                                     real numberDensity, std::function<real(real3)> region, real3 resolution)
{
    auto simPl = computeTask ?
        std::make_shared<DensityOutletPlugin> (
                state, name, extractPVNames(pvs), numberDensity, region, resolution)
        : nullptr;
    return { simPl, nullptr };
}

PairPlugin createPlaneOutletPlugin(bool computeTask, const MirState *state, std::string name,
                                   std::vector<ParticleVector*> pvs, real4 plane)
{
    if (!computeTask)
        return { nullptr, nullptr };

    return { std::make_shared<PlaneOutletPlugin> (
            state, name, extractPVNames(pvs), plane), nullptr };
}

PairPlugin createRateOutletPlugin(bool computeTask, const MirState *state, std::string name, std::vector<ParticleVector*> pvs,
                                  real rate, std::function<real(real3)> region, real3 resolution)
{
    auto simPl = computeTask ?
        std::make_shared<RateOutletPlugin> (state, name, extractPVNames(pvs), rate, region, resolution)
        : nullptr;
    return { simPl, nullptr };
}

PairPlugin createDumpAveragePlugin(bool computeTask, const MirState *state, std::string name, std::vector<ParticleVector*> pvs,
                                   int sampleEvery, int dumpEvery, real3 binSize,
                                   std::vector<std::string> channelNames, std::string path)
{
    auto simPl  = computeTask ?
        std::make_shared<Average3D> (state, name, extractPVNames(pvs), channelNames, sampleEvery, dumpEvery, binSize) :
        nullptr;

    auto postPl = computeTask ? nullptr : std::make_shared<UniformCartesianDumper> (name, path);

    return { simPl, postPl };
}

PairPlugin createDumpAverageRelativePlugin(bool computeTask, const MirState *state, std::string name, std::vector<ParticleVector*> pvs,
                                           ObjectVector* relativeToOV, int relativeToId,
                                           int sampleEvery, int dumpEvery, real3 binSize,
                                           std::vector<std::string> channelNames, std::string path)
{
    auto simPl  = computeTask ?
        std::make_shared<AverageRelative3D> (state, name, extractPVNames(pvs),
                                             channelNames, sampleEvery, dumpEvery,
                                             binSize, relativeToOV->getName(), relativeToId) :
        nullptr;

    auto postPl = computeTask ? nullptr : std::make_shared<UniformCartesianDumper> (name, path);

    return { simPl, postPl };
}

PairPlugin createDumpMeshPlugin(bool computeTask, const MirState *state, std::string name, ObjectVector* ov, int dumpEvery, std::string path)
{
    auto simPl  = computeTask ? std::make_shared<MeshPlugin> (state, name, ov->getName(), dumpEvery) : nullptr;
    auto postPl = computeTask ? nullptr : std::make_shared<MeshDumper> (name, path);

    return { simPl, postPl };
}

PairPlugin createDumpParticlesPlugin(bool computeTask, const MirState *state, std::string name, ParticleVector *pv, int dumpEvery,
                                     const std::vector<std::string>& channelNames, std::string path)
{
    auto simPl  = computeTask ? std::make_shared<ParticleSenderPlugin> (state, name, pv->getName(), dumpEvery, channelNames) : nullptr;
    auto postPl = computeTask ? nullptr : std::make_shared<ParticleDumperPlugin> (name, path);

    return { simPl, postPl };
}

PairPlugin createDumpParticlesWithMeshPlugin(bool computeTask, const MirState *state, std::string name, ObjectVector *ov, int dumpEvery,
                                             const std::vector<std::string>& channelNames, std::string path)
{
    auto simPl  = computeTask ? std::make_shared<ParticleWithMeshSenderPlugin> (state, name, ov->getName(), dumpEvery, channelNames) : nullptr;
    auto postPl = computeTask ? nullptr : std::make_shared<ParticleWithMeshDumperPlugin> (name, path);

    return { simPl, postPl };
}

PairPlugin createDumpXYZPlugin(bool computeTask, const MirState *state, std::string name, ParticleVector* pv, int dumpEvery, std::string path)
{
    auto simPl  = computeTask ? std::make_shared<XYZPlugin> (state, name, pv->getName(), dumpEvery) : nullptr;
    auto postPl = computeTask ? nullptr : std::make_shared<XYZDumper> (name, path);

    return { simPl, postPl };
}

PairPlugin createDumpObjStats(bool computeTask, const MirState *state, std::string name, ObjectVector* ov, int dumpEvery, std::string path)
{
    auto simPl  = computeTask ? std::make_shared<ObjStatsPlugin> (state, name, ov->getName(), dumpEvery) : nullptr;
    auto postPl = computeTask ? nullptr : std::make_shared<ObjStatsDumper> (name, path);

    return { simPl, postPl };
}

PairPlugin createExchangePVSFluxPlanePlugin(bool computeTask, const MirState *state, std::string name, ParticleVector *pv1, ParticleVector *pv2, real4 plane)
{
    auto simPl = computeTask ?
        std::make_shared<ExchangePVSFluxPlanePlugin> (state, name, pv1->getName(), pv2->getName(), plane) : nullptr;
        
    return { simPl, nullptr };    
}

PairPlugin createForceSaverPlugin(bool computeTask,  const MirState *state, std::string name, ParticleVector *pv)
{
    auto simPl = computeTask ? std::make_shared<ForceSaverPlugin> (state, name, pv->getName()) : nullptr;
    return { simPl, nullptr };
}

PairPlugin createImposeProfilePlugin(bool computeTask,  const MirState *state, std::string name, ParticleVector* pv, 
                                     real3 low, real3 high, real3 velocity, real kBT)
{
    auto simPl = computeTask ?
        std::make_shared<ImposeProfilePlugin> (state, name, pv->getName(), low, high, velocity, kBT) :
        nullptr;
            
    return { simPl, nullptr };
}

PairPlugin createImposeVelocityPlugin(bool computeTask,  const MirState *state, std::string name,
                                      std::vector<ParticleVector*> pvs, int every,
                                      real3 low, real3 high, real3 velocity)
{
    auto simPl = computeTask ?
        std::make_shared<ImposeVelocityPlugin> (state, name, extractPVNames(pvs), low, high, velocity, every) :
        nullptr;
                                    
    return { simPl, nullptr };
}

PairPlugin createMagneticOrientationPlugin(bool computeTask, const MirState *state, std::string name, RigidObjectVector *rov, real3 moment,
                                           std::function<real3(real)> magneticFunction)
{
    auto simPl = computeTask ?
        std::make_shared<MagneticOrientationPlugin>(state, name, rov->getName(), moment, magneticFunction)
        : nullptr;

    return { simPl, nullptr };
}

PairPlugin createMembraneExtraForcePlugin(bool computeTask, const MirState *state, std::string name, ParticleVector *pv, const std::vector<real3>& forces)
{
    auto simPl = computeTask ?
        std::make_shared<MembraneExtraForcePlugin> (state, name, pv->getName(), forces) : nullptr;

    return { simPl, nullptr };
}

PairPlugin createParticleChannelSaverPlugin(bool computeTask, const MirState *state, std::string name, ParticleVector *pv,
                                            std::string channelName, std::string savedName)
{
    auto simPl = computeTask ? std::make_shared<ParticleChannelSaverPlugin> (state, name, pv->getName(), channelName, savedName) : nullptr;
    return { simPl, nullptr };
}

PairPlugin createParticleCheckerPlugin(bool computeTask, const MirState *state, std::string name, int checkEvery)
{
    auto simPl = computeTask ? std::make_shared<ParticleCheckerPlugin> (state, name, checkEvery) : nullptr;
    return { simPl, nullptr };
}

PairPlugin createParticleDisplacementPlugin(bool computeTask, const MirState *state, std::string name, ParticleVector *pv, int updateEvery)
{
    auto simPl = computeTask ?
        std::make_shared<ParticleDisplacementPlugin> (state, name, pv->getName(), updateEvery) :
        nullptr;
    return { simPl, nullptr };
}

PairPlugin createParticleDragPlugin(bool computeTask, const MirState *state, std::string name, ParticleVector *pv, real drag)
{
    auto simPl = computeTask ?
        std::make_shared<ParticleDragPlugin> (state, name, pv->getName(), drag) :
        nullptr;
    return { simPl, nullptr };
}

const real PinObjectMock::Unrestricted = PinObjectPlugin::Unrestricted;

PairPlugin createPinObjPlugin(bool computeTask, const MirState *state, std::string name, ObjectVector *ov,
                              int dumpEvery, std::string path, real3 velocity, real3 omega)
{
    auto simPl  = computeTask ? std::make_shared<PinObjectPlugin> (state, name, ov->getName(), velocity, omega, dumpEvery) : 
        nullptr;
    auto postPl = computeTask ? nullptr : std::make_shared<ReportPinObjectPlugin> (name, path);

    return { simPl, postPl };
}

PairPlugin createPinRodExtremityPlugin(bool computeTask, const MirState *state, std::string name, RodVector *rv, int segmentId,
                                       real fmagn, real3 targetDirection)
{
    auto simPl  = computeTask ?
        std::make_shared<PinRodExtremityPlugin> (state, name, rv->getName(), segmentId, fmagn, targetDirection) : 
        nullptr;

    return { simPl, nullptr };
}

PairPlugin createVelocityControlPlugin(bool computeTask, const MirState *state, std::string name, std::string filename, std::vector<ParticleVector*> pvs,
                                       real3 low, real3 high, int sampleEvery, int tuneEvery, int dumpEvery, real3 targetVel, real Kp, real Ki, real Kd)
{
    auto simPl = computeTask ?
        std::make_shared<SimulationVelocityControl>(state, name, extractPVNames(pvs), low, high,
                                                    sampleEvery, tuneEvery, dumpEvery,
                                                    targetVel, Kp, Ki, Kd) :
        nullptr;

    auto postPl = computeTask ?
        nullptr :
        std::make_shared<PostprocessVelocityControl> (name, filename);

    return { simPl, postPl };
}

PairPlugin createStatsPlugin(bool computeTask, const MirState *state, std::string name, std::string filename, int every)
{
    auto simPl  = computeTask ? std::make_shared<SimulationStats> (state, name, every) : nullptr;
    auto postPl = computeTask ? nullptr : std::make_shared<PostprocessStats> (name, filename);

    return { simPl, postPl };
}

PairPlugin createTemperaturizePlugin(bool computeTask, const MirState *state, std::string name, ParticleVector* pv, real kBT, bool keepVelocity)
{
    auto simPl = computeTask ? std::make_shared<TemperaturizePlugin> (state, name, pv->getName(), kBT, keepVelocity) : nullptr;
    return { simPl, nullptr };
}

PairPlugin createVirialPressurePlugin(bool computeTask, const MirState *state, std::string name, ParticleVector *pv,
                                      std::function<real(real3)> region, real3 h, int dumpEvery, std::string path)
{
    auto simPl  = computeTask ? std::make_shared<VirialPressurePlugin> (state, name, pv->getName(), region, h, dumpEvery)
        : nullptr;
    auto postPl = computeTask ? nullptr : std::make_shared<VirialPressureDumper> (name, path);
    return { simPl, postPl };
}

PairPlugin createVelocityInletPlugin(bool computeTask, const MirState *state, std::string name, ParticleVector *pv,
                                     std::function< real(real3)> implicitSurface,
                                     std::function<real3(real3)> velocityField,
                                     real3 resolution, real numberDensity, real kBT)
{
    auto simPl  = computeTask ?
        std::make_shared<VelocityInletPlugin> (state, name, pv->getName(),
                                               implicitSurface, velocityField,
                                               make_real3(resolution),
                                               numberDensity, kBT)
        : nullptr;

    return { simPl, nullptr };
}
    
PairPlugin createWallRepulsionPlugin(bool computeTask, const MirState *state, std::string name, ParticleVector* pv, Wall* wall,
                                     real C, real h, real maxForce)
{
    auto simPl = computeTask ? std::make_shared<WallRepulsionPlugin> (state, name, pv->getName(), wall->getName(), C, h, maxForce) : nullptr;
    return { simPl, nullptr };
}

PairPlugin createWallForceCollectorPlugin(bool computeTask, const MirState *state, std::string name, Wall *wall, ParticleVector* pvFrozen,
                                          int sampleEvery, int dumpEvery, std::string filename)
{
    auto simPl = computeTask ?
        std::make_shared<WallForceCollectorPlugin> (state, name, wall->getName(), pvFrozen->getName(), sampleEvery, dumpEvery) :
        nullptr;

    auto postPl = computeTask ?
        nullptr :
        std::make_shared<WallForceDumperPlugin> (name, filename);
        
    return { simPl, postPl };
}

PluginFactoryContainer::OptionalPluginPair loadPlugins(
        bool computeTask, const MirState *state, Loader& loader,
        const ConfigObject *sim, const ConfigObject* post)
{
    std::string simType  = sim  ? sim->at("__type").getString()  : std::string();
    std::string postType = post ? post->at("__type").getString() : std::string();

    // Create a pair of sim and post plugins if the type names match.
#define MIR_LOAD_PLUGIN_PAIR(A, B) do {                                           \
        if (simType == #A && postType == #B) {                                    \
            if (computeTask)                                                      \
                return {true, std::make_shared<A>(state, loader, *sim), nullptr}; \
            else                                                                  \
                return {true, nullptr, std::make_shared<B>(loader, *post)};       \
        }                                                                         \
    } while (0)

    // Create a simulation-only plugin, if the type names match.
#define MIR_LOAD_SIM_PLUGIN(A) do {                                            \
        if (simType == #A && postType.empty()) {                               \
            return {true,                                                      \
                    computeTask ? std::make_shared<A>(state, loader, *sim)     \
                                : nullptr,                                     \
                    nullptr};                                                  \
        }                                                                      \
    } while (0)

    // List all supported plugins.
    MIR_LOAD_PLUGIN_PAIR(MeshPlugin, MeshDumper);
    MIR_LOAD_PLUGIN_PAIR(ParticleSenderPlugin, ParticleDumperPlugin);
    MIR_LOAD_PLUGIN_PAIR(SimulationStats, PostprocessStats);
    MIR_LOAD_SIM_PLUGIN(BerendsenThermostatPlugin);
    MIR_LOAD_SIM_PLUGIN(ForceSaverPlugin);
    MIR_LOAD_SIM_PLUGIN(MembraneExtraForcePlugin);
    MIR_LOAD_SIM_PLUGIN(WallRepulsionPlugin);

#undef MIR_LOAD_SIM_PLUGIN
#undef MIR_LOAD_PLUGIN_PAIR

    return {false, nullptr, nullptr};
}

static bool pluginsRegistered_ = false;
static PluginRegistrant registrant_;

PluginRegistrant::PluginRegistrant()
{
    if (!pluginsRegistered_) {
        PluginFactoryContainer::get().registerPluginFactory(loadPlugins);
        pluginsRegistered_ = true;
    }
}

} // namespace plugin_factory
} // namespace mirheo
