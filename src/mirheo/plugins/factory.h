#pragma once

#include <mirheo/core/plugins.h>
#include <mirheo/core/pvs/object_vector.h>
#include <mirheo/core/pvs/particle_vector.h>
#include <mirheo/core/pvs/rigid_object_vector.h>
#include <mirheo/core/pvs/rod_vector.h>
#include <mirheo/core/snapshot.h>
#include <mirheo/core/walls/interface.h>

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace mirheo
{

namespace plugin_factory
{
using PairPlugin = std::pair<std::shared_ptr<SimulationPlugin>,
                             std::shared_ptr<PostprocessPlugin>>;
    

PairPlugin createAddForcePlugin(bool computeTask, const MirState *state, std::string name, ParticleVector *pv, real3 force);

PairPlugin createAddTorquePlugin(bool computeTask, const MirState *state, std::string name, ParticleVector *pv, real3 torque);

PairPlugin createAnchorParticlesPlugin(bool computeTask, const MirState *state, std::string name, ParticleVector *pv,
                                       std::function<std::vector<real3>(real)> positions,
                                       std::function<std::vector<real3>(real)> velocities,
                                       std::vector<int> pids, int reportEvery, const std::string& path);

PairPlugin createDensityControlPlugin(bool computeTask, const MirState *state, std::string name, std::string fname, std::vector<ParticleVector*> pvs,
                                      real targetDensity, std::function<real(real3)> region, real3 resolution,
                                      real levelLo, real levelHi, real levelSpace, real Kp, real Ki, real Kd,
                                      int tuneEvery, int dumpEvery, int sampleEvery);

PairPlugin createDensityOutletPlugin(bool computeTask, const MirState *state, std::string name, std::vector<ParticleVector*> pvs,
                                     real numberDensity, std::function<real(real3)> region, real3 resolution);

PairPlugin createPlaneOutletPlugin(bool computeTask, const MirState *state, std::string name,
                                   std::vector<ParticleVector*> pvs, real4 plane);

PairPlugin createRateOutletPlugin(bool computeTask, const MirState *state, std::string name, std::vector<ParticleVector*> pvs,
                                  real rate, std::function<real(real3)> region, real3 resolution);

PairPlugin createDumpAveragePlugin(bool computeTask, const MirState *state, std::string name, std::vector<ParticleVector*> pvs,
                                   int sampleEvery, int dumpEvery, real3 binSize, std::vector<std::string> channelNames, std::string path);

PairPlugin createDumpAverageRelativePlugin(bool computeTask, const MirState *state, std::string name, std::vector<ParticleVector*> pvs,
                                           ObjectVector* relativeToOV, int relativeToId,
                                           int sampleEvery, int dumpEvery, real3 binSize,
                                           std::vector<std::string> channelNames, std::string path);

PairPlugin createDumpMeshPlugin(bool computeTask, const MirState *state, std::string name, ObjectVector* ov, int dumpEvery, std::string path);

PairPlugin createDumpParticlesPlugin(bool computeTask, const MirState *state, std::string name, ParticleVector *pv, int dumpEvery,
                                     const std::vector<std::string>& channelNames, std::string path);

PairPlugin createDumpParticlesWithMeshPlugin(bool computeTask, const MirState *state, std::string name, ObjectVector *ov, int dumpEvery,
                                             const std::vector<std::string>& channelNames, std::string path);

PairPlugin createDumpXYZPlugin(bool computeTask, const MirState *state, std::string name, ParticleVector *pv, int dumpEvery, std::string path);

PairPlugin createDumpObjStats(bool computeTask, const MirState *state, std::string name, ObjectVector *ov, int dumpEvery, std::string path);

PairPlugin createExchangePVSFluxPlanePlugin(bool computeTask, const MirState *state, std::string name, ParticleVector *pv1, ParticleVector *pv2, real4 plane);

PairPlugin createForceSaverPlugin(bool computeTask,  const MirState *state, std::string name, ParticleVector *pv);

PairPlugin createImposeProfilePlugin(bool computeTask,  const MirState *state, std::string name, ParticleVector* pv, 
                                     real3 low, real3 high, real3 velocity, real kBT);

PairPlugin createImposeVelocityPlugin(bool computeTask,  const MirState *state, std::string name,
                                      std::vector<ParticleVector*> pvs, int every,
                                      real3 low, real3 high, real3 velocity);

PairPlugin createMagneticOrientationPlugin(bool computeTask, const MirState *state, std::string name, RigidObjectVector *rov, real3 moment,
                                           std::function<real3(real)> magneticFunction);

PairPlugin createMembraneExtraForcePlugin(bool computeTask, const MirState *state, std::string name, ParticleVector *pv, const std::vector<real3>& forces);

PairPlugin createParticleChannelSaverPlugin(bool computeTask, const MirState *state, std::string name, ParticleVector *pv,
                                            std::string channelName, std::string savedName);

PairPlugin createParticleCheckerPlugin(bool computeTask, const MirState *state, std::string name, int checkEvery);

PairPlugin createParticleDisplacementPlugin(bool computeTask, const MirState *state, std::string name, ParticleVector *pv, int updateEvery);

PairPlugin createParticleDragPlugin(bool computeTask, const MirState *state, std::string name, ParticleVector *pv, real drag);

struct PinObjectMock
{
    const static real Unrestricted;
};

PairPlugin createPinObjPlugin(bool computeTask, const MirState *state, std::string name, ObjectVector *ov,
                              int dumpEvery, std::string path, real3 velocity, real3 omega);

PairPlugin createPinRodExtremityPlugin(bool computeTask, const MirState *state, std::string name, RodVector *rv, int segmentId,
                                       real fmagn, real3 targetDirection);

PairPlugin createVelocityControlPlugin(bool computeTask, const MirState *state, std::string name, std::string filename, std::vector<ParticleVector*> pvs,
                                       real3 low, real3 high, int sampleEvery, int tuneEvery, int dumpEvery, real3 targetVel, real Kp, real Ki, real Kd);

PairPlugin createStatsPlugin(bool computeTask, const MirState *state, std::string name, std::string filename, int every);

PairPlugin createTemperaturizePlugin(bool computeTask, const MirState *state, std::string name, ParticleVector* pv, real kBT, bool keepVelocity);

PairPlugin createVirialPressurePlugin(bool computeTask, const MirState *state, std::string name, ParticleVector *pv,
                                      std::function<real(real3)> region, real3 h, int dumpEvery, std::string path);

PairPlugin createVelocityInletPlugin(bool computeTask, const MirState *state, std::string name, ParticleVector *pv,
                                     std::function< real(real3)> implicitSurface,
                                     std::function<real3(real3)> velocityField,
                                     real3 resolution, real numberDensity, real kBT);

PairPlugin createWallRepulsionPlugin(bool computeTask, const MirState *state, std::string name, ParticleVector* pv, Wall* wall, real C, real h, real maxForce);

PairPlugin createWallForceCollectorPlugin(bool computeTask, const MirState *state, std::string name, Wall *wall, ParticleVector* pvFrozen,
                                          int sampleEvery, int dumpEvery, std::string filename);


/** \brief Construct a simulation & postprocess plugin pair given their ConfigObjects.
    \param [in] computeTask True if the current rank is a compute rank, false otherwise.
    \param [in] state The Mirheo state object.
    \param [in,out] loader The \c Loader object. Provides load context and unserialization functions.
    \param [in] sim The ConfigObject describing the simulation part of the plugin pair (optional).
    \param [in] post The ConfigObject describing the postprocess part of the plugin pair (optional).

    This factory function tries to match the given type names (`__type` field of ConfigObjects) with builtin plugins names.
    If the match is found, the corresponding plugins are created and returned.
    The \c ConfigObject arguments are optional, but at least one of them has to be given.
    Depending on whether the current rank is a compute or a postprocess rank, the simulation or postprocess plugin will be created.

    If the type names are not recognized, the factory returns null pointers.
    The error should be diagnosed by the caller.

    \return An optional-like 3-tuple (bool matchFound, simulation plugin shared pointer, postprocess plugin shared pointer).
 */
PluginFactoryContainer::OptionalPluginPair loadPlugins(
        bool computeTask, const MirState *state, Loader& loader,
        const ConfigObject *sim = nullptr, const ConfigObject* post = nullptr);

/// Helper type for registering `loadPlugins` to the core.
struct PluginRegistrant
{
    PluginRegistrant();
};

} // namespace plugin_factory
} // namespace mirheo
