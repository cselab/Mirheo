// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/plugins.h>
#include <mirheo/core/pvs/chain_vector.h>
#include <mirheo/core/pvs/object_vector.h>
#include <mirheo/core/pvs/particle_vector.h>
#include <mirheo/core/pvs/rigid_object_vector.h>
#include <mirheo/core/pvs/rod_vector.h>
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

PairPlugin createAddFourRollMillForcePlugin(bool computeTask, const MirState *state,
                                            std::string name, ParticleVector *pv, real intensity);

PairPlugin createAddForcePlugin(bool computeTask, const MirState *state, std::string name, ParticleVector *pv, real3 force);

PairPlugin createAddTorquePlugin(bool computeTask, const MirState *state, std::string name, ParticleVector *pv, real3 torque);

PairPlugin createAnchorParticlesPlugin(bool computeTask, const MirState *state, std::string name, ParticleVector *pv,
                                       std::function<std::vector<real3>(real)> positions,
                                       std::function<std::vector<real3>(real)> velocities,
                                       std::vector<int> pids, int reportEvery, const std::string& path);

PairPlugin createBerendsenThermostatPlugin(bool computeTask, const MirState *state, std::string name,
                                           const std::vector<ParticleVector *> &pv, real tau, real kBT, bool increaseIfLower);

PairPlugin createDensityControlPlugin(bool computeTask, const MirState *state, std::string name,
                                      std::string fname, std::vector<ParticleVector*> pvs,
                                      real targetDensity, std::function<real(real3)> region, real3 resolution,
                                      real levelLo, real levelHi, real levelSpace, real Kp, real Ki, real Kd,
                                      int tuneEvery, int dumpEvery, int sampleEvery);

PairPlugin createDensityOutletPlugin(bool computeTask, const MirState *state, std::string name, std::vector<ParticleVector*> pvs,
                                     real numberDensity, std::function<real(real3)> region, real3 resolution);

PairPlugin createPlaneOutletPlugin(bool computeTask, const MirState *state, std::string name,
                                   std::vector<ParticleVector*> pvs, real4 plane);

PairPlugin createRateOutletPlugin(bool computeTask, const MirState *state, std::string name, std::vector<ParticleVector*> pvs,
                                  real rate, std::function<real(real3)> region, real3 resolution);

PairPlugin createDumpAveragePlugin(bool computeTask, const MirState *state, std::string name,
                                   std::vector<ParticleVector*> pvs, int sampleEvery, int dumpEvery,
                                   real3 binSize, std::vector<std::string> channelNames, std::string path);

PairPlugin createDumpAverageRelativePlugin(bool computeTask, const MirState *state, std::string name,
                                           std::vector<ParticleVector*> pvs,
                                           ObjectVector* relativeToOV, int relativeToId,
                                           int sampleEvery, int dumpEvery, real3 binSize,
                                           std::vector<std::string> channelNames, std::string path);

PairPlugin createDumpMeshPlugin(bool computeTask, const MirState *state, std::string name,
                                ObjectVector* ov, int dumpEvery, std::string path);

PairPlugin createDumpParticlesPlugin(bool computeTask, const MirState *state, std::string name,
                                     ParticleVector *pv, int dumpEvery,
                                     const std::vector<std::string>& channelNames, std::string path);

PairPlugin createDumpParticlesWithMeshPlugin(bool computeTask, const MirState *state, std::string name,
                                             ObjectVector *ov, int dumpEvery,
                                             const std::vector<std::string>& channelNames, std::string path);

PairPlugin createDumpParticlesWithPolylinesPlugin(bool computeTask, const MirState *state, std::string name,
                                                  ChainVector *ov, int dumpEvery,
                                                  const std::vector<std::string>& channelNames, std::string path);

PairPlugin createDumpXYZPlugin(bool computeTask, const MirState *state, std::string name,
                               ParticleVector *pv, int dumpEvery, std::string path);

PairPlugin createDumpObjStats(bool computeTask, const MirState *state, std::string name,
                              ObjectVector *ov, int dumpEvery, std::string filename);

PairPlugin createExchangePVSFluxPlanePlugin(bool computeTask, const MirState *state, std::string name,
                                            ParticleVector *pv1, ParticleVector *pv2, real4 plane);

PairPlugin createExternalMagneticTorquePlugin(bool computeTask, const MirState *state, std::string name,
                                              RigidObjectVector *rov, real3 moment,
                                              std::function<real3(real)> magneticFunction);

PairPlugin createForceSaverPlugin(bool computeTask,  const MirState *state, std::string name, ParticleVector *pv);

PairPlugin createImposeProfilePlugin(bool computeTask,  const MirState *state, std::string name, ParticleVector* pv,
                                     real3 low, real3 high, real3 velocity, real kBT);

PairPlugin createImposeVelocityPlugin(bool computeTask,  const MirState *state, std::string name,
                                      std::vector<ParticleVector*> pvs, int every,
                                      real3 low, real3 high, real3 velocity);

PairPlugin createMagneticDipoleInteractionsPlugin(bool computeTask, const MirState *state, std::string name,
                                                  RigidObjectVector *rov, real3 moment, real mu0);

PairPlugin createMembraneExtraForcePlugin(bool computeTask, const MirState *state, std::string name, ParticleVector *pv, const std::vector<real3>& forces);

PairPlugin createMsdPlugin(bool computeTask, const MirState *state, std::string name, ParticleVector *pv,
                           MirState::TimeType startTime, MirState::TimeType endTime, int dumpEvery, std::string path);

PairPlugin createParticleChannelAveragerPlugin(bool computeTask, const MirState *state, std::string name, ParticleVector *pv,
                                               std::string channelName, std::string averageName, real updateEvery);

PairPlugin createParticleChannelSaverPlugin(bool computeTask, const MirState *state, std::string name, ParticleVector *pv,
                                            std::string channelName, std::string savedName);

PairPlugin createParticleCheckerPlugin(bool computeTask, const MirState *state, std::string name, int checkEvery);

PairPlugin createParticleDisplacementPlugin(bool computeTask, const MirState *state, std::string name, ParticleVector *pv, int updateEvery);

PairPlugin createParticleDragPlugin(bool computeTask, const MirState *state, std::string name, ParticleVector *pv, real drag);

/// Dummy class to have the special value available.
struct PinObjectMock
{
    const static real Unrestricted; ///< Mock unrestricted special value.
};

PairPlugin createPinObjPlugin(bool computeTask, const MirState *state, std::string name, ObjectVector *ov,
                              int dumpEvery, std::string path, real3 velocity, real3 omega);

PairPlugin createPinRodExtremityPlugin(bool computeTask, const MirState *state, std::string name, RodVector *rv, int segmentId,
                                       real fmagn, real3 targetDirection);

PairPlugin createVelocityControlPlugin(bool computeTask, const MirState *state, std::string name, std::string filename, std::vector<ParticleVector*> pvs,
                                       real3 low, real3 high, int sampleEvery, int tuneEvery, int dumpEvery, real3 targetVel, real Kp, real Ki, real Kd);

PairPlugin createRdfPlugin(bool computeTask, const MirState *state, std::string name, ParticleVector *pv, real maxDist, int nbins, std::string basename, int every);

PairPlugin createRmacfPlugin(bool computeTask, const MirState *state, std::string name, ChainVector *cv,
                             MirState::TimeType startTime, MirState::TimeType endTime, int dumpEvery, std::string path);

PairPlugin createStatsPlugin(bool computeTask, const MirState *state, std::string name, int every, const std::vector<ParticleVector*>& pvs, std::string filename);

PairPlugin createTemperaturizePlugin(bool computeTask, const MirState *state, std::string name, ParticleVector* pv, real kBT, bool keepVelocity);

PairPlugin createVacfPlugin(bool computeTask, const MirState *state, std::string name, ParticleVector *pv,
                            MirState::TimeType startTime, MirState::TimeType endTime, int dumpEvery, std::string path);

PairPlugin createVirialPressurePlugin(bool computeTask, const MirState *state, std::string name, ParticleVector *pv,
                                      std::function<real(real3)> region, real3 h, int dumpEvery, std::string path);

PairPlugin createVelocityInletPlugin(bool computeTask, const MirState *state, std::string name, ParticleVector *pv,
                                     std::function< real(real3)> implicitSurface,
                                     std::function<real3(real3)> velocityField,
                                     real3 resolution, real numberDensity, real kBT);

PairPlugin createWallRepulsionPlugin(bool computeTask, const MirState *state, std::string name, ParticleVector* pv, Wall* wall, real C, real h, real maxForce);

PairPlugin createWallForceCollectorPlugin(bool computeTask, const MirState *state, std::string name, Wall *wall, ParticleVector* pvFrozen,
                                          int sampleEvery, int dumpEvery, std::string filename, bool detailedDump);


} // namespace plugin_factory
} // namespace mirheo
