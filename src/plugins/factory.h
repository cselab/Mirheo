#pragma once

#include <memory>

#include <core/pvs/object_vector.h>
#include <core/pvs/particle_vector.h>
#include <core/utils/pytypes.h>
#include <core/walls/interface.h>

#include "interface.h"

#include "add_force.h"
#include "add_torque.h"
#include "average_flow.h"
#include "average_relative_flow.h"
#include "channel_dumper.h"
#include "dump_mesh.h"
#include "dump_obj_position.h"
#include "dump_particles.h"
#include "dump_particles_with_mesh.h"
#include "dumpxyz.h"
#include "exchange_pvs_flux_plane.h"
#include "force_saver.h"
#include "impose_profile.h"
#include "impose_velocity.h"
#include "magnetic_orientation.h"
#include "membrane_extra_force.h"
#include "pin_object.h"
#include "stats.h"
#include "temperaturize.h"
#include "velocity_control.h"
#include "wall_repulsion.h"

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
            else if (typeStr == "vector")             types.push_back(Average3D::ChannelType::Vector_float3);
            else if (typeStr == "vector_from_float4") types.push_back(Average3D::ChannelType::Vector_float4);
            else if (typeStr == "vector_from_float8") types.push_back(Average3D::ChannelType::Vector_2xfloat4);
            else if (typeStr == "tensor6")            types.push_back(Average3D::ChannelType::Tensor6);
            else die("Unable to get parse channel type '%s'", typeStr.c_str());
        }
    }

    static void extractPVsNames(const std::vector<ParticleVector*>& pvs, std::vector<std::string>& pvNames)
    {
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

    

    
    static pair_shared< AddForcePlugin, PostprocessPlugin >
    createAddForcePlugin(bool computeTask, std::string name, ParticleVector* pv, PyTypes::float3 force)
    {
        auto simPl = computeTask ? std::make_shared<AddForcePlugin> (name, pv->name, make_float3(force)) : nullptr;
        return { simPl, nullptr };
    }

    static pair_shared< AddTorquePlugin, PostprocessPlugin >
    createAddTorquePlugin(bool computeTask, std::string name, ParticleVector* pv, PyTypes::float3 torque)
    {
        auto simPl = computeTask ? std::make_shared<AddTorquePlugin> (name, pv->name, make_float3(torque)) : nullptr;
        return { simPl, nullptr };
    }

    static pair_shared< Average3D, UniformCartesianDumper >
    createDumpAveragePlugin(bool computeTask, std::string name, std::vector<ParticleVector*> pvs,
                            int sampleEvery, int dumpEvery, PyTypes::float3 binSize,
                            std::vector< std::pair<std::string, std::string> > channels,
                            std::string path)
    {
        std::vector<std::string> names, pvNames;
        std::vector<Average3D::ChannelType> types;

        extractChannelsInfos(channels, names, types);
        
        if (computeTask) extractPVsNames(pvs, pvNames);
        
        auto simPl  = computeTask ?
            std::make_shared<Average3D> (name, pvNames, names, types, sampleEvery, dumpEvery, make_float3(binSize)) :
            nullptr;

        auto postPl = computeTask ? nullptr : std::make_shared<UniformCartesianDumper> (name, path);

        return { simPl, postPl };
    }

    static pair_shared< AverageRelative3D, UniformCartesianDumper >
    createDumpAverageRelativePlugin(bool computeTask, std::string name, std::vector<ParticleVector*> pvs,
                                    ObjectVector* relativeToOV, int relativeToId,
                                    int sampleEvery, int dumpEvery, PyTypes::float3 binSize,
                                    std::vector< std::pair<std::string, std::string> > channels,
                                    std::string path)
    {
        std::vector<std::string> names, pvNames;
        std::vector<Average3D::ChannelType> types;

        extractChannelsInfos(channels, names, types);

        if (computeTask) extractPVsNames(pvs, pvNames);
    
        auto simPl  = computeTask ?
            std::make_shared<AverageRelative3D> (name, pvNames,
                                                 names, types, sampleEvery, dumpEvery,
                                                 make_float3(binSize), relativeToOV->name, relativeToId) :
            nullptr;

        auto postPl = computeTask ? nullptr : std::make_shared<UniformCartesianDumper> (name, path);

        return { simPl, postPl };
    }

    static pair_shared< MeshPlugin, MeshDumper >
    createDumpMeshPlugin(bool computeTask, std::string name, ObjectVector* ov, int dumpEvery, std::string path)
    {
        auto simPl  = computeTask ? std::make_shared<MeshPlugin> (name, ov->name, dumpEvery) : nullptr;
        auto postPl = computeTask ? nullptr : std::make_shared<MeshDumper> (name, path);

        return { simPl, postPl };
    }

    static pair_shared< ParticleSenderPlugin, ParticleDumperPlugin >
    createDumpParticlesPlugin(bool computeTask, std::string name, ParticleVector *pv, int dumpEvery,
                              std::vector< std::pair<std::string, std::string> > channels, std::string path)
    {
        std::vector<std::string> names;
        std::vector<ParticleSenderPlugin::ChannelType> types;

        extractChannelInfos(channels, names, types);
        
        auto simPl  = computeTask ? std::make_shared<ParticleSenderPlugin> (name, pv->name, dumpEvery, names, types) : nullptr;
        auto postPl = computeTask ? nullptr : std::make_shared<ParticleDumperPlugin> (name, path);

        return { simPl, postPl };
    }

    static pair_shared< ParticleWithMeshSenderPlugin, ParticleWithMeshDumperPlugin >
    createDumpParticlesWithMeshPlugin(bool computeTask, std::string name, ObjectVector *ov, int dumpEvery,
                                      std::vector< std::pair<std::string, std::string> > channels, std::string path)
    {
        std::vector<std::string> names;
        std::vector<ParticleSenderPlugin::ChannelType> types;

        extractChannelInfos(channels, names, types);
        
        auto simPl  = computeTask ? std::make_shared<ParticleWithMeshSenderPlugin> (name, ov->name, dumpEvery, names, types) : nullptr;
        auto postPl = computeTask ? nullptr : std::make_shared<ParticleWithMeshDumperPlugin> (name, path);

        return { simPl, postPl };
    }

    static pair_shared< XYZPlugin, XYZDumper >
        createDumpXYZPlugin(bool computeTask, std::string name, ParticleVector* pv, int dumpEvery, std::string path)
    {
        auto simPl  = computeTask ? std::make_shared<XYZPlugin> (name, pv->name, dumpEvery) : nullptr;
        auto postPl = computeTask ? nullptr : std::make_shared<XYZDumper> (name, path);

        return { simPl, postPl };
    }

    static pair_shared< ObjPositionsPlugin, ObjPositionsDumper >
        createDumpObjPosition(bool computeTask, std::string name, ObjectVector* ov, int dumpEvery, std::string path)
    {
        auto simPl  = computeTask ? std::make_shared<ObjPositionsPlugin> (name, ov->name, dumpEvery) : nullptr;
        auto postPl = computeTask ? nullptr : std::make_shared<ObjPositionsDumper> (name, path);

        return { simPl, postPl };
    }

    static pair_shared< ExchangePVSFluxPlanePlugin, PostprocessPlugin >
    createExchangePVSFluxPlanePlugin(bool computeTask, std::string name, ParticleVector *pv1, ParticleVector *pv2, PyTypes::float4 plane)
    {
        auto simPl = computeTask ?
            std::make_shared<ExchangePVSFluxPlanePlugin> (name, pv1->name, pv2->name, make_float4(plane)) : nullptr;
        
        return { simPl, nullptr };    
    }

    static pair_shared< ForceSaverPlugin, PostprocessPlugin >
    createForceSaverPlugin(bool computeTask,  std::string name, ParticleVector *pv)
    {
        auto simPl = computeTask ? std::make_shared<ForceSaverPlugin> (name, pv->name) : nullptr;
        return { simPl, nullptr };
    }

    static pair_shared< ImposeProfilePlugin, PostprocessPlugin >
    createImposeProfilePlugin(bool computeTask,  std::string name, ParticleVector* pv, 
                              PyTypes::float3 low, PyTypes::float3 high, PyTypes::float3 velocity, float kbt)
    {
        auto simPl = computeTask ?
            std::make_shared<ImposeProfilePlugin> (name, pv->name, make_float3(low), make_float3(high), make_float3(velocity), kbt) :
            nullptr;
            
        return { simPl, nullptr };
    }

    static pair_shared< ImposeVelocityPlugin, PostprocessPlugin >
    createImposeVelocityPlugin(bool computeTask, 
                               std::string name, std::vector<ParticleVector*> pvs, int every,
                               PyTypes::float3 low, PyTypes::float3 high, PyTypes::float3 velocity)
    {
        std::vector<std::string> pvNames;
        if (computeTask) extractPVsNames(pvs, pvNames);
            
        auto simPl = computeTask ?
            std::make_shared<ImposeVelocityPlugin> (name, pvNames, make_float3(low), make_float3(high), make_float3(velocity), every) :
            nullptr;
                                    
        return { simPl, nullptr };
    }

    static pair_shared< MagneticOrientationPlugin, PostprocessPlugin >
    createMagneticOrientationPlugin(bool computeTask, std::string name, RigidObjectVector *rov, PyTypes::float3 moment,
                                    std::function<PyTypes::float3(float)> magneticFunction)
                                    //MagneticOrientationPlugin::UniformMagneticFunc magneticFunction)
    {
        auto simPl = computeTask ?
            std::make_shared<MagneticOrientationPlugin>(name, rov->name, make_float3(moment),
                                                        [magneticFunction](float t)
                                                        {return make_float3(magneticFunction(t));})
            : nullptr;

        return { simPl, nullptr };
    }

    static pair_shared< MembraneExtraForcePlugin, PostprocessPlugin >
    createMembraneExtraForcePlugin(bool computeTask, std::string name, ParticleVector *pv, PyTypes::VectorOfFloat3 forces)
    {
        auto simPl = computeTask ?
            std::make_shared<MembraneExtraForcePlugin> (name, pv->name, forces) : nullptr;

        return { simPl, nullptr };
    }

    static pair_shared< PinObjectPlugin, ReportPinObjectPlugin >
    createPinObjPlugin(bool computeTask, std::string name, ObjectVector* ov,
                       int dumpEvery, std::string path,
                       PyTypes::float3 velocity, PyTypes::float3 omega)
    {
        auto simPl  = computeTask ? std::make_shared<PinObjectPlugin> (name, ov->name,
                                                                       make_float3(velocity), make_float3(omega),
                                                                       dumpEvery) : 
            nullptr;
        auto postPl = computeTask ? nullptr : std::make_shared<ReportPinObjectPlugin> (name, path);

        return { simPl, postPl };
    }

    static pair_shared< SimulationVelocityControl, PostprocessVelocityControl >
    createSimulationVelocityControlPlugin(bool computeTask, std::string name, std::string filename, std::vector<ParticleVector*> pvs,
                                          PyTypes::float3 low, PyTypes::float3 high,
                                          int sampleEvery, int tuneEvery, int dumpEvery,
                                          PyTypes::float3 targetVel, float Kp, float Ki, float Kd)
    {
        std::vector<std::string> pvNames;
        if (computeTask) extractPVsNames(pvs, pvNames);
        
        auto simPl = computeTask ?
            std::make_shared<SimulationVelocityControl>(name, pvNames, make_float3(low), make_float3(high),
                                                        sampleEvery, tuneEvery, dumpEvery,
                                                        make_float3(targetVel), Kp, Ki, Kd) :
            nullptr;

        auto postPl = computeTask ?
            nullptr :
            std::make_shared<PostprocessVelocityControl> (name, filename);

        return { simPl, postPl };
    }

    static pair_shared< SimulationStats, PostprocessStats >
    createStatsPlugin(bool computeTask, std::string name, std::string filename, int every)
    {
        auto simPl  = computeTask ? std::make_shared<SimulationStats> (name, every) : nullptr;
        auto postPl = computeTask ? nullptr : std::make_shared<PostprocessStats> (name, filename);

        return { simPl, postPl };
    }

    static pair_shared< TemperaturizePlugin, PostprocessPlugin >
    createTemperaturizePlugin(bool computeTask, std::string name, ParticleVector* pv, float kbt, bool keepVelocity)
    {
        auto simPl = computeTask ? std::make_shared<TemperaturizePlugin> (name, pv->name, kbt, keepVelocity) : nullptr;
        return { simPl, nullptr };
    }

    static pair_shared< WallRepulsionPlugin, PostprocessPlugin >
    createWallRepulsionPlugin(bool computeTask, std::string name, ParticleVector* pv, Wall* wall,
                              float C, float h, float maxForce)
    {
        auto simPl = computeTask ? std::make_shared<WallRepulsionPlugin> (name, pv->name, wall->name, C, h, maxForce) : nullptr;
        return { simPl, nullptr };
    }
};
