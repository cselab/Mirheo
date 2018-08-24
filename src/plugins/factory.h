#pragma once

#include "interface.h"

#include <core/utils/pytypes.h>

#include <core/pvs/particle_vector.h>
#include <core/pvs/object_vector.h>
#include <core/walls/interface.h>

#include <plugins/average_flow.h>
#include <plugins/average_relative_flow.h>
#include <plugins/channel_dumper.h>
#include <plugins/dumpxyz.h>
#include <plugins/dump_mesh.h>
#include <plugins/stats.h>
#include <plugins/temperaturize.h>
#include <plugins/dump_obj_position.h>
#include <plugins/exchange_pvs_flux_plane.h>
#include <plugins/impose_velocity.h>
#include <plugins/impose_profile.h>
#include <plugins/pin_object.h>
#include <plugins/add_force.h>
#include <plugins/add_torque.h>
#include <plugins/wall_repulsion.h>
#include <plugins/velocity_control.h>

namespace PluginFactory
{
    static std::pair< ImposeVelocityPlugin*, PostprocessPlugin* >
        createImposeVelocityPlugin(bool computeTask, 
            std::string name, ParticleVector* pv, int every,
            pyfloat3 low, pyfloat3 high, pyfloat3 velocity)
    {
        auto simPl = computeTask ? new ImposeVelocityPlugin(
                                        name, pv->name, make_float3(low), make_float3(high), make_float3(velocity), every) :
                                    nullptr;
                                    
        return { simPl, nullptr };
    }

    static std::pair< TemperaturizePlugin*, PostprocessPlugin* >
        createTemperaturizePlugin(bool computeTask, std::string name, ParticleVector* pv, float kbt, bool keepVelocity)
    {
        auto simPl = computeTask ? new TemperaturizePlugin(name, pv->name, kbt, keepVelocity) : nullptr;
        return { simPl, nullptr };
    }

    static std::pair< AddForcePlugin*, PostprocessPlugin* >
        createAddForcePlugin(bool computeTask, std::string name, ParticleVector* pv, pyfloat3 force)
    {
        auto simPl = computeTask ? new AddForcePlugin(name, pv->name, make_float3(force)) : nullptr;
        return { simPl, nullptr };
    }

    static std::pair< AddTorquePlugin*, PostprocessPlugin* >
        createAddTorquePlugin(bool computeTask, std::string name, ParticleVector* pv, pyfloat3 torque)
    {
        auto simPl = computeTask ? new AddTorquePlugin(name, pv->name, make_float3(torque)) : nullptr;
        return { simPl, nullptr };
    }

    static std::pair< ImposeProfilePlugin*, PostprocessPlugin* >
        createImposeProfilePlugin(bool computeTask,  std::string name, ParticleVector* pv, 
                                   pyfloat3 low, pyfloat3 high, pyfloat3 velocity, float kbt)
    {
        auto simPl = computeTask ? new ImposeProfilePlugin(
            name, pv->name, make_float3(low), make_float3(high), make_float3(velocity), kbt) : nullptr;
            
        return { simPl, nullptr };
    }

    static std::pair< SimulationVelocityControl*, PostprocessVelocityControl* >
    createSimulationVelocityControlPlugin(bool computeTask, std::string name, std::string filename, ParticleVector *pv,
                                          pyfloat3 low, pyfloat3 high, int sampleEvery, int dumpEvery,
                                          pyfloat3 targetVel, float Kp, float Ki, float Kd)
    {
        auto simPl = computeTask ?
            new SimulationVelocityControl(name, pv->name, make_float3(low), make_float3(high), sampleEvery, dumpEvery,
                                          make_float3(targetVel), Kp, Ki, Kd) :
            nullptr;

        auto postPl = computeTask ?
            nullptr :
            new PostprocessVelocityControl(name, filename);

        return { simPl, postPl };
    }

    static std::pair< ExchangePVSFluxPlanePlugin*, PostprocessPlugin* >
    createExchangePVSFluxPlanePlugin(bool computeTask, std::string name, ParticleVector *pv1, ParticleVector *pv2, pyfloat4 plane)
    {
        auto simPl = computeTask ?
            new ExchangePVSFluxPlanePlugin(name, pv1->name, pv2->name, make_float4(plane)) : nullptr;
        
        return { simPl, nullptr };    
    }

    static std::pair< WallRepulsionPlugin*, PostprocessPlugin* >
        createWallRepulsionPlugin(bool computeTask, std::string name, ParticleVector* pv, Wall* wall,
                                  float C, float h, float maxForce)
    {
        auto simPl = computeTask ? new WallRepulsionPlugin(name, pv->name, wall->name, C, h, maxForce) : nullptr;
        return { simPl, nullptr };
    }



    static std::pair< SimulationStats*, PostprocessStats* >
        createStatsPlugin(bool computeTask, std::string name, std::string filename, int every)
    {
        auto simPl  = computeTask ? new SimulationStats(name, every) : nullptr;
        auto postPl = computeTask ? nullptr :new PostprocessStats(name, filename);

        return { simPl, postPl };
    }

    static std::pair< Average3D*, UniformCartesianDumper* >
    createDumpAveragePlugin(bool computeTask, std::string name, std::vector<ParticleVector*> pvs,
                                int sampleEvery, int dumpEvery, pyfloat3 binSize,
                                std::vector< std::pair<std::string, std::string> > channels,
                                std::string path)
    {
        std::vector<std::string> names, pvNames;
        std::vector<Average3D::ChannelType> types;
        for (auto& p : channels)
        {
            names.push_back(p.first);
            std::string typeStr = p.second;

            if      (typeStr == "scalar")             types.push_back(Average3D::ChannelType::Scalar);
            else if (typeStr == "vector")             types.push_back(Average3D::ChannelType::Vector_float3);
            else if (typeStr == "vector_from_float4") types.push_back(Average3D::ChannelType::Vector_float4);
            else if (typeStr == "vector_from_float8") types.push_back(Average3D::ChannelType::Vector_2xfloat4);
            else if (typeStr == "tensor6")            types.push_back(Average3D::ChannelType::Tensor6);
            else die("Unable to get parse channel type '%s'", typeStr.c_str());
        }

        if (computeTask) {
            for (auto &pv : pvs)
                pvNames.push_back(pv->name);
        }
        
        auto simPl  = computeTask ?
                new Average3D(name, pvNames, names, types, sampleEvery, dumpEvery, make_float3(binSize)) :
                nullptr;

        auto postPl = computeTask ? nullptr : new UniformCartesianDumper(name, path);

        return { simPl, postPl };
    }

    static std::pair< AverageRelative3D*, UniformCartesianDumper* >
    createDumpAverageRelativePlugin(bool computeTask, std::string name, std::vector<ParticleVector*> pvs,
                                        ObjectVector* relativeToOV, int relativeToId,
                                        int sampleEvery, int dumpEvery, pyfloat3 binSize,
                                        std::vector< std::pair<std::string, std::string> > channels,
                                        std::string path)
    {
        std::vector<std::string> names, pvNames;
        std::vector<Average3D::ChannelType> types;
        for (auto& p : channels)
        {
            names.push_back(p.first);
            std::string typeStr = p.second;

            if      (typeStr == "scalar")             types.push_back(Average3D::ChannelType::Scalar);
            else if (typeStr == "vector")             types.push_back(Average3D::ChannelType::Vector_float3);
            else if (typeStr == "vector_from_float4") types.push_back(Average3D::ChannelType::Vector_float4);
            else if (typeStr == "vector_from_float8") types.push_back(Average3D::ChannelType::Vector_2xfloat4);
            else if (typeStr == "tensor6")            types.push_back(Average3D::ChannelType::Tensor6);
            else die("Unable to get parse channel type '%s'", typeStr.c_str());
        }

        if (computeTask) {
            for (auto &pv : pvs)
                pvNames.push_back(pv->name);
        }
    
        auto simPl  = computeTask ?
                new AverageRelative3D(name, pvNames,
                                      names, types, sampleEvery, dumpEvery,
                                      make_float3(binSize), relativeToOV->name, relativeToId) :
                nullptr;

        auto postPl = computeTask ? nullptr : new UniformCartesianDumper(name, path);

        return { simPl, postPl };
    }

    static std::pair< XYZPlugin*, XYZDumper* >
        createDumpXYZPlugin(bool computeTask, std::string name, ParticleVector* pv, int dumpEvery, std::string path)
    {
        auto simPl  = computeTask ? new XYZPlugin(name, pv->name, dumpEvery) : nullptr;
        auto postPl = computeTask ? nullptr : new XYZDumper(name, path);

        return { simPl, postPl };
    }

    static std::pair< MeshPlugin*, MeshDumper* >
        createDumpMeshPlugin(bool computeTask, std::string name, ObjectVector* ov, int dumpEvery, std::string path)
    {
        auto simPl  = computeTask ? new MeshPlugin(name, ov->name, dumpEvery) : nullptr;
        auto postPl = computeTask ? nullptr : new MeshDumper(name, path);

        return { simPl, postPl };
    }

    static std::pair< ObjPositionsPlugin*, ObjPositionsDumper* >
        createDumpObjPosition(bool computeTask, std::string name, ObjectVector* ov, int dumpEvery, std::string path)
    {
        auto simPl  = computeTask ? new ObjPositionsPlugin(name, ov->name, dumpEvery) : nullptr;
        auto postPl = computeTask ? nullptr : new ObjPositionsDumper(name, path);

        return { simPl, postPl };
    }

    static std::pair< PinObjectPlugin*, ReportPinObjectPlugin* >
        createPinObjPlugin(bool computeTask, std::string name, ObjectVector* ov,
                           int dumpEvery, std::string path,
                           pyint3 pinTranslation, pyint3 pinRotation)
    {
        auto simPl  = computeTask ? new PinObjectPlugin(name, ov->name,
                                                        make_int3(pinTranslation), make_int3(pinRotation),
                                                        dumpEvery) : 
                                    nullptr;
        auto postPl = computeTask ? nullptr : new ReportPinObjectPlugin(name, path);

        return { simPl, postPl };
    }
};
