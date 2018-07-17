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
#include <plugins/impose_velocity.h>
#include <plugins/impose_profile.h>
#include <plugins/pin_object.h>
#include <plugins/add_force.h>
#include <plugins/add_torque.h>
#include <plugins/wall_repulsion.h>

namespace PluginFactory
{
    static std::pair< ImposeVelocityPlugin*, PostprocessPlugin* >
        createImposeVelocityPlugin(
            std::string name, ParticleVector* pv, int every,
            pyfloat3 low, pyfloat3 high, pyfloat3 velocity,
            bool computeTask)
    {
        auto simPl = computeTask ? new ImposeVelocityPlugin(
                                        name, pv->name, make_float3(low), make_float3(high), make_float3(velocity), every) :
                                    nullptr;
        return { simPl, nullptr };
    }

    static std::pair< TemperaturizePlugin*, PostprocessPlugin* >
        createTemperaturizePlugin(std::string name, ParticleVector* pv, float kbt, bool keepVelocity, bool computeTask)
    {
        auto simPl = computeTask ? new TemperaturizePlugin(name, pv->name, kbt, keepVelocity) : nullptr;
        return { simPl, nullptr };
    }

    static std::pair< AddForcePlugin*, PostprocessPlugin* >
        createAddForcePlugin(std::string name, ParticleVector* pv, pyfloat3 force, bool computeTask)
    {
        auto simPl = computeTask ? new AddForcePlugin(name, pv->name, make_float3(force)) : nullptr;
        return { simPl, nullptr };
    }

    static std::pair< AddTorquePlugin*, PostprocessPlugin* >
        createAddTorquePlugin(std::string name, ParticleVector* pv, pyfloat3 torque, bool computeTask)
    {
        auto simPl = computeTask ? new AddTorquePlugin(name, pv->name, make_float3(torque)) : nullptr;
        return { simPl, nullptr };
    }

    static std::pair< ImposeProfilePlugin*, PostprocessPlugin* >
        createImposeProfilePlugin( std::string name, ParticleVector* pv, 
                                   pyfloat3 low, pyfloat3 high, pyfloat3 velocity, float kbt,
                                   bool computeTask)
    {
        auto simPl = computeTask ? new ImposeProfilePlugin(
            name, pv->name, make_float3(low), make_float3(high), make_float3(velocity), kbt) : nullptr;
            
        return { simPl, nullptr };
    }

    static std::pair< WallRepulsionPlugin*, PostprocessPlugin* >
        createWallRepulsionPlugin(std::string name, ParticleVector* pv, Wall* wall,
                                  float C, float h, float maxForce, bool computeTask)
    {
        auto simPl = computeTask ? new WallRepulsionPlugin(name, pv->name, wall->name, C, h, maxForce) : nullptr;
        return { simPl, nullptr };
    }



    static std::pair< SimulationStats*, PostprocessStats* >
        createStatsPlugin(std::string name, int every, bool computeTask)
    {
        auto simPl  = computeTask ? new SimulationStats(name, every) : nullptr;
        auto postPl = computeTask ? nullptr :new PostprocessStats(name);

        return { simPl, postPl };
    }

    static std::pair< Average3D*, UniformCartesianDumper* >
        createDumpAveragePlugin(std::string name, ParticleVector* pv,
                                int sampleEvery, int dumpEvery, pyfloat3 binSize,
                                std::vector< std::pair<std::string, std::string> > channels,
                                std::string path,
                                bool computeTask)
    {
        std::vector<std::string> names;
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

        auto simPl  = computeTask ?
                new Average3D(name, pv->name, names, types, sampleEvery, dumpEvery, make_float3(binSize)) :
                nullptr;

        auto postPl = computeTask ? nullptr : new UniformCartesianDumper(name, path);

        return { simPl, postPl };
    }

    static std::pair< AverageRelative3D*, UniformCartesianDumper* >
        createDumpAverageRelativePlugin(std::string name, ParticleVector* pv,
                                        ObjectVector* relativeToOV, int relativeToId,
                                        int sampleEvery, int dumpEvery, pyfloat3 binSize,
                                        std::vector< std::pair<std::string, std::string> > channels,
                                        std::string path,
                                        bool computeTask)
    {
        std::vector<std::string> names;
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

        auto simPl  = computeTask ?
                new AverageRelative3D(name, pv->name,
                                      names, types, sampleEvery, dumpEvery,
                                      make_float3(binSize), relativeToOV->name, relativeToId) :
                nullptr;

        auto postPl = computeTask ? nullptr : new UniformCartesianDumper(name, path);

        return { simPl, postPl };
    }

    static std::pair< XYZPlugin*, XYZDumper* >
        createDumpXYZPlugin(std::string name, ParticleVector* pv, int dumpEvery, std::string path, bool computeTask)
    {
        auto simPl  = computeTask ? new XYZPlugin(name, pv->name, dumpEvery) : nullptr;
        auto postPl = computeTask ? nullptr : new XYZDumper(name, path);

        return { simPl, postPl };
    }

    static std::pair< MeshPlugin*, MeshDumper* >
        createDumpMeshPlugin(std::string name, ObjectVector* ov, int dumpEvery, std::string path, bool computeTask)
    {
        auto simPl  = computeTask ? new MeshPlugin(name, ov->name, dumpEvery) : nullptr;
        auto postPl = computeTask ? nullptr : new MeshDumper(name, path);

        return { simPl, postPl };
    }

    static std::pair< ObjPositionsPlugin*, ObjPositionsDumper* >
        createDumpObjPosition(std::string name, ObjectVector* ov, int dumpEvery, std::string path, bool computeTask)
    {
        auto simPl  = computeTask ? new ObjPositionsPlugin(name, ov->name, dumpEvery) : nullptr;
        auto postPl = computeTask ? nullptr : new ObjPositionsDumper(name, path);

        return { simPl, postPl };
    }

    static std::pair< PinObjectPlugin*, ReportPinObjectPlugin* >
        createPinObjPlugin(std::string name, ObjectVector* ov,
                           int dumpEvery, std::string path,
                           pyint3 pinTranslation, pyint3 pinRotation, 
                           bool computeTask)
    {
        auto simPl  = computeTask ? new PinObjectPlugin(name, ov->name,
                                                        make_int3(pinTranslation), make_int3(pinRotation),
                                                        dumpEvery) : 
                                    nullptr;
        auto postPl = computeTask ? nullptr : new ReportPinObjectPlugin(name, path);

        return { simPl, postPl };
    }
};
