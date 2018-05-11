//================================================================================================
// Plugins
//================================================================================================

#pragma once

#include <core/xml/pugixml.hpp>
#include <core/utils/make_unique.h>

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

class PluginFactory
{
private:

	using PluginRetType = std::pair< std::unique_ptr<SimulationPlugin>, std::unique_ptr<PostprocessPlugin> >;

	static PluginRetType createImposeVelocityPlugin(pugi::xml_node node, bool computeTask)
	{
		auto name   = node.attribute("name").as_string();
		auto pvName = node.attribute("pv_name").as_string();

		auto every  = node.attribute("every").as_int(5);
		auto low    = node.attribute("low").as_float3();
		auto high   = node.attribute("high").as_float3();
		auto target = node.attribute("target_velocity").as_float3();

		auto simPl = computeTask ? std::make_unique<ImposeVelocityPlugin>(name, pvName, low, high, target, every) : nullptr;

		return { std::move(simPl), nullptr };
	}

	static PluginRetType createTemperaturizePlugin(pugi::xml_node node, bool computeTask)
	{
		auto name    = node.attribute("name").as_string();

		auto pvName  = node.attribute("pv_name").as_string();
		auto kbT     = node.attribute("kbt").as_float();
		auto keepVel = node.attribute("keep_velocity").as_bool(false);

		auto simPl = computeTask ? std::make_unique<TemperaturizePlugin>(name, pvName, kbT, keepVel) : nullptr;

		return { std::move(simPl), nullptr };
	}

	static PluginRetType createAddForcePlugin(pugi::xml_node node, bool computeTask)
	{
		auto name    = node.attribute("name").as_string();

		auto pvName  = node.attribute("pv_name").as_string();
		auto force   = node.attribute("force").as_float3();

		auto simPl = computeTask ? std::make_unique<AddForcePlugin>(name, pvName, force) : nullptr;

		return { std::move(simPl), nullptr };
	}

	static PluginRetType createAddTorquePlugin(pugi::xml_node node, bool computeTask)
	{
		auto name    = node.attribute("name").as_string();

		auto pvName  = node.attribute("pv_name").as_string();
		auto torque  = node.attribute("torque").as_float3();

		auto simPl = computeTask ? std::make_unique<AddTorquePlugin>(name, pvName, torque) : nullptr;

		return { std::move(simPl), nullptr };
	}

	static PluginRetType createImposeProfilePlugin(pugi::xml_node node, bool computeTask)
	{
		auto name    = node.attribute("name").as_string();

		auto pvName  = node.attribute("pv_name").as_string();
		auto vel   = node.attribute("velocity").as_float3();
		auto low   = node.attribute("low").as_float3();
		auto high  = node.attribute("high").as_float3();
		auto kbT   = node.attribute("kbt").as_float();

		auto simPl = computeTask ? std::make_unique<ImposeProfilePlugin>(name, pvName, low, high, vel, kbT) : nullptr;

		return { std::move(simPl), nullptr };
	}

	static PluginRetType createWallRepulsionPlugin(pugi::xml_node node, bool computeTask)
	{
		auto name    = node.attribute("name").as_string();

		auto pvName   = node.attribute("pv_name").as_string();
		auto wallName = node.attribute("wall_name").as_string();
		auto C         = node.attribute("C").as_float();
		auto h         = node.attribute("h").as_float(0.2f);
		auto maxForce  = node.attribute("maxForce").as_float(1e3f);

		auto simPl = computeTask ? std::make_unique<WallRepulsionPlugin>(name, pvName, wallName, C, h, maxForce) : nullptr;

		return { std::move(simPl), nullptr };
	}



	static PluginRetType createStatsPlugin(pugi::xml_node node, bool computeTask)
	{
		auto name   = node.attribute("name").as_string();

		auto every  = node.attribute("every").as_int(1000);

		auto simPl  = computeTask ? std::make_unique<SimulationStats>(name, every) : nullptr;
		auto postPl = computeTask ? nullptr :std::make_unique<PostprocessStats>(name);

		return { std::move(simPl), std::move(postPl) };
	}

	static PluginRetType createDumpAveragePlugin(pugi::xml_node node, bool computeTask)
	{
		auto name        = node.attribute("name").as_string();

		auto pvName      = node.attribute("pv_name").as_string();
		auto sampleEvery = node.attribute("sample_every").as_int(50);
		auto dumpEvery   = node.attribute("dump_every").as_int(5000);
		auto binSize     = node.attribute("bin_size").as_float3( {1, 1, 1} );

		std::vector<std::string> names;
		std::vector<Average3D::ChannelType> types;
		for (auto n : node.children("channel"))
		{
			names.push_back(n.attribute("name").as_string());
			std::string typeStr = n.attribute("type").as_string();

			if      (typeStr == "scalar")             types.push_back(Average3D::ChannelType::Scalar);
			else if (typeStr == "vector")             types.push_back(Average3D::ChannelType::Vector_float3);
			else if (typeStr == "vector_from_float4") types.push_back(Average3D::ChannelType::Vector_float4);
			else if (typeStr == "vector_from_float8") types.push_back(Average3D::ChannelType::Vector_2xfloat4);
			else if (typeStr == "tensor6")            types.push_back(Average3D::ChannelType::Tensor6);
			else die("Unable to parse input at %s, unknown type: '%s'", n.path().c_str(), typeStr.c_str());
		}

		auto path = node.attribute("path").as_string("xdmf");

		auto simPl  = computeTask ?
				std::make_unique<Average3D>(name, pvName, names, types, sampleEvery, dumpEvery, binSize) :
				nullptr;

		auto postPl = computeTask ? nullptr : std::make_unique<UniformCartesianDumper>(name, path);

		return { std::move(simPl), std::move(postPl) };
	}

	static PluginRetType createDumpAverageRelativePlugin(pugi::xml_node node, bool computeTask)
	{
		auto name        = node.attribute("name").as_string();

		auto pvName      = node.attribute("pv_name").as_string();
		auto sampleEvery = node.attribute("sample_every").as_int(50);
		auto dumpEvery   = node.attribute("dump_every").as_int(5000);
		auto binSize     = node.attribute("bin_size").as_float3( {1, 1, 1} );

		std::vector<std::string> names;
		std::vector<Average3D::ChannelType> types;
		for (auto n : node.children("channel"))
		{
			names.push_back(n.attribute("name").as_string());
			std::string typeStr = n.attribute("type").as_string();

			if      (typeStr == "scalar")             types.push_back(Average3D::ChannelType::Scalar);
			else if (typeStr == "vector")             types.push_back(Average3D::ChannelType::Vector_float3);
			else if (typeStr == "vector_from_float4") types.push_back(Average3D::ChannelType::Vector_float4);
			else if (typeStr == "vector_from_float8") types.push_back(Average3D::ChannelType::Vector_2xfloat4);
			else if (typeStr == "tensor6")            types.push_back(Average3D::ChannelType::Tensor6);
			else die("Unable to parse input at %s, unknown type: '%s'", n.path().c_str(), typeStr.c_str());
		}

		auto path   = node.attribute("path").as_string("xdmf");

		auto ovName = node.attribute("relative_to_ov").as_string();
		auto id     = node.attribute("relative_to_id").as_int();

		auto simPl  = computeTask ?
				std::make_unique<AverageRelative3D>(name, pvName, names, types, sampleEvery, dumpEvery, binSize, ovName, id) :
				nullptr;

		auto postPl = computeTask ? nullptr : std::make_unique<UniformCartesianDumper>(name, path);

		return { std::move(simPl), std::move(postPl) };
	}

	static PluginRetType createDumpXYZPlugin(pugi::xml_node node, bool computeTask)
	{
		auto name      = node.attribute("name").as_string();

		auto pvName    = node.attribute("pv_name").as_string();
		auto dumpEvery = node.attribute("dump_every").as_int(1000);

		auto path      = node.attribute("path").as_string("xyz/");

		auto simPl  = computeTask ? std::make_unique<XYZPlugin>(name, pvName, dumpEvery) : nullptr;
		auto postPl = computeTask ? nullptr : std::make_unique<XYZDumper>(name, path);

		return { std::move(simPl), std::move(postPl) };
	}

	static PluginRetType createDumpMeshPlugin(pugi::xml_node node, bool computeTask)
	{
		auto name      = node.attribute("name").as_string();

		auto ovName    = node.attribute("ov_name").as_string();
		auto dumpEvery = node.attribute("dump_every").as_int(1000);

		auto path      = node.attribute("path").as_string("ply/");

		auto simPl  = computeTask ? std::make_unique<MeshPlugin>(name, ovName, dumpEvery) : nullptr;
		auto postPl = computeTask ? nullptr : std::make_unique<MeshDumper>(name, path);

		return { std::move(simPl), std::move(postPl) };
	}

	static PluginRetType createDumpObjPosition(pugi::xml_node node, bool computeTask)
	{
		auto name      = node.attribute("name").as_string();

		auto ovName    = node.attribute("ov_name").as_string();
		auto dumpEvery = node.attribute("dump_every").as_int(1000);

		auto path      = node.attribute("path").as_string("pos/");

		auto simPl  = computeTask ? std::make_unique<ObjPositionsPlugin>(name, ovName, dumpEvery) : nullptr;
		auto postPl = computeTask ? nullptr : std::make_unique<ObjPositionsDumper>(name, path);

		return { std::move(simPl), std::move(postPl) };
	}

	static PluginRetType createPinObjPlugin(pugi::xml_node node, bool computeTask)
	{
		auto name      = node.attribute("name").as_string();

		auto ovName    = node.attribute("ov_name").as_string();
		auto dumpEvery = node.attribute("dump_every").as_int(1000);
		auto translate = node.attribute("pin_translation").as_int3({0,0,0});
		auto rotate    = node.attribute("pin_rotation").as_int3({0,0,0});

		auto path      = node.attribute("path").as_string("pos/");


		auto simPl  = computeTask ? std::make_unique<PinObjectPlugin>(name, ovName, translate, rotate, dumpEvery) : nullptr;
		auto postPl = computeTask ? nullptr : std::make_unique<ReportPinObjectPlugin>(name, path);

		return { std::move(simPl), std::move(postPl) };
	}

public:
	static PluginRetType create(pugi::xml_node node, bool computeTask)
	{
		std::string type = node.attribute("type").as_string();

		std::map<std::string, std::function< PluginRetType(pugi::xml_node, bool) >> plugins = {
				{"temperaturize",          createTemperaturizePlugin         },
				{"impose_profile",         createImposeProfilePlugin         },
				{"add_torque",             createAddTorquePlugin             },
				{"add_force",              createAddForcePlugin              },
				{"wall_repulsion",         createWallRepulsionPlugin         },
				{"stats",                  createStatsPlugin                 },
				{"dump_avg_flow",          createDumpAveragePlugin           },
				{"dump_avg_relative_flow", createDumpAverageRelativePlugin   },
				{"dump_xyz",               createDumpXYZPlugin               },
				{"dump_mesh",              createDumpMeshPlugin              },
				{"dump_obj_pos",           createDumpObjPosition             },
				{"impose_velocity",        createImposeVelocityPlugin        },
				{"pin_object",             createPinObjPlugin                }
		};

		if (plugins.find(type) != plugins.end())
			return plugins[type](node, computeTask);
		else
			die("Unable to parse input at %s, unknown 'type': '%s'", node.path().c_str(), type.c_str());

		// shut up warning
		return {nullptr, nullptr};
	}
};
