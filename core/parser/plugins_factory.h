//================================================================================================
// Plugins
//================================================================================================

#pragma once

#include <core/xml/pugixml.hpp>

#include <plugins/dumpavg.h>
#include <plugins/dumpxyz.h>
#include <plugins/stats.h>
#include <plugins/temperaturize.h>
#include <plugins/dump_obj_position.h>
#include <plugins/impose_velocity.h>
#include <plugins/pin_object.h>
#include <plugins/add_force.h>

class PluginFactory
{
private:
	static std::pair<SimulationPlugin*, PostprocessPlugin*> createImposeVelocityPlugin(pugi::xml_node node, bool computeTask)
	{
		auto name   = node.attribute("name").as_string();
		auto pvName = node.attribute("pv_name").as_string();

		auto every  = node.attribute("every").as_int(5);
		auto low    = node.attribute("low").as_float3();
		auto high   = node.attribute("high").as_float3();
		auto target = node.attribute("target_velocity").as_float3();

		auto simPl = computeTask ? new ImposeVelocityPlugin(name, pvName, low, high, target, every) : nullptr;

		return { (SimulationPlugin*) simPl, nullptr };
	}

	static std::pair<SimulationPlugin*, PostprocessPlugin*> createTemperaturizePlugin(pugi::xml_node node, bool computeTask)
	{
		auto name    = node.attribute("name").as_string();

		auto pvName  = node.attribute("pv_name").as_string();
		auto kbT     = node.attribute("kbt").as_float();
		auto keepVel = node.attribute("keep_velocity").as_bool(false);

		auto simPl = computeTask ? new TemperaturizePlugin(name, pvName, kbT, keepVel) : nullptr;

		return { (SimulationPlugin*) simPl, nullptr };
	}

	static std::pair<SimulationPlugin*, PostprocessPlugin*> createAddForcePlugin(pugi::xml_node node, bool computeTask)
	{
		auto name    = node.attribute("name").as_string();

		auto pvName  = node.attribute("pv_name").as_string();
		auto force   = node.attribute("force").as_float3();

		auto simPl = computeTask ? new AddForcePlugin(name, pvName, force) : nullptr;

		return { (SimulationPlugin*) simPl, nullptr };
	}


	static std::pair<SimulationPlugin*, PostprocessPlugin*> createStatsPlugin(pugi::xml_node node, bool computeTask)
	{
		auto name   = node.attribute("name").as_string();

		auto every  = node.attribute("every").as_int(1000);

		auto simPl  = computeTask ? new SimulationStats(name, every) : nullptr;
		auto postPl = computeTask ? nullptr :new PostprocessStats(name);

		return { (SimulationPlugin*) simPl, (PostprocessPlugin*) postPl };
	}

	static std::pair<SimulationPlugin*, PostprocessPlugin*> createDumpavgPlugin(pugi::xml_node node, bool computeTask)
	{
		auto name        = node.attribute("name").as_string();

		auto pvNames     = node.attribute("pv_names").as_string();
		auto sampleEvery = node.attribute("sample_every").as_int(50);
		auto dumpEvery   = node.attribute("dump_every").as_int(5000);
		auto binSize     = node.attribute("bin_size").as_float3( {1, 1, 1} );
		auto momentum    = node.attribute("need_momentum").as_bool(true);
		auto force       = node.attribute("need_force").as_bool(false);

		auto path        = node.attribute("path").as_string("xdmf");

		auto simPl  = computeTask ? new Avg3DPlugin(name, pvNames, sampleEvery, dumpEvery, binSize, momentum, force) : nullptr;
		auto postPl = computeTask ? nullptr : new Avg3DDumper(name, path);

		return { (SimulationPlugin*) simPl, (PostprocessPlugin*) postPl };
	}

	static std::pair<SimulationPlugin*, PostprocessPlugin*> createDumpXYZPlugin(pugi::xml_node node, bool computeTask)
	{
		auto name      = node.attribute("name").as_string();

		auto pvName    = node.attribute("pv_name").as_string();
		auto dumpEvery = node.attribute("dump_every").as_int(1000);

		auto path      = node.attribute("path").as_string("xyz/");

		auto simPl  = computeTask ? new XYZPlugin(name, pvName, dumpEvery) : nullptr;
		auto postPl = computeTask ? nullptr : new XYZDumper(name, path);

		return { (SimulationPlugin*) simPl, (PostprocessPlugin*) postPl };
	}

	static std::pair<SimulationPlugin*, PostprocessPlugin*> createDumpObjPosition(pugi::xml_node node, bool computeTask)
	{
		auto name      = node.attribute("name").as_string();

		auto ovName    = node.attribute("ov_name").as_string();
		auto dumpEvery = node.attribute("dump_every").as_int(1000);

		auto path      = node.attribute("path").as_string("pos/");

		auto simPl  = computeTask ? new ObjPositionsPlugin(name, ovName, dumpEvery) : nullptr;
		auto postPl = computeTask ? nullptr : new ObjPositionsDumper(name, path);

		return { (SimulationPlugin*) simPl, (PostprocessPlugin*) postPl };
	}

	static std::pair<SimulationPlugin*, PostprocessPlugin*> createPinObjPlugin(pugi::xml_node node, bool computeTask)
	{
		auto name      = node.attribute("name").as_string();

		auto ovName    = node.attribute("ov_name").as_string();
		auto dumpEvery = node.attribute("dump_every").as_int(1000);
		auto translate = node.attribute("pin_translation").as_int3({0,0,0});
		auto rotate    = node.attribute("pin_rotation").as_int3({0,0,0});

		auto path      = node.attribute("path").as_string("pos/");



		auto simPl  = computeTask ? new PinObjectPlugin(name, ovName, translate, rotate, dumpEvery) : nullptr;
		auto postPl = computeTask ? nullptr : new ReportPinObjectPlugin(name, path);

		return { (SimulationPlugin*) simPl, (PostprocessPlugin*) postPl };
	}

public:
	static std::pair<SimulationPlugin*, PostprocessPlugin*> create(pugi::xml_node node, bool computeTask)
	{
		std::string type = node.attribute("type").as_string();

		if (type == "temperaturize")
			return createTemperaturizePlugin(node, computeTask);
		if (type == "add_force")
			return createAddForcePlugin(node, computeTask);
		if (type == "stats")
			return createStatsPlugin(node, computeTask);
		if (type == "dump_avg_flow")
			return createDumpavgPlugin(node, computeTask);
		if (type == "dump_xyz")
			return createDumpXYZPlugin(node, computeTask);
		if (type == "dump_obj_pos")
			return createDumpObjPosition(node, computeTask);
		if (type == "impose_velocity")
			return createImposeVelocityPlugin(node, computeTask);
		if (type == "pin_object")
			return createPinObjPlugin(node, computeTask);

		die("Unable to parse input at %s, unknown 'type' %s", node.path().c_str(), type.c_str());

		return {nullptr, nullptr};
	}
};
