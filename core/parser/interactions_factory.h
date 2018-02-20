//================================================================================================
// Interactions
//================================================================================================

#pragma once

#include <core/xml/pugixml.hpp>

#include <core/interactions/pairwise.h>
#include <core/interactions/pairwise_with_stress.h>
#include <core/interactions/sampler.h>
#include <core/interactions/rbc.h>

#include <core/interactions/pairwise_interactions/dpd.h>
#include <core/interactions/pairwise_interactions/lj.h>
#include <core/interactions/pairwise_interactions/lj_object_aware.h>

class InteractionFactory
{
private:
	static Interaction* createDPD(pugi::xml_node node)
	{
		auto name         = node.attribute("name").as_string("");
		auto rc           = node.attribute("rc").as_float(1.0f);
		auto stressPeriod = node.attribute("stress_period").as_float(-1.0f);

		auto a     = node.attribute("a")    .as_float(50);
		auto gamma = node.attribute("gamma").as_float(20);
		auto kbT   = node.attribute("kbt")  .as_float(1.0);
		auto dt    = node.attribute("dt")   .as_float(0.01);
		auto power = node.attribute("power").as_float(1.0f);

		Pairwise_DPD dpd(rc, a, gamma, kbT, dt, power);

		if (stressPeriod > 0.0f)
			return (Interaction*) new InteractionPair_withStress<Pairwise_DPD> (name, rc, stressPeriod, dpd);
		else
			return (Interaction*) new InteractionPair<Pairwise_DPD>            (name, rc, dpd);
	}

	static Interaction* createLJ(pugi::xml_node node)
	{
		auto name = node.attribute("name").as_string("");
		auto rc   = node.attribute("rc").as_float(1.0f);

		auto epsilon = node.attribute("epsilon").as_float(10.0f);
		auto sigma   = node.attribute("sigma")  .as_float(0.5f);

		Pairwise_LJ lj(rc, sigma, epsilon);

		return (Interaction*) new InteractionPair<Pairwise_LJ>(name, rc, lj);
	}

	static Interaction* createLJ_objectAware(pugi::xml_node node)
	{
		auto name = node.attribute("name").as_string("");
		auto rc   = node.attribute("rc").as_float(1.0f);

		auto epsilon = node.attribute("epsilon").as_float(10.0f);
		auto sigma   = node.attribute("sigma")  .as_float(0.5f);

		Pairwise_LJObjectAware ljo(rc, sigma, epsilon);

		return (Interaction*) new InteractionPair<Pairwise_LJObjectAware>(name, rc, ljo);
	}

	static Interaction* createRBC(pugi::xml_node node)
	{
		auto name = node.attribute("name").as_string("");

		RBCParameters p;
		std::string preset = node.attribute("preset").as_string();

		auto setIfNotEmpty_float = [&node] (float& val, const char* name)
		{
			if (!node.attribute(name).empty())
				val = node.attribute(name).as_float();
		};

		if (preset == "lina")
			p = Lina_parameters;
		else
			error("Unknown predefined parameter set for '%s' interaction: '%s'",
					name, preset.c_str());

		setIfNotEmpty_float(p.x0,         "x0");
		setIfNotEmpty_float(p.p,          "p");
		setIfNotEmpty_float(p.ka,         "ka");
		setIfNotEmpty_float(p.kb,         "kb");
		setIfNotEmpty_float(p.kd,         "kd");
		setIfNotEmpty_float(p.kv,         "kv");
		setIfNotEmpty_float(p.gammaC,     "gammaC");
		setIfNotEmpty_float(p.gammaT,     "gammaT");
		setIfNotEmpty_float(p.kbT,        "kbT");
		setIfNotEmpty_float(p.mpow,       "mpow");
		setIfNotEmpty_float(p.theta,      "theta");
		setIfNotEmpty_float(p.totArea0,   "area");
		setIfNotEmpty_float(p.totVolume0, "volume");

		return (Interaction*) new InteractionRBCMembrane(name, p);
	}

public:
	static Interaction* create(pugi::xml_node node)
	{
		std::string type = node.attribute("type").as_string();

		if (type == "dpd")
			return createDPD(node);
		if (type == "lj")
			return createLJ(node);
		if (type == "lj_object")
			return createLJ_objectAware(node);
		if (type == "rbc")
			return createRBC(node);

		die("Unable to parse input at %s, unknown 'type': '%s'", node.path().c_str(), type.c_str());

		return nullptr;
	}
};
