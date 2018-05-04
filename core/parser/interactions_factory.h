//================================================================================================
// Interactions
//================================================================================================

#pragma once

#include <core/xml/pugixml.hpp>

#include <core/interactions/pairwise.h>
#include <core/interactions/pairwise_with_stress.h>
#include <core/interactions/sampler.h>
#include <core/interactions/membrane.h>

#include <core/interactions/pairwise_interactions/dpd.h>
#include <core/interactions/pairwise_interactions/lj.h>
#include <core/interactions/pairwise_interactions/lj_object_aware.h>

class InteractionFactory
{
private:

	static void setIfNotEmpty_float(pugi::xml_node node, float& val, const char* name)
	{
		if (!node.attribute(name).empty())
			val = node.attribute(name).as_float();
	};

	template<typename T>
	static std::unique_ptr<Interaction> _parseDPDparameters(std::unique_ptr<T> intPtr, pugi::xml_node node)
	{
		auto rc    = node.attribute("rc").as_float(1.0f);

		auto a     = node.attribute("a")    .as_float(50);
		auto gamma = node.attribute("gamma").as_float(20);
		auto kbT   = node.attribute("kbt")  .as_float(1.0);
		auto dt    = node.attribute("dt")   .as_float(0.01);
		auto power = node.attribute("power").as_float(1.0f);

		// Override default parameters for some pairs
		for (auto apply_to : node.children("apply_to"))
		{
			setIfNotEmpty_float(apply_to, a,     "a");
			setIfNotEmpty_float(apply_to, gamma, "gamma" );
			setIfNotEmpty_float(apply_to, kbT,   "kbT");
			setIfNotEmpty_float(apply_to, dt,    "dt");
			setIfNotEmpty_float(apply_to, power, "power");

			Pairwise_DPD dpd(rc, a, gamma, kbT, dt, power);
			intPtr->createPairwise(apply_to.attribute("pv1").as_string(),
								   apply_to.attribute("pv2").as_string(), dpd);

			info("The following interaction was set up: pairwise dpd between '%s' and '%s' with parameters "
					"rc = %g, a = %g, gamma = %g, kbT = %g, dt = %g, power = %g",
					apply_to.attribute("pv1").as_string(),
					apply_to.attribute("pv2").as_string(),
					rc, a, gamma, kbT, dt, power);
		}

		return std::move(intPtr);
	}

	static std::unique_ptr<Interaction> createDPD(pugi::xml_node node)
	{
		auto name  = node.attribute("name").as_string("");
		auto rc    = node.attribute("rc").as_float(1.0f);
		auto stressPeriod = node.attribute("stress_period").as_float(-1.0f);

		if (stressPeriod > 0.0f)
			return _parseDPDparameters(std::make_unique<InteractionPair_withStress<Pairwise_DPD>>(name, rc, stressPeriod), node);
		else
			return _parseDPDparameters(std::make_unique<InteractionPair<Pairwise_DPD>>(name, rc), node);
	}


	template<typename T>
	static std::unique_ptr<Interaction> createLJ(pugi::xml_node node)
	{
		auto name = node.attribute("name").as_string("");
		auto rc   = node.attribute("rc").as_float(1.0f);

		auto epsilon = node.attribute("epsilon").as_float(10.0f);
		auto sigma   = node.attribute("sigma")  .as_float(0.5f);

		auto res = std::make_unique<InteractionPair<T>>(name, rc);

		for (auto apply_to : node.children("apply_to"))
		{
			setIfNotEmpty_float(apply_to, epsilon, "epsilon");
			setIfNotEmpty_float(apply_to, sigma,   "sigma" );

			T lj(rc, sigma, epsilon);
			res->createPairwise(apply_to.attribute("pv1").as_string(),
								apply_to.attribute("pv2").as_string(), lj);

			info("The following interaction set up: pairwise Lennard-Jones between '%s' and '%s' with parameters "
					"epsilon = %g, sigma = %g",
					apply_to.attribute("pv1").as_string(),
					apply_to.attribute("pv2").as_string(),
					epsilon, sigma);
		}

		return std::move(res);
	}

	static std::unique_ptr<Interaction> createMembrane(pugi::xml_node node)
	{
		auto name = node.attribute("name").as_string("");

		MembraneParameters p{};
		std::string preset = node.attribute("preset").as_string();
		float growUntil = node.attribute("grow_until").as_float(-1.0f);
		bool stressFree = node.attribute("stress_free").as_bool(false);

		if (preset == "lina")
			p = Lina_parameters;
		else
			error("Unknown predefined parameter set for '%s' interaction: '%s'",
					name, preset.c_str());

		setIfNotEmpty_float(node, p.x0,         "x0");
		setIfNotEmpty_float(node, p.p,          "p");
		setIfNotEmpty_float(node, p.ka,         "ka");
		setIfNotEmpty_float(node, p.kb,         "kb");
		setIfNotEmpty_float(node, p.kd,         "kd");
		setIfNotEmpty_float(node, p.kv,         "kv");
		setIfNotEmpty_float(node, p.gammaC,     "gammaC");
		setIfNotEmpty_float(node, p.gammaT,     "gammaT");
		setIfNotEmpty_float(node, p.kbT,        "kbT");
		setIfNotEmpty_float(node, p.mpow,       "mpow");
		setIfNotEmpty_float(node, p.theta,      "theta");
		setIfNotEmpty_float(node, p.totArea0,   "area");
		setIfNotEmpty_float(node, p.totVolume0, "volume");

		if (growUntil > 0.0f)
			return std::make_unique<InteractionMembrane>( name, p, stressFree, [growUntil] (float t) { return min(1.0f, 0.5f + 0.5f * (t / growUntil)); } );
		else
			return std::make_unique<InteractionMembrane>( name, p, stressFree );
	}

public:
	static std::unique_ptr<Interaction> create(pugi::xml_node node)
	{
		std::string type = node.attribute("type").as_string();

		if (type == "dpd")
			return createDPD(node);
		if (type == "lj")
			return createLJ<Pairwise_LJ>(node);
		if (type == "lj_object")
			return createLJ<Pairwise_LJObjectAware>(node);
		if (type == "membrane")
			return createMembrane(node);

		die("Unable to parse input at %s, unknown 'type': '%s'", node.path().c_str(), type.c_str());

		return nullptr;
	}
};
