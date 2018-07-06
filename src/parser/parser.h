#pragma once

#include <string>

#include <core/logger.h>
#include <core/xml/pugixml.hpp>

class uDeviceX;

class Parser
{
private:
	pugi::xml_document config;
	int forceDebugLvl;

public:
	Parser(std::string xmlname, int forceDebugLvl, std::string variables);
	Parser(const pugi::xml_document& config, int forceDebugLvl, std::string variables);

	std::unique_ptr<uDeviceX> setup_uDeviceX(Logger& logger, bool useGpuAwareMPI);
	int getNIterations();
};
