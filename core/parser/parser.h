#pragma once

#include <string>

#include <core/logger.h>
#include <core/xml/pugixml.hpp>

class uDeviceX;

class Parser
{
	pugi::xml_document config;

public:
	Parser(std::string xmlname);
	Parser(const pugi::xml_document& config);

	std::unique_ptr<uDeviceX> setup_uDeviceX(Logger& logger, bool useGpuAwareMPI);
	int getNIterations();
};
