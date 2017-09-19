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

	uDeviceX* setup_uDeviceX(int argc, char** argv, Logger& logger);
	int getNIterations();
};
