#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <regex>

#include "logger.h"

#pragma once

class IniParser
{
public:
	std::unordered_map<std::string, std::string> content;
	mutable std::unordered_map<std::string, std::unordered_map<std::string, void*>> cache;

	const std::regex sectionRX;
	const std::regex boolRX;
	const std::regex commentRX;

	template <typename T, typename Parser>
	T get(const std::string& sectionName, const std::string& key, const T def, Parser parse)
	{
		// Check if the section exists
		if (content.find(sectionName) == content.end()) return def;

		// Check the cache
		{
			auto& cachedSection = cache[sectionName];
			auto value = cachedSection.find(key);
			if (value != cachedSection.end()) return *(T*)value->second;
		}

		// Not in cache, read and update cache
		const std::regex key_valRX("(?:^|\\n)\\s*"+key+"\\s*(?:\\:|\\=)\\s*(.*?)(?:$|\\n)", std::regex::icase);
		std::string& sectionContent = content[sectionName];
		std::string  value;

		std::smatch result;
		if (std::regex_search(sectionContent, result, key_valRX))
		{
			value = result[1];

			T* result = new T;
			*result = parse(value);

			cache[sectionName][key] = (void*)result;
			return *result;
		}
		else return def;
	}


public:
	IniParser(const std::string& fname) :
		sectionRX("\\[(.*?)\\]([^\\[]*)", std::regex::optimize | std::regex::icase),
		boolRX("\\s*(?:1|true)\\s*", std::regex::optimize | std::regex::icase),
		commentRX("#.*?(?:$|\\n)", std::regex::optimize | std::regex::icase)
	{
		std::ifstream inpFile(fname);
		if (!inpFile.good())
		{
			error("Unable to open settings file, falling back to defaults");
			return;
		}

		std::stringstream buffer;
		buffer << inpFile.rdbuf();
		std::string strbuf = buffer.str();

		std::smatch result;
		while (std::regex_search(strbuf, result, sectionRX))
		{
			std::string sectName = result[1];
			std::string sectContent = result[2];
			content[sectName] = std::regex_replace(sectContent, commentRX, "\n");
			strbuf = result.suffix().str();

			//std::cout << "'" << sectName << "'  ->  " << content[sectName] << std::endl;
		}
	}

	template<typename T>
	void setDefaultValue(const std::string& section, const std::string& key, const T value)
	{
		T* data = new T;
		*data = value;
		cache[section][key] = (void*)data;
	}

	int getInt(const std::string& section, const std::string& key, const int def = 0)
	{
		return get(section, key, def, [](const std::string& value) { return std::stoi(value); } );
	}

	int3 getInt3(const std::string& section, const std::string& key, const int3 def = make_int3(0, 0, 0))
	{
		return get(section, key, def, [](const std::string& value) {
			std::istringstream stream(value);
			int3 res;
			stream >> res.x >> res.y >> res.z;
			return res;
		} );
	}

	float getFloat(const std::string& section, const std::string& key, const float def = 0)
	{
		return get(section, key, def, [](const std::string& value) { return std::stof(value); } );
	}

	float3 getFloat3(const std::string& section, const std::string& key, const float3 def = make_float3(0, 0, 0))
	{
		return get(section, key, def, [](const std::string& value) {
			std::istringstream stream(value);
			float3 res;
			stream >> res.x >> res.y >> res.z;
			return res;
		} );
	}


	double getDouble(const std::string& section, const std::string& key, const double def = 0)
	{
		return get(section, key, def, [](const std::string& value) { return std::stod(value); } );
	}

	bool getBool(const std::string& section, const std::string& key, const bool def = false)
	{
		return get(section, key, def, [this](const std::string& value) {
			return std::regex_match(value, boolRX);
		});
	}

	std::string getString(const std::string& section, const std::string& key, const std::string def = "")
	{
		return get(section, key, def, [] (const std::string& value) { return value; } );
	}
};
