#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <regex>

class IniParser
{
	std::unordered_map<std::string, std::string> content;

	const std::regex sectionRX;
	const std::regex boolRX;
	const std::regex commentRX;

	bool __readValue(const std::string& section, const std::string& key, std::string& value)
	{
		if (content.find(section) == content.end()) return false;

		const std::regex key_valRX("(?:^|\\n)\\s*"+key+"\\s*(?:\\:|\\=)\\s*(.*?)(?:$|\\n)");

		std::string& sectContent = content[section];
		std::smatch result;
		if (std::regex_search(sectContent, result, key_valRX))
		{
			value = result[1];
			return true;
		}
		else return false;
	}

public:
	IniParser(const std::string& fname) :
		sectionRX("\\[(.*?)\\]([^\\[]*)", std::regex::optimize),
		boolRX("\\s*(?:1|true)\\s*"),
		commentRX("#.*?(?:$|\\n)")
	{
		std::ifstream inpFile(fname);
		if (!inpFile.good()) throw std::runtime_error("No such file: "+fname);

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

			std::cout << sectName << "  ->  " << content[sectName] << std::endl;
		}
	}

	int getInt(const std::string& section, const std::string& key, const int def = 0)
	{
		std::string value;
		if (__readValue(section, key, value))
		{
			std::size_t pos;
			int res = std::stoi(value, &pos);
			if (pos < value.length())
				throw std::runtime_error("Bad argument for key "+key+" in section ["+section+"]: "+value);
			return res;
		}
		return def;
	}

	float getFloat(const std::string& section, const std::string& key, const float def = 0)
	{
		std::string value;
		if (__readValue(section, key, value))
		{
			std::size_t pos;
			float res = std::stof(value, &pos);
			if (pos < value.length())
				throw std::runtime_error("Bad argument for key "+key+" in section ["+section+"]: "+value);
			return res;
		}
		return def;
	}

	double getDouble(const std::string& section, const std::string& key, const double def = 0)
	{
		std::string value;
		if (__readValue(section, key, value))
		{
			std::size_t pos;
			double res = std::stod(value, &pos);
			if (pos < value.length())
				throw std::runtime_error("Bad argument for key "+key+" in section ["+section+"]: "+value);
			return res;
		}
		return def;
	}

	bool getBool(const std::string& section, const std::string& key, const bool def = false)
	{
		std::string value;
		if (__readValue(section, key, value))
			return std::regex_match(value, boolRX);
		else
			return def;
	}

	std::string getString(const std::string& section, const std::string& key, const std::string def = "")
	{
		std::string value;
		if (__readValue(section, key, value))
			return value;
		else
			return def;
	}
};
