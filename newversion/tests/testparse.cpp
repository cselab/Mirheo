#include "iniparser.h"

int main()
{
	IniParser parser("config.cfg");

	std::cout << parser.getInt("Section1", "g", -100) << std::endl;
	std::cout << parser.getDouble("Section2", "num", -1.234) << std::endl;
	std::cout << parser.getDouble("Section3", "num", 144) << std::endl;
	std::cout << parser.getString("Section1", "ddd") << std::endl;
	std::cout << parser.getInt("Section2", "nn", -1) << std::endl;

	return 0;
}
