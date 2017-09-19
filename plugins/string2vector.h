#pragma once

#include <string>
#include <vector>
#include <sstream>

inline std::vector<std::string> splitByDelim(std::string str, char delim = ',')
{
	std::stringstream sstream(str);
	std::string word;
	std::vector<std::string> splitted;

	while(std::getline(sstream, word, delim))
	{
		splitted.push_back(word);
	}

	return splitted;
}
