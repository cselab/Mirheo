/*
 *  ArgumentParser.h
 *  rl
 *
 *  Created by Dmitry Alexeev on 02.05.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once

#include <string>
#include <vector>
#include <map>

#include <getopt.h>

using namespace std;

namespace ArgumentParser
{
	enum Types { BOOL, INT, DOUBLE, CHAR, STRING };
	
	struct OptionStruct
	{
		char   shortOpt;
		string longOpt;
		Types  type;
		string description;
		void*  value;
		
		template <typename T>
		OptionStruct(char shortOpt, string longOpt, Types type, string description, T* val, T defVal) :
		shortOpt(shortOpt), longOpt(longOpt), type(type), description(description)
		{
			value = (void*)val;
			*val = defVal;
		}
		
		OptionStruct() {};
	};

	class Parser
	{
	private:
		bool output;
		int nOpt;
		vector<OptionStruct> opts;
		map<char, OptionStruct> optsMap;
		std::vector<option> long_options;
		string ctrlString;
		
	public:
		
		Parser(const std::vector<OptionStruct>& optionsMap, bool output = true);
		void parse(int argc, char * const * argv);
	};
}
