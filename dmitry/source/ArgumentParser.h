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

using namespace std;

namespace ArgumentParser
{
	enum Types { NONE, INT, UINT, DOUBLE, CHAR, STRING };
	
	struct OptionStruct
	{
		char   shortOpt;
		string longOpt;
		Types  type;
		string description;
		void*  value;
	};

	class Parser
	{
	private:
		int nOpt; 
		vector<OptionStruct> opts;
		map<char, OptionStruct> optsMap;
		struct option* long_options;
		string ctrlString;
		
	public:
		
		Parser(const std::vector<OptionStruct> optionsMap);
		void parse(int argc, char * const * argv);
	};
}
