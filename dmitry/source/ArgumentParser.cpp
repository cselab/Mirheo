/*
 *  ArgumentParser.cpp
 *  rl
 *
 *  Created by Dmitry Alexeev on 02.05.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#include <getopt.h>
#include <map>
#include <cstdlib>
#include <cstdio>

#include "ArgumentParser.h"

namespace ArgumentParser
{
	
	Parser::Parser(const std::vector<OptionStruct> optionsMap):opts(optionsMap)
	{
		ctrlString = "";
		nOpt = opts.size();
		long_options = new option[nOpt + 1];
		
		for (int i=0; i<nOpt; i++)
		{
			long_options[i].name = opts[i].longOpt.c_str();
			long_options[i].flag = NULL;
			long_options[i].val = opts[i].shortOpt;

			if (opts[i].type == NONE) long_options[i].has_arg = no_argument;
			else                      long_options[i].has_arg = required_argument;

			
			ctrlString += opts[i].shortOpt;
			if (opts[i].type != NONE) ctrlString += ':';
			
			if (optsMap.find(long_options[i].val) != optsMap.end())
			{
				fprintf(stderr, "Duplicate short options in declaration, please correct the source code\n");
				exit(1);
			}
			else optsMap[long_options[i].val] = opts[i];
			
		}
		
		long_options[nOpt].has_arg = 0;
		long_options[nOpt].flag = NULL;
		long_options[nOpt].name = NULL;
		long_options[nOpt].val  = 0;
	}
	
	void Parser::parse(int argc, char * const * argv)
	{
		int option_index = 0;
		int c = 0;
		
		while((c = getopt_long (argc, argv, ctrlString.c_str(), long_options, &option_index)) != -1)
		{
			if (c == 0) continue;
			if (optsMap.find(c) == optsMap.end())
			{
				printf("Available options:\n");
				
				for (int i=0; i<nOpt; i++)
				{
					OptionStruct& myOpt = opts[i];
					if (myOpt.longOpt.length() > 4)
					{
						printf("-%c  or  --%s \t: %s\n", myOpt.shortOpt, myOpt.longOpt.c_str(), myOpt.description.c_str());
					}
					else
					{
						printf("-%c  or  --%s \t\t: %s\n", myOpt.shortOpt, myOpt.longOpt.c_str(), myOpt.description.c_str());
					}
				}
				exit(1);
			}
			
			OptionStruct& myOpt = optsMap[c];
			
			switch (myOpt.type)
			{
				case NONE:
					*((bool*)myOpt.value) = true;
					break;
					
				case INT:
					*((int*)myOpt.value) = atoi(optarg);
					break;
					
				case UINT:
					*((unsigned int*)myOpt.value) = atoi(optarg);
					break;
					
				case DOUBLE:
					*((double*)myOpt.value) = atof(optarg);
					break;
					
				case STRING:
					*((string*)myOpt.value) = optarg;
					break;
					
			}
		}
		
		for (int i=0; i<nOpt; i++)
		{
			OptionStruct& myOpt = opts[i];
			printf("%s: ", myOpt.description.c_str());
			
			switch (myOpt.type)
			{
				case NONE:
					printf( ( *((bool*)myOpt.value)) ? "enabled" : "disabled" );
					break;
					
				case INT:
					printf("%d", *((int*)myOpt.value));
					break;
					
				case UINT:
					printf("%d", *((unsigned int*)myOpt.value));
					break;
					
				case DOUBLE:
					printf("%f", *((double*)myOpt.value));
					break;
					
				case STRING:
					printf("%s", ((string*)myOpt.value)->c_str());
					break;
			}
			
			printf("\n");
		}
			
	}
}
