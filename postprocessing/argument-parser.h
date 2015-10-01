/*
 *  ArgumentParser.h
 *  Cubism
 *
 *This argument parser assumes that all arguments are optional ie, each of the argument names is preceded by a '-'
 *all arguments are however NOT optional to avoid a mess with default values and returned values when not found!
 *
 *More converter could be required:
 *add as needed
 *TypeName as{TypeName}() in Value
 *
 *  Created by Christian Conti on 6/7/10. That is a long time ago.
 *  Modified by Diego Rossinelli several times after his dreadlocks hair cut.
 *  Copyright 2010 ETH Zurich. All rights reserved.
 *
 */

#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <map>
#include <vector>
#include <string>
#include <sstream>
#include <iterator>

using namespace std;

class Value
{
private:
    string content;

public:

Value() : content("") {}

Value(string content_) : content(content_) { /*printf("%s\n",content.c_str());*/ }

    double asDouble(double def=0) const
    {
	if (content == "") return def;
	return (double) atof(content.c_str());
    }

    int asInt(int def=0) const
    {
	if (content == "") return def;
	return atoi(content.c_str());
    }

    bool asBool(bool def=false) const
    {
	if (content == "") return def;
	if (content == "0") return false;
	if (content == "false") return false;

	return true;
    }

    string asString(string def="") const
    {
	if (content == "") return def;

	return content;
    }

    vector<float> asVecFloat(const int musthave_size = -1) const
    {
	//printf("mycontent is %s\n", content.c_str());
	std::stringstream ss(content);
	//assert(ss.good());
	vector<float> retval;
	double e;

	while (ss >> e)
	{
	    retval.push_back(e);
	    //  printf("reading %f\n", e);
	    if (ss.peek() == ',')
		ss.ignore();
	}

	if (musthave_size > 0)
	    assert(musthave_size == (int)retval.size());

	return retval;
    }
};

class ArgumentParser
{
private:

    map<string,Value> mapArguments;

    const int iArgC;
    const char** vArgV;
    bool bStrictMode, bVerbose;

    const char delimiter;
public:

    Value operator()(const string arg)
    {
	map<string,Value>::const_iterator it = mapArguments.find(arg);

	if (bStrictMode)
	{
	    if (it == mapArguments.end())
	    {
		printf("Runtime option NOT SPECIFIED! ABORTING! name: %s\n",arg.data());
		abort();
	    }
	}

	if (bVerbose)
	    printf("%s is %s\n", arg.data(), mapArguments[arg].asString().data());

	if (it != mapArguments.end())
	    return mapArguments[arg];
	else
	    return Value();
    }

    bool check(const string arg) const
    {
	return mapArguments.find(arg) != mapArguments.end();
    }

ArgumentParser(const int argc, const char ** argv, bool bVerbose = false, const char delimiter = '=') :
    mapArguments(), iArgC(argc), vArgV(argv), bStrictMode(false), bVerbose(bVerbose), delimiter(delimiter)
    {
	for (int i = 1; i<argc; i++)
	{
	    int sep = 0;
	    while(argv[i][sep] != '\0' && argv[i][sep] != delimiter)
		++sep;

	    string value;

	    if (argv[i][sep] != '\0')
		value = string(argv[i] + sep + 1);
	    else
		value = "1";

	    mapArguments[string(argv[i], sep)] = Value(value);
	}

	mute();
    }

ArgumentParser(vector<string> args, bool bVerbose = false, const char delimiter = '='):
    mapArguments(), iArgC(args.size()), vArgV(NULL), bStrictMode(false), bVerbose(bVerbose), delimiter(delimiter)
    {
	for(vector<string>::iterator it = args.begin(); it != args.end(); ++it)
	{
	    const char * arg = it->c_str();

	    int sep = 0;
	    while(arg[sep] != '\0' && arg[sep] != delimiter)
		++sep;

	    string value;

	    if (arg[sep] != '\0')
		value = string(arg + sep + 1);
	    else
		value = "1";

	    mapArguments[string(arg, sep)] = Value(value);
	}

	mute();
    }

    int getargc() const { return iArgC; }

    const char** getargv() const { return vArgV; }

    void set_strict_mode()
    {
	bStrictMode = true;
    }

    void unset_strict_mode()
    {
	bStrictMode = false;
    }

    void mute()
    {
	bVerbose = false;
    }

    void loud()
    {
	bVerbose = true;
    }

    void print_arguments(FILE * f = stdout)
    {
	printf("PRINTOUT OF THE RUNTIME OPTIONS\n");

	for(map<string,Value>::const_iterator it=mapArguments.begin(); it!=mapArguments.end(); it++)
	    fprintf(f, "%s: <%s>\n", it->first.c_str(), it->second.asString().c_str());

	printf("END OF THE PRINTOUT.\n");
    }

    void print_arguments(string path2log)
    {
	FILE * f = fopen(path2log.c_str(), "w");

	if (f == NULL)
	{
	    printf("could not save the log to <%s>. Exiting now\n", path2log.c_str());
	    exit(-1);
	}

	print_arguments(f);

	fclose(f);
    }

    vector<string> find(string name)
    {
	map<string, Value>::iterator itb = mapArguments.lower_bound(name);
	map<string, Value>::iterator ite = mapArguments.end();

	vector<string> retval;
	for(map<string, Value>::iterator it = itb; it != ite; ++it)
	{
	    if (it->first.find(name) == string::npos)
		break;

	    retval.push_back(it->first);
	}
	return retval;
    }
};
