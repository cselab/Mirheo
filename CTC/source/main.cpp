/*
 *  proj1.cpp
 *  hpchw
 *
 *  Created by Dmitry Alexeev on 15.10.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */


#include <iostream>
#include <fstream>
#include <list>

#include "timer.h"
#include "ArgumentParser.h"
#include "Simulation.h"
#include "Savers.h"
#include "ErrorHandling.h"
#include "minIni.h"
#include "Misc.h"

using namespace std;
using namespace ArgumentParser;
using namespace ErrorHandling;

int ErrorHandling::debugLvl;
int ErrorHandling::rank;

minIni *configParser;


#define TYPES 2

template<int N> string Saver<N>::folder("");

int main (int argc, char **argv)
{
    string folder = "res/";
    string config = "/Users/alexeedm/Documents/projects/CTC/CTC/makefiles/100relax.ini";
    debugLvl = 2;
	
	vector<OptionStruct> vopts =
	{
        {'f', "folder",     STRING, "Result folder",      &folder},
        {'d', "debug",      INT,    "DebugLvl",           &debugLvl},
        {'c', "config",     STRING, "Config file",        &config},
	};
	
    Parser parser(vopts);
	parser.parse(argc, argv);
	
    configParser = new minIni(config);
    folder = configParser->gets("Savers", "resFolder", folder);
	
	Saver<TYPES>::makedir(folder + "/");
	Saver<TYPES>     *enSaver   = new SaveEnergy<TYPES>      (configParser->gets("Savers", "energyFile", "nrg.txt"));
	Saver<TYPES>     *posSaver  = new SavePos<TYPES>         (configParser->gets("Savers", "positionFile", "pos.xyz"));
	Saver<TYPES>     *linSaver  = new SaveLinMom<TYPES>      (configParser->gets("Savers", "linMomentumFile", "lin.txt"));
	Saver<TYPES> 	 *angSaver  = new SaveAngMom<TYPES>	     (configParser->gets("Savers", "angMomentumFile", "ang.txt"));
	Saver<TYPES>     *massSaver = new SaveCenterOfMass<TYPES>(configParser->gets("Savers", "COMFile", "com.txt"));
	Saver<TYPES>     *timeSaver = new SaveTiming<TYPES>      (configParser->gets("Savers", "timingFile", "screen"));
    Saver<TYPES>     *tempSaver = new SaveTemperature<TYPES> (configParser->gets("Savers", "temperatureFile", "temp.txt"));
    Saver<TYPES>     *restarter = new SaveRestart<TYPES>     (configParser->gets("Savers", "restartFile", "restart"));
    Saver<TYPES>     *strSaver  = new SaveStrain<TYPES>      (configParser->gets("Savers", "strainFile", "strain.txt"));

    int n0       = configParser->geti("Particles", "Ndpd", 3500);
    int n1       = configParser->geti("Particles", "Nsem", 125);
    real rCut0 = configParser->getf("Particles", "cutdpd", 1);
    real rCut1 = configParser->getf("Particles", "cutsem", 2.5);
    
    real temp = configParser->getf("Basic", "temperature", 0.1);
    real dt   = configParser->getf("Basic", "dt", 0.001);
    real end  = configParser->getf("Basic", "endTime",  100);

    
    vector<int>  nums  = {n0, n1};
    vector<real> rCuts = {rCut0, rCut1};
	Simulation<TYPES> simulation(dt);
    
    string restartFile = configParser->gets("Basic", "restartFrom", "none");
    if (restartFile == "none")
        simulation.setIC(nums, rCuts);
    else
        simulation.loadRestart(restartFile, rCuts);
    
	simulation.registerSaver(enSaver,   configParser->geti("Savers", "energyPeriod", 100));
	simulation.registerSaver(posSaver,  configParser->geti("Savers", "positionPeriod", 100));
	simulation.registerSaver(linSaver,  configParser->geti("Savers", "linMomentumPeriod", 100));
	simulation.registerSaver(angSaver,  configParser->geti("Savers", "angMomentumPeriod", 100));
	simulation.registerSaver(massSaver, configParser->geti("Savers", "COMPeriod", 100));
	simulation.registerSaver(timeSaver, configParser->geti("Savers", "timingPeriod", 100));
    simulation.registerSaver(tempSaver, configParser->geti("Savers", "temperaturePeriod", 100));
    simulation.registerSaver(restarter, configParser->geti("Savers", "restartPeriod", 1000));
    simulation.registerSaver(strSaver,  configParser->geti("Savers", "strainPeriod", 100));
	simulation.profiler.millisec();
	
	int iters = ceil(end / dt);
	for (int i=0; i<=iters; i++)
	{
		if (i % 500 == 0) printf("Simulation time is %f\n", i * dt);
		simulation.runOneStep();
	}
}



