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

string Saver::folder("");

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
	
	Saver::makedir(folder + "/");
	Saver *enSaver   = new SaveEnergy      (configParser->gets("Savers", "energyFile", "nrg.txt"));
	Saver *posSaver  = new SavePos         (configParser->gets("Savers", "positionFile", "pos.xyz"));
	Saver *linSaver  = new SaveLinMom      (configParser->gets("Savers", "linMomentumFile", "lin.txt"));
	Saver *angSaver  = new SaveAngMom	   (configParser->gets("Savers", "angMomentumFile", "ang.txt"));
	Saver *massSaver = new SaveCenterOfMass(configParser->gets("Savers", "COMFile", "com.txt"));
	Saver *timeSaver = new SaveTiming      (configParser->gets("Savers", "timingFile", "screen"));
    Saver *tempSaver = new SaveTemperature (configParser->gets("Savers", "temperatureFile", "temp.txt"));
    Saver *restarter = new SaveRestart     (configParser->gets("Savers", "restartFile", "restart"));
    Saver *strSaver  = new SaveStrain      (configParser->gets("Savers", "strainFile", "strain.txt"));
    Saver *rdfSaver  = new SaveRdf         (configParser->gets("Savers", "rdfFile", "rdf.txt"), configParser->geti("Savers", "rdfPrnPeriod"));

    
    real temp = configParser->getf("Basic", "temperature", 0.1);
    real dt   = configParser->getf("Basic", "dt", 0.001);
    real end  = configParser->getf("Basic", "endTime",  100);

    
    int nTypes = configParser->geti("Particles", "types", 2);
	Simulation simulation(nTypes, dt);
    
    string restartFile = configParser->gets("Basic", "restartFrom", "none");
    if (restartFile == "none")
        simulation.setIC();
    else
        simulation.loadRestart(restartFile);
    
	simulation.registerSaver(enSaver,   configParser->geti("Savers", "energyPeriod", 0));
	simulation.registerSaver(posSaver,  configParser->geti("Savers", "positionPeriod", 0));
	simulation.registerSaver(linSaver,  configParser->geti("Savers", "linMomentumPeriod", 0));
	simulation.registerSaver(angSaver,  configParser->geti("Savers", "angMomentumPeriod", 0));
	simulation.registerSaver(massSaver, configParser->geti("Savers", "COMPeriod", 0));
	simulation.registerSaver(timeSaver, configParser->geti("Savers", "timingPeriod", 0));
    simulation.registerSaver(tempSaver, configParser->geti("Savers", "temperaturePeriod", 0));
    simulation.registerSaver(restarter, configParser->geti("Savers", "restartPeriod", 0));
    simulation.registerSaver(strSaver,  configParser->geti("Savers", "strainPeriod", 0));
    simulation.registerSaver(rdfSaver,  configParser->geti("Savers", "rdfPeriod", 0));
	simulation.profiler.millisec();
	
	int iters = ceil(end / dt);
	for (int i=0; i<=iters; i++)
	{
		if (i % 500 == 0) printf("Simulation time is %f\n", i * dt);
		simulation.runOneStep();
	}
}



