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

using namespace std;
using namespace ArgumentParser;

string Saver::folder("");

int main (int argc, char **argv)
{
	int n, iters;
	double temp, gamma, alpha, cutRad, dt, endTime, L;
	string folder = "res/";

	n       = 100;
	gamma   = 1;
	alpha   = 0.1;
	cutRad  = -1;
	dt      = 1e-3;
	endTime = 1;
	L       = 2;
    temp    = 1;
	
	vector<OptionStruct> vopts =
	{
		{'n', "particles",  INT,    "Number of particles",     &n},
		{'t', "dt",         DOUBLE, "Simulation timestep",     &dt},
		{'f', "end_time",   DOUBLE, "End time of simulaiton",  &endTime},
        {'c', "cutoff",		DOUBLE, "Cutoff in sigma units",   &cutRad},
		{'l', "length",     DOUBLE, "Domain lenght",           &L},
		{'b', "temp",       DOUBLE, "Temperature",             &temp},
		{'a', "alpha",      DOUBLE, "Alpha",                   &alpha},
		{'g', "gamma",      DOUBLE, "Gamma",                   &gamma},
	};
	
    Parser parser(vopts);
	parser.parse(argc, argv);
	
	if (cutRad < 0)
	{
		cutRad = 3;
		printf("Cut-off is automatically set to %f\n", cutRad);
	}
	
	Saver::makedir(folder + "/");
	SaveEnergy       *enSaver   = new SaveEnergy      ("nrg.txt");
	SavePos          *posSaver  = new SavePos         ("pos.xyz");
	SaveLinMom       *linSaver  = new SaveLinMom      ("lin.txt");
	SaveAngMom		 *angSaver  = new SaveAngMom	  ("ang.txt");
	SaveCenterOfMass *massSaver = new SaveCenterOfMass("mass.txt");
	SaveTiming       *timeSaver = new SaveTiming      (&cout);
    SaveTemperature  *tempSaver = new SaveTemperature (&cout);
	
	Simulation simulation(n, temp, 1, alpha, gamma, cutRad, dt, L);
//	simulation.registerSaver(enSaver, 100);
	simulation.registerSaver(posSaver, 100);
	simulation.registerSaver(linSaver, 100);
//	simulation.registerSaver(angSaver, 100);
//	simulation.registerSaver(massSaver, 100);
	simulation.registerSaver(timeSaver, 100);
    simulation.registerSaver(tempSaver, 100);
	simulation.profiler.millisec();
	
	iters = ceil(endTime / dt);
	for (int i=0; i<=iters; i++)
	{
		if (i % 500 == 0) printf("Simulation time is %f\n", i * dt);
		simulation.runOneStep();
	}
}



