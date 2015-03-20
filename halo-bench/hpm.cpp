/*
 *  hpm.cpp
 *  Part of CTC/halo-bench/
 *
 *  Created and authored by Panagiotis Chatzidoukas on 2015-03-09.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <vector>
#include <string>
#include <unistd.h>
#include <algorithm>
#include <sys/time.h>

using namespace std;

class HPM
{
	map<string, vector<double> > hpm;

double wtime()
{
        struct timeval t;
        gettimeofday(&t, NULL);
        return (double)t.tv_sec + (double)t.tv_usec*1.0E-6;
}


public:

void HPM_Start(string str)
{
	double t = wtime();
	hpm[str].push_back(t);
}

void HPM_Stop(string str)
{
	double t = wtime();
	double t0 = hpm[str].back();
	hpm[str].back() = (t - t0)*1e6;
}

void HPM_Report()
{
	for( map<string, vector<double> >::const_iterator ptr=hpm.begin(); ptr!=hpm.end(); ptr++) {
		cout << ptr->first << ": ";
		for( vector<double>::const_iterator eptr=ptr->second.begin(); eptr!=ptr->second.end(); eptr++)
			cout << *eptr << " ";
		cout << endl;
	}

#if 0
	for( map<string, vector<double> >::iterator ptr=hpm.begin(); ptr!=hpm.end(); ptr++) {
		cout << ptr->first << " ";
		int n = ptr->second.size();
		cout << "(" << n << "): ";
		sort(ptr->second.begin(), ptr->second.end());
		const int I1 = n*.1;
		const int I2 = n*.5;
		const int I3 = n*.9;

//		for( vector<double>::const_iterator eptr=ptr->second.begin(); eptr!=ptr->second.end(); eptr++)
//			cout << *eptr << " ";
//		cout << "Min=" << ptr->second[0] << " 10\%=" << ptr->second[I1] << " 50\%=" << ptr->second[I2] << " 90\%=" << ptr->second[I2] << " Max=" << ptr->second[n-1];
		cout << " " << ptr->second[0] << " " << ptr->second[I1] << " " << ptr->second[I2] << " " << ptr->second[I2] << " " << ptr->second[n-1];
		cout << endl;
	}
#endif
}

void HPM_Stats()
{
	for( map<string, vector<double> >::iterator ptr=hpm.begin(); ptr!=hpm.end(); ptr++) {
		cout << ptr->first << " ";
		int n = ptr->second.size();
		cout << "(" << n << "): ";
		sort(ptr->second.begin(), ptr->second.end());
		const int I1 = n*.1;
		const int I2 = n*.5;
		const int I3 = n*.9;

		cout << " " << ptr->second[0] << " " << ptr->second[I1] << " " << ptr->second[I2] << " " << ptr->second[I2] << " " << ptr->second[n-1];
		cout << endl;

	}
}

};


#if 0
HPM hpm;


int main()
{
	for (int i = 0; i < 10; i++) {
		double t1 = wtime();
		hpm.HPM_Start("t2");
		usleep(100*1000);
		hpm.HPM_Stop("t2");
		double t2 = wtime();
		//hpm["t1"].push_back(t2-t1);
	}

	hpm.HPM_Stats();

	return 0;

}
#endif
