#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <cassert>

#include <vector>
#include <string>
#include <iostream>

#ifdef USE_CUDA
#include "cuda-dpd.h"
#endif
#include "funnel-obstacle.h"

#include "particles.h"
#include "funnel-bouncer.h"

int main()
{
    const float L = 40;
    const int Nm = 3;
    const int n = L * L * L * Nm;
    const float dt = 0.02;

    printf("START EQUILIBRATION\n");
    Particles particles(n, L);
    particles.equilibrate(0.1, 100 * dt, dt, nullptr);
    printf("STOP EQUILIBRATION\n");
    const float sandwich_half_width = L / 2 - 1.7;
#if 1
    TomatoSandwich bouncer(L);
    bouncer.radius2 = 4;
    bouncer.half_width = sandwich_half_width;
#else
    SandwichBouncer bouncer(L);
    bouncer.half_width = sandwich_half_width;
#endif
 printf("START CARVING\n");
    Particles remaining = bouncer.carve(particles);

    printf("STOP TOMATO\n");
    //bouncer.frozenLayer[0].lammps_dump("icy.dump", 0);
    //bouncer.frozenLayer[1].lammps_dump("icy.dump", 1);
    //bouncer.frozenLayer[2].lammps_dump("icy.dump", 2);

    remaining.name = "fluid";
    
    //remaining1.bouncer = &bouncer;
    remaining.yg = 0.01;
    remaining.steps_per_dump = 10;
    remaining.equilibrate(0.1, 1500 * dt, dt, &bouncer);
    printf("particles have been equilibrated");
}

    
