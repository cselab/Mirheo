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
    const float L = 10;
    const int Nm = 3;
    const int n = L * L * L * Nm;
    const float dt = 0.02;

    Particles particles(n, L);
    particles.equilibrate(.1, 10*dt, dt, nullptr);

    const float sandwich_half_width = L / 2 - 1.7;
#if 1
    TomatoSandwich bouncer(L);
    bouncer.radius2 = 4;
    bouncer.half_width = sandwich_half_width;
#else
    SandwichBouncer bouncer(L);
    bouncer.half_width = sandwich_half_width;
#endif

    Particles remaining1 = bouncer.carve(particles);
/*
    // check angle indexes
    Particles ppp[] = {Particles(0, L), Particles(0, L), Particles(0, L), Particles(0, L),
            Particles(0, L), Particles(0, L)};
    for (int k = 0; k < 3; ++k)
        for (int i = 0; i < bouncer.frozenLayer[k].n; ++i) {

            int slice = bouncer.angleIndex[k].getIndex(i);

            assert(slice < 6);
            ppp[slice].xp.push_back(bouncer.frozenLayer[k].xp[i]);
            ppp[slice].yp.push_back(bouncer.frozenLayer[k].yp[i]);
            ppp[slice].zp.push_back(bouncer.frozenLayer[k].zp[i]);
            ppp[slice].n++;
        }

    for (int i = 0; i < 6; ++i)
        ppp[i].lammps_dump("icy3.dump", i);
*/

    bouncer.frozenLayer[0].lammps_dump("icy.dump", 0);
    bouncer.frozenLayer[1].lammps_dump("icy.dump", 1);
    bouncer.frozenLayer[2].lammps_dump("icy.dump", 2);

    bouncer.frozen.lammps_dump("icy2.dump", 0);
    remaining1.name = "fluid";
    
    //remaining1.bouncer = &bouncer;
    remaining1.yg = 0.02;
    remaining1.steps_per_dump = 10;
    remaining1.equilibrate(.1, 1000*dt, dt, &bouncer);
    printf("particles have been equilibrated");
}

    
