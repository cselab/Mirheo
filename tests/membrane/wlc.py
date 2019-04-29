#!/usr/bin/env python

import numpy as np
import ymero as ymr
import sys, argparse

parser = argparse.ArgumentParser()
parser.add_argument('--stressFree', action="store_true", default=False)
args = parser.parse_args()

dt = 0.001

ranks  = (1, 1, 1)
domain = (12, 8, 10)

u = ymr.ymero(ranks, domain, dt, debug_level=3, log_filename='log')

mesh_rbc = ymr.ParticleVectors.MembraneMesh("rbc_mesh.off")
pv_rbc   = ymr.ParticleVectors.MembraneVector("rbc", mass=1.0, mesh=mesh_rbc)
ic_rbc   = ymr.InitialConditions.Membrane([[8.0, 4.0, 5.0,   1.0, 0.0, 0.0, 0.0]])
u.registerParticleVector(pv_rbc, ic_rbc)

prm_rbc = {
    "x0"     : 0.457,
    "ka_tot" : 0.0,
    "kv_tot" : 0.0,
    "ka"     : 0.0,
    "ks"     : 1000.0,
    "mpow"   : 2,
    "gammaC" : 0.0,
    "gammaT" : 0.0,
    "kBT"    : 0.0,
    "tot_area"   : 62.2242,
    "tot_volume" : 26.6649,
    "kb"     : 0.0,
    "theta"  : 0.0
}
    
int_rbc = ymr.Interactions.MembraneForces("int_rbc", "wlc", "Kantor", **prm_rbc, stress_free=args.stressFree)
u.registerInteraction(int_rbc)
u.setInteraction(int_rbc, pv_rbc, pv_rbc)


dump_every = 1

u.registerPlugins(ymr.Plugins.createForceSaver("forceSaver", pv_rbc))

u.registerPlugins(ymr.Plugins.createDumpParticlesWithMesh("meshdump",
                                                          pv_rbc,
                                                          dump_every,
                                                          [["forces", "vector"]],
                                                          "h5/rbc-"))

u.run(2)

# nTEST: membrane.shear.wlc
# cd membrane
# cp ../../data/rbc_mesh.off .
# ymr.run --runargs "-n 2" ./wlc.py > /dev/null
# ymr.post ./utils/post.forces.py --file h5/rbc-00001.h5 --out forces.out.txt

# nTEST: membrane.shear.wlc.stressFree.biconcave
# cd membrane
# cp ../../data/rbc_mesh.off .
# ymr.run --runargs "-n 2" ./wlc.py --stressFree > /dev/null
# ymr.post ./utils/post.forces.py --file h5/rbc-00001.h5 --out forces.out.txt
