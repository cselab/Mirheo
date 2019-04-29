#!/usr/bin/env python

import numpy as np
import ymero as ymr
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--stress_free', action="store_true", default=False)
parser.add_argument('--ka', type=float, default=0.0)
parser.add_argument('--a3', type=float, default=0.0)
parser.add_argument('--a4', type=float, default=0.0)
parser.add_argument('--mu', type=float, default=0.0)
parser.add_argument('--b1', type=float, default=0.0)
parser.add_argument('--b2', type=float, default=0.0)
args = parser.parse_args()

dt = 0.001

ranks  = (1, 1, 1)
domain = (12, 8, 10)

u = ymr.ymero(ranks, domain, dt, debug_level=3, log_filename='log')

mesh_rbc = ymr.ParticleVectors.MembraneMesh("data/rbc.off", "data/sphere.off")
pv_rbc   = ymr.ParticleVectors.MembraneVector("rbc", mass=1.0, mesh=mesh_rbc)
ic_rbc   = ymr.InitialConditions.Membrane([[8.0, 4.0, 5.0,   1.0, 0.0, 0.0, 0.0]])
u.registerParticleVector(pv_rbc, ic_rbc)

prm_rbc = {
    "ka_tot" : 0.0,
    "kv_tot" : 0.0,
    "tot_area"   : 62.2242,
    "tot_volume" : 26.6649,
    "ka"     : args.ka,
    "a3"     : args.a3,
    "a4"     : args.a4,
    "mu"     : args.mu,
    "b1"     : args.b1,
    "b2"     : args.b2,
    "gammaC" : 0.0,
    "gammaT" : 0.0,
    "kBT"    : 0.0,
    "kb"     : 0.0,
    "theta"  : 0.0
}
    
int_rbc = ymr.Interactions.MembraneForces("int_rbc", "Lim", "Kantor", **prm_rbc, stress_free=args.stress_free)
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

# nTEST: membrane.shear.lim.ka.stressFree
# cd membrane
# cp ../../data/rbc_mesh.off .
# ymr.run --runargs "-n 2" ./lim.py --stress_free --ka 1000.0 > /dev/null
# ymr.post ./utils/post.forces.py --file h5/rbc-00001.h5 --out forces.out.txt

# nTEST: membrane.shear.lim.ka.nl.stressFree
# cd membrane
# cp ../../data/rbc_mesh.off .
# ymr.run --runargs "-n 2" ./lim.py --stress_free --ka 1000.0 --a3 2.0 --a4 4.0 > /dev/null
# ymr.post ./utils/post.forces.py --file h5/rbc-00001.h5 --out forces.out.txt

# nTEST: membrane.shear.lim.mu.stressFree
# cd membrane
# cp ../../data/rbc_mesh.off .
# ymr.run --runargs "-n 2" ./lim.py --stress_free --mu 1000.0 > /dev/null
# ymr.post ./utils/post.forces.py --file h5/rbc-00001.h5 --out forces.out.txt

# nTEST: membrane.shear.lim.mu.nl.stressFree
# cd membrane
# cp ../../data/rbc_mesh.off .
# ymr.run --runargs "-n 2" ./lim.py --stress_free --mu 1000.0 --b1 2.0 --b2 4.0 > /dev/null
# ymr.post ./utils/post.forces.py --file h5/rbc-00001.h5 --out forces.out.txt

