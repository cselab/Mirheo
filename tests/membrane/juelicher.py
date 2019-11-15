#!/usr/bin/env python

import numpy as np
import mirheo as mir
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--kb', type=float, default=0.0)
parser.add_argument('--C0', type=float, default=0.0)
parser.add_argument('--kad', type=float, default=0.0)
parser.add_argument('--DA0', type=float, default=0.0)
parser.add_argument('--ncells', type=int, default=1)
args = parser.parse_args()

dt = 0.001

ranks  = (1, 1, 1)
domain = (12, 8, 10)

u = mir.Mirheo(ranks, domain, dt, debug_level=3, log_filename='log', no_splash=True)

mesh_rbc = mir.ParticleVectors.MembraneMesh("rbc_mesh.off")
pv_rbc   = mir.ParticleVectors.MembraneVector("rbc", mass=1.0, mesh=mesh_rbc)
ic_rbc   = mir.InitialConditions.Membrane([[8.0, 4.0, 5.0,   1.0, 0.0, 0.0, 0.0]]*args.ncells)
u.registerParticleVector(pv_rbc, ic_rbc)

prm_rbc = {
    "x0"     : 0.457,
    "ka_tot" : 0.0,
    "kv_tot" : 0.0,
    "ka"     : 0.0,
    "ks"     : 0.0,
    "mpow"   : 2,
    "gammaC" : 0.0,
    "gammaT" : 0.0,
    "kBT"    : 0.0,
    "tot_area"   : 62.2242,
    "tot_volume" : 26.6649,

    "kb"  : args.kb,
    "C0"  : args.C0,
    "kad" : args.kad,
    "DA0" : args.DA0
}
    
int_rbc = mir.Interactions.MembraneForces("int_rbc", "wlc", "Juelicher", **prm_rbc, stress_free=False)
u.registerInteraction(int_rbc)
u.setInteraction(int_rbc, pv_rbc, pv_rbc)

dump_every = 1

u.registerPlugins(mir.Plugins.createForceSaver("forceSaver", pv_rbc))

u.registerPlugins(mir.Plugins.createDumpParticlesWithMesh("meshdump",
                                                          pv_rbc,
                                                          dump_every,
                                                          ["areas", "mean_curvatures", "forces"],
                                                          "h5/rbc-"))

u.run(2)

# nTEST: membrane.bending.juelicher
# cd membrane
# cp ../../data/rbc_mesh.off .
# mir.run --runargs "-n 2" ./juelicher.py --kb 1000.0
# mir.post ./utils/post.forces.py --file h5/rbc-00001.h5 --out forces.out.txt

# nTEST: membrane.bending.juelicher.C0
# cd membrane
# cp ../../data/rbc_mesh.off .
# mir.run --runargs "-n 2" ./juelicher.py --kb 1000.0 --C0 0.5
# mir.post ./utils/post.forces.py --file h5/rbc-00001.h5 --out forces.out.txt

# nTEST: membrane.bending.juelicher.AD
# cd membrane
# cp ../../data/rbc_mesh.off .
# mir.run --runargs "-n 2" ./juelicher.py --kad 1000.0 --DA0 1.0
# mir.post ./utils/post.forces.py --file h5/rbc-00001.h5 --out forces.out.txt

# nTEST: membrane.bending.juelicher.multiple
# cd membrane
# cp ../../data/rbc_mesh.off .
# mir.run --runargs "-n 2" ./juelicher.py --kb 1000.0 --C0 1.0 --kad 1000.0 --DA0 1.0 --ncells 4
# mir.post ./utils/post.forces.py --file h5/rbc-00001.h5 --out forces.out.txt
