#!/usr/bin/env python

import sys
import numpy as np
import mirheo as mir

import sys, argparse
sys.path.append("..")
from common.membrane_params import lina_parameters

parser = argparse.ArgumentParser()
parser.add_argument('--vis',     action='store_true', default=False)
parser.add_argument('--restart', action='store_true', default=False)
parser.add_argument("--ranks", type=int, nargs=3, default = [1,1,1])
args = parser.parse_args()

dt = 0.001
t_end = 2.5 + dt # time for one restart batch

ranks  = args.ranks
domain = (8, 8, 8)

u = mir.Mirheo(ranks, domain, dt, debug_level=3, log_filename='log',
              checkpoint_every = int(0.5/dt), no_splash=True)

nparts = 1000
np.random.seed(42)
pos = np.random.normal(loc   = [0.5, 0.5 * domain[1] + 1.0, 0.5 * domain[2]],
                       scale = [0.1, 0.3, 0.3],
                       size  = (nparts, 3))

vel = np.random.normal(loc   = [1.0, 0., 0.],
                       scale = [0.1, 0.01, 0.01],
                       size  = (nparts, 3))


pvSolvent = mir.ParticleVectors.ParticleVector('pv', mass = 1)
icSolvent = mir.InitialConditions.FromArray(pos=pos.tolist(), vel=vel.tolist())
vv        = mir.Integrators.VelocityVerlet('vv')
u.registerParticleVector(pvSolvent, icSolvent)
u.registerIntegrator(vv)
u.setIntegrator(vv, pvSolvent)

mesh_rbc = mir.ParticleVectors.MembraneMesh("rbc_mesh.off")
pv_rbc   = mir.ParticleVectors.MembraneVector("rbc", mass=1.0, mesh=mesh_rbc)
ic_rbc   = mir.InitialConditions.Membrane(
    [[0.5 * domain[0], 0.5 * domain[1], 0.5 * domain[2],   0.7071, 0.0, 0.7071, 0.0]]
)

u.registerParticleVector(pv_rbc, ic_rbc)
u.setIntegrator(vv, pv_rbc)

prm_rbc = lina_parameters(1.0)
int_rbc = mir.Interactions.MembraneForces("int_rbc", "wlc", "Kantor", **prm_rbc, stress_free=True)
u.registerInteraction(int_rbc)
u.setInteraction(int_rbc, pv_rbc, pv_rbc)


bb = mir.Bouncers.Mesh("bounce_rbc", "bounce_maxwell", kBT=0.0)
u.registerBouncer(bb)
u.setBouncer(bb, pv_rbc, pvSolvent)

if args.vis:
    dump_every = int(0.1 / dt)
    u.registerPlugins(mir.Plugins.createDumpParticles('partDump', pvSolvent, dump_every, [], 'h5/solvent-'))
    u.registerPlugins(mir.Plugins.createDumpMesh("mesh_dump", pv_rbc, dump_every, path="ply/"))


if args.restart:
    u.restart("restart/")

niters = int (t_end / dt)    
u.run(niters)

if args.restart and u.isComputeTask():
    pos = pv_rbc.getCoordinates()
    if len(pos) > 0:
        np.savetxt("pos.rbc.txt", pos)

# nTEST: restart.bounce
# set -eu
# cd restart
# rm -rf pos.rbc.txt pos.rbc.out.txt restart 
# cp ../../data/rbc_mesh.off .
# mir.run --runargs "-n 2" ./bounce.py --vis
# mir.run --runargs "-n 2" ./bounce.py --vis --restart
# mv pos.rbc.txt pos.rbc.out.txt

# nTEST: restart.bounce.mpi
# set -eu
# cd restart
# rm -rf pos.rbc.txt pos.rbc.out.txt restart 
# cp ../../data/rbc_mesh.off .
# mir.run --runargs "-n 4" ./bounce.py --ranks 2 1 1 --vis
# mir.run --runargs "-n 4" ./bounce.py --ranks 2 1 1 --vis --restart
# mv pos.rbc.txt pos.rbc.out.txt
