#!/usr/bin/env python

import sys
import numpy as np

import ymero as ymr

import sys, argparse
sys.path.append("..")
from common.membrane_params import lina_parameters

parser = argparse.ArgumentParser()
parser.add_argument('--vis',     action='store_true', default=False)
parser.add_argument('--restart', action='store_true', default=False)
args = parser.parse_args()

dt = 0.001
t_end = 2.5 + dt # time for one restart batch

ranks  = (1, 1, 1)
domain = (8, 8, 8)

u = ymr.ymero(ranks, domain, dt, debug_level=3, log_filename='log', checkpoint_every = int(0.5/dt), no_splash=True)

nparts = 1000
pos = np.random.normal(loc   = [0.5, 0.5 * domain[1] + 1.0, 0.5 * domain[2]],
                       scale = [0.1, 0.3, 0.3],
                       size  = (nparts, 3))

vel = np.random.normal(loc   = [1.0, 0., 0.],
                       scale = [0.1, 0.01, 0.01],
                       size  = (nparts, 3))


pvSolvent = ymr.ParticleVectors.ParticleVector('pv', mass = 1)
icSolvent = ymr.InitialConditions.FromArray(pos=pos.tolist(), vel=vel.tolist())
vv        = ymr.Integrators.VelocityVerlet('vv')
u.registerParticleVector(pvSolvent, icSolvent)
u.registerIntegrator(vv)
u.setIntegrator(vv, pvSolvent)

meshRbc = ymr.ParticleVectors.MembraneMesh("rbc_mesh.off")
pvRbc   = ymr.ParticleVectors.MembraneVector("rbc", mass=1.0, mesh=meshRbc)
icRbc   = ymr.InitialConditions.Membrane(
    [[0.5 * domain[0], 0.5 * domain[1], 0.5 * domain[2],   0.7071, 0.0, 0.7071, 0.0]]
)

u.registerParticleVector(pvRbc, icRbc)
u.setIntegrator(vv, pvRbc)

prm_rbc = lina_parameters(1.0)
int_rbc = ymr.Interactions.MembraneForces("int_rbc", "wlc", "Kantor", **prm_rbc, stress_free=True)
u.registerInteraction(int_rbc)
u.setInteraction(int_rbc, pvRbc, pvRbc)


bb = ymr.Bouncers.Mesh("bounceRbc", kbt=0.0)
u.registerBouncer(bb)
u.setBouncer(bb, pvRbc, pvSolvent)

if args.vis:
    dump_every = int(0.1 / dt)
    u.registerPlugins(ymr.Plugins.createDumpParticles('partDump', pvSolvent, dump_every, [], 'h5/solvent-'))
    u.registerPlugins(ymr.Plugins.createDumpMesh("mesh_dump", pvRbc, dump_every, path="ply/"))


if args.restart:
    u.restart("restart/")

niters = int (t_end / dt)    
u.run(niters)

if args.restart and pvRbc is not None:
    rbc_pos = pvRbc.getCoordinates()
    np.savetxt("pos.rbc.txt", rbc_pos)


# nTEST: restart.bounce
# set -eu
# cd restart
# rm -rf pos.rbc.txt pos.rbc.out.txt restart 
# cp ../../data/rbc_mesh.off .
# ymr.run --runargs "-n 2" ./bounce.py --vis
# ymr.run --runargs "-n 2" ./bounce.py --vis --restart
# mv pos.rbc.txt pos.rbc.out.txt
