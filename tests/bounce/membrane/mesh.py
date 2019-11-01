#!/usr/bin/env python

import sys, argparse
import numpy as np
import mirheo as mir

sys.path.append("../..")
from common.membrane_params import lina_parameters

parser = argparse.ArgumentParser()
parser.add_argument('--subStep', action='store_true', default=False)
parser.add_argument('--xorigin', type=float, default=0)
parser.add_argument('--vis',     action='store_true', default=False)
args = parser.parse_args()

dt = 0.001

substeps = 10
if args.subStep:
    dt = dt * substeps

ranks  = (1, 1, 1)
domain = (8, 8, 8)

u = mir.Mirheo(ranks, domain, dt, debug_level=3, log_filename='log', no_splash=True)

nparts = 1000
np.random.seed(42)
pos = np.random.normal(loc   = [0.5 + args.xorigin, 0.5 * domain[1] + 1.0, 0.5 * domain[2]],
                       scale = [0.1, 0.3, 0.3],
                       size  = (nparts, 3))

vel = np.random.normal(loc   = [1.0, 0., 0.],
                       scale = [0.1, 0.01, 0.01],
                       size  = (nparts, 3))


pv_sol = mir.ParticleVectors.ParticleVector('pv', mass = 1)
ic_sol = mir.InitialConditions.FromArray(pos=pos.tolist(), vel=vel.tolist())
vv     = mir.Integrators.VelocityVerlet('vv')
u.registerParticleVector(pv_sol, ic_sol)
u.registerIntegrator(vv)
u.setIntegrator(vv, pv_sol)

xrbc = 0.5 * domain[0] + args.xorigin
while xrbc >= domain[0]: xrbc -= domain[0]

mesh_rbc = mir.ParticleVectors.MembraneMesh("rbc_mesh.off")
pv_rbc   = mir.ParticleVectors.MembraneVector("rbc", mass=1.0, mesh=mesh_rbc)
ic_rbc   = mir.InitialConditions.Membrane(
    [[xrbc, 0.5 * domain[1], 0.5 * domain[2],   0.7071, 0.0, 0.7071, 0.0]]
)

u.registerParticleVector(pv_rbc, ic_rbc)

prm_rbc = lina_parameters(1.0)
int_rbc = mir.Interactions.MembraneForces("int_rbc", "wlc", "Kantor", **prm_rbc, stress_free=True)

if args.subStep:
    integrator = mir.Integrators.SubStep('substep_membrane', substeps, [int_rbc])
    u.registerIntegrator(integrator)
    u.setIntegrator(integrator, pv_rbc)
else:
    u.setIntegrator(vv, pv_rbc)
    u.registerInteraction(int_rbc)
    u.setInteraction(int_rbc, pv_rbc, pv_rbc)


bb = mir.Bouncers.Mesh("bounce_rbc", "bounce_maxwell", kBT=0.0)
u.registerBouncer(bb)
u.setBouncer(bb, pv_rbc, pv_sol)


if args.vis:
    dump_every = int(0.1 / dt)
    u.registerPlugins(mir.Plugins.createDumpParticles('partDump', pv_sol, dump_every, [], 'h5/solvent-'))
    u.registerPlugins(mir.Plugins.createDumpMesh("mesh_dump", pv_rbc, dump_every, path="ply/"))

tend = int(5.0 / dt)
    
u.run(tend)

if pv_rbc is not None:
    rbc_pos = pv_rbc.getCoordinates()
    np.savetxt("pos.rbc.txt", rbc_pos)


# nTEST: bounce.membrane.mesh
# set -eu
# cd bounce/membrane
# rm -rf pos.rbc.txt pos.rbc.out.txt 
# cp ../../../data/rbc_mesh.off .
# mir.run --runargs "-n 2" ./mesh.py
# mv pos.rbc.txt pos.rbc.out.txt 

# nTEST: bounce.membrane.mesh.substep
# set -eu
# cd bounce/membrane
# rm -rf pos.rbc.txt pos.rbc.out.txt 
# cp ../../../data/rbc_mesh.off .
# mir.run --runargs "-n 2" ./mesh.py --subStep
# mv pos.rbc.txt pos.rbc.out.txt 

# nTEST: bounce.membrane.mesh.exchange
# set -eu
# cd bounce/membrane
# rm -rf pos.rbc.txt pos.rbc.out.txt 
# cp ../../../data/rbc_mesh.off .
# mir.run --runargs "-n 2" ./mesh.py --xorigin 4.1
# mv pos.rbc.txt pos.rbc.out.txt 
