#!/usr/bin/env python

import sys, argparse
import numpy as np
import mirheo as mir

sys.path.append("..")
from common.membrane_params import lina_parameters

parser = argparse.ArgumentParser()
parser.add_argument('--density', type=float)
parser.add_argument('--axes',    type=float, nargs=3)
parser.add_argument('--coords',  type=str)
parser.add_argument('--bounce_back', action='store_true', default=False)
parser.add_argument('--substep',    action='store_true', default=False)
args = parser.parse_args()

tend = 10.0
dt = 0.001
a = 1.0

if args.substep:
    substeps = 10
    dt = dt * substeps

ranks  = (1, 1, 1)
domain = (12, 8, 10)

u = mir.mirheo(ranks, domain, dt, debug_level=3, log_filename='log', no_splash=True)

pv_sol = mir.ParticleVectors.ParticleVector('solvent', mass = 1)
ic_sol = mir.InitialConditions.Uniform(density=args.density)
u.registerParticleVector(pv=pv_sol, ic=ic_sol)


mesh_rbc = mir.ParticleVectors.MembraneMesh("rbc_mesh.off")
pv_rbc   = mir.ParticleVectors.MembraneVector("rbc", mass=1.0, mesh=mesh_rbc)

com_q_rbc = [[2.0, 5.0, 5.0,   1.0, np.pi/2, np.pi/3, 0.0],
             [6.0, 3.0, 5.0,   1.0, np.pi/2, np.pi/3, 0.0]]

com_q_rig = [[4.0, 4.0, 5.0,   1.0, np.pi/2, np.pi/3, 0.0]]

ic_rbc   = mir.InitialConditions.Membrane(com_q_rbc)
u.registerParticleVector(pv_rbc, ic_rbc)

dpd = mir.Interactions.Pairwise('dpd', rc=1.0, kind="DPD", a=10.0, gamma=10.0, kBT=0.01, power=0.25)
cnt = mir.Interactions.Pairwise('cnt', rc=1.0, kind="RepulsiveLJ", epsilon=0.35, sigma=0.8, max_force=400.0)

prm_rbc = lina_parameters(1.0)    
int_rbc = mir.Interactions.MembraneForces("int_rbc", "wlc", "Kantor", **prm_rbc, stress_free=True)

coords = np.loadtxt(args.coords).tolist()
pv_ell = mir.ParticleVectors.RigidEllipsoidVector('ellipsoid', mass=1, object_size=len(coords), semi_axes=args.axes)
ic_ell = mir.InitialConditions.Rigid(com_q=com_q_rig, coords=coords)
vv_ell = mir.Integrators.RigidVelocityVerlet("ellvv")

u.registerParticleVector(pv=pv_ell, ic=ic_ell)
u.registerIntegrator(vv_ell)
u.setIntegrator(vv_ell, pv_ell)

u.registerInteraction(dpd)
u.registerInteraction(cnt)

if args.substep:
    integrator = mir.Integrators.SubStep('substep_membrane', substeps, int_rbc)
    u.registerIntegrator(integrator)
    u.setIntegrator(integrator, pv_rbc)
else:
    vv = mir.Integrators.VelocityVerlet('vv')
    u.registerInteraction(int_rbc)
    u.setInteraction(int_rbc, pv_rbc, pv_rbc)
    u.registerIntegrator(vv)
    u.setIntegrator(vv, pv_rbc)

u.setInteraction(dpd, pv_sol, pv_sol)
u.setInteraction(dpd, pv_sol, pv_rbc)
u.setInteraction(dpd, pv_sol, pv_ell)
u.setInteraction(cnt, pv_rbc, pv_rbc)
u.setInteraction(cnt, pv_rbc, pv_ell)
u.setInteraction(cnt, pv_ell, pv_ell)

vv_dp = mir.Integrators.VelocityVerlet_withPeriodicForce('vv_dp', force=a, direction='x')
u.registerIntegrator(vv_dp)
u.setIntegrator(vv_dp, pv_sol)

belonging_checker = mir.BelongingCheckers.Ellipsoid("ellipsoidChecker")

u.registerObjectBelongingChecker(belonging_checker, pv_ell)
u.applyObjectBelongingChecker(belonging_checker, pv=pv_sol, correct_every=0, inside="none", outside="")

if args.bounce_back:
    bb = mir.Bouncers.Ellipsoid("bounce_ellipsoid")
    u.registerBouncer(bb)
    u.setBouncer(bb, pv_ell, pv_sol)


debug = 0

if debug:
    dump_every=(int)(0.15/dt)
    u.registerPlugins(mir.Plugins.createDumpMesh("mesh_dump", pv_rbc, dump_every, "ply/"))
    u.registerPlugins(mir.Plugins.createDumpObjectStats("objStats", ov=pv_ell, dump_every=dump_every, path="stats"))
    u.registerPlugins(mir.Plugins.createDumpXYZ('xyz', pv_ell, dump_every, "xyz/"))

nsteps = (int) (tend/dt)
u.run(nsteps)

if pv_rbc is not None:
    rbc_pos = pv_rbc.getCoordinates()
    np.savetxt("pos.rbc.txt", rbc_pos)


# nTEST: contact.mix.dp
# cd contact
# rm -rf pos.rbc.out.txt pos.rbc.txt
# f="pos.txt"
# rho=8.0; ax=2.0; ay=1.0; az=1.0
# cp ../../data/ellipsoid_coords_${rho}_${ax}_${ay}_${az}.txt $f
# cp ../../data/rbc_mesh.off .
# mir.run --runargs "-n 2" ./mix.dp.py --density $rho --axes $ax $ay $az --coords $f
# mv pos.rbc.txt pos.rbc.out.txt 
