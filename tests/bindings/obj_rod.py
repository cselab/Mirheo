#!/usr/bin/env python

import numpy as np
import mirheo as mir
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--axes',   type=float, nargs=3)
parser.add_argument('--coords', type=str)
parser.add_argument('--vis',    action='store_true', default=False)
parser.add_argument('--drag',   type=float,          default=0.0)
args = parser.parse_args()

ranks  = (1, 1, 1)
domain = [16, 16, 16]

dt    = 1e-3
t_end = 10.0
t_dump_every = 1.0
L = 14.0
num_segments = 10
mass = 1.0

u = mir.mirheo(ranks, tuple(domain), dt, debug_level=3, log_filename='log', no_splash=True)

# rod

com_q_rod = [[ 0.5 * domain[0],
               0.5 * domain[1],
               0.5 * domain[2] - L/2,
               1.0, 0.0, 0.0, 0.0]]

def center_line(s): return (0, 0, (0.5-s) * L)
def torsion(s):     return 0.0

def length(a, b):
    return np.sqrt(
        (a[0] - b[0])**2 +
        (a[1] - b[1])**2 +
        (a[2] - b[2])**2)

h = 1.0 / num_segments
l0 = length(center_line(h), center_line(0))
a0 = l0/2
pv_rod = mir.ParticleVectors.RodVector('rod', mass, num_segments)
ic_rod = mir.InitialConditions.Rod(com_q_rod, center_line, torsion, a0)


# ellipsoid

axes = tuple(args.axes)
com_q_ell = [[0.5 * domain[0],
              0.5 * domain[1],
              0.5 * domain[2] + axes[2],
              1., 0, 0, 0]]
coords = np.loadtxt(args.coords).tolist()

if args.vis:
    import trimesh
    ell = trimesh.creation.icosphere(subdivisions=2, radius = 1.0)
    for i in range(3):
        ell.vertices[:,i] *= axes[i]
    mesh = mir.ParticleVectors.Mesh(ell.vertices.tolist(), ell.faces.tolist())
    pv_ell = mir.ParticleVectors.RigidEllipsoidVector('ellipsoid', mass, object_size=len(coords), semi_axes=axes, mesh=mesh)
else:
    pv_ell = mir.ParticleVectors.RigidEllipsoidVector('ellipsoid', mass, object_size=len(coords), semi_axes=axes)

ic_ell = mir.InitialConditions.Rigid(com_q_ell, coords)
vv_ell = mir.Integrators.RigidVelocityVerlet("vv_ell")

u.registerParticleVector(pv_ell, ic_ell)
u.registerParticleVector(pv_rod, ic_rod)

u.registerIntegrator(vv_ell)
u.setIntegrator(vv_ell, pv_ell)

# interactions

prms = {
    "a0" : a0,
    "l0" : l0,
    "k_s_center" : 100.0,
    "k_s_frame"  : 100.0,
    "k_bending"  : (10.0, 0.0, 10.0),
    "k_twist"    : 10.0,
    "tau0"       : 0,
    "kappa0"     : (0., 0.)
}

int_rod = mir.Interactions.RodForces("rod_forces", **prms);
u.registerInteraction(int_rod)
u.setInteraction(int_rod, pv_rod, pv_rod)

anchor=(0.0, 0.0, -axes[2])
torque  = 0.1
k_bound = 100.0
int_bind = mir.Interactions.ObjRodBinding("binding", torque, anchor, k_bound);
u.registerInteraction(int_bind)
u.setInteraction(int_bind, pv_ell, pv_rod)

vv_rod = mir.Integrators.VelocityVerlet('vv_rod')
u.registerIntegrator(vv_rod)
u.setIntegrator(vv_rod, pv_rod)

if args.drag > 0.0:
    u.registerPlugins(mir.Plugins.createParticleDrag('rod_drag', pv_rod, args.drag))

if args.vis:
    dump_every = int (t_dump_every/dt)
    u.registerPlugins(mir.Plugins.createDumpParticles('rod_dump', pv_rod, dump_every, [], 'h5/rod_particles-'))
    u.registerPlugins(mir.Plugins.createDumpMesh("mesh_dump", pv_ell, dump_every, path="ply/"))


u.run(int (t_end / dt))

if pv_rod is not None:
    pos_rod = pv_rod.getCoordinates()
    pos_ell = pv_ell.getCoordinates()
    np.savetxt("pos.txt", np.vstack((pos_rod, pos_ell)))

del u

# nTEST: bindings.obj_rod.one
# cd bindings
# rm -rf h5 pos*txt
# f="pos.txt"
# rho=8.0; ax=2.0; ay=1.0; az=1.0
# cp ../../data/ellipsoid_coords_${rho}_${ax}_${ay}_${az}.txt $f
# mir.run --runargs "-n 2" ./obj_rod.py --axes $ax $ay $az --coords $f --vis
# cat pos.txt > pos.out.txt
