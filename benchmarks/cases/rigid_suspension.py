#!/usr/bin/env python

import ymero as ymr
import ymero.tools
import argparse
import generate_rigids
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--nranks', type=int, nargs=3, default=(1, 1, 1))
parser.add_argument('--domain', type=float, nargs=3, default=[64, 64, 64])
parser.add_argument('--coords', dest='coords', type=str, required=True)
parser.add_argument('--withMesh', action='store_true', default=False)
parser.add_argument('--tend', type=float, default=50.0)
parser.add_argument('--tDumpEvery', type=float, default=0.1)
parser.add_argument('--f', type=float, default=0.01)
args = parser.parse_args()

rigids_numdensity = 0.002

rc = 1.0

ranks  = args.nranks
domain = args.domain

tend = args.tend
tDumpEvery = args.tDumpEvery
dt = 0.001
density = 10
mass = 1.0

kBT = 1e-1
adpd = 20.0
gdpd = 20.0
dpdPower = 0.25

u = ymr.ymero(ranks, domain, debug_level=3, log_filename='log')

pvSolvent = ymr.ParticleVectors.ParticleVector('solvent', mass)
icSolvent = ymr.InitialConditions.Uniform(density)

dpd = ymr.Interactions.DPD('dpd', rc, adpd, gdpd, kBT, dt, dpdPower)
cnt = ymr.Interactions.LJ('contact', rc=1.0, epsilon=1.0, sigma=0.9, object_aware=True, max_force=750)

vv = ymr.Integrators.VelocityVerlet_withPeriodicForce('vv', dt, args.f, direction='x')

u.registerParticleVector(pvSolvent, icSolvent)
u.registerInteraction(dpd)
u.registerInteraction(cnt)

totVolume = domain[0] * domain[1] * domain[2]
numObjects  = int(rigids_numdensity * totVolume)

axes = (3, 1, 1)

coords = np.loadtxt(args.coords).tolist()

if args.withMesh:
    import trimesh
    ell = trimesh.creation.icosphere(subdivisions=2, radius = 1.0)
    for i in range(3):
        ell.vertices[:,i] *= axes[i]
    mesh = ymr.ParticleVectors.Mesh(ell.vertices.tolist(), ell.faces.tolist())
    pvEllipsoids = ymr.ParticleVectors.RigidEllipsoidVector('ellipsoids', mass=1, object_size=len(coords), semi_axes=axes, mesh=mesh)
else:
    pvEllipsoids = ymr.ParticleVectors.RigidEllipsoidVector('ellipsoids', mass=1, object_size=len(coords), semi_axes=axes)

q = ymero.tools.eulerToQuaternion(0., np.pi/2, 0.)
(n, com_q) = generate_rigids.ellipsoids(domain, axes, rigids_numdensity, q)
icEllipsoids = ymr.InitialConditions.Rigid(com_q=com_q, coords=coords)
vvEllipsoids = ymr.Integrators.RigidVelocityVerlet("vvRigids", dt)

u.registerParticleVector(pvEllipsoids, icEllipsoids)

belongingChecker = ymr.BelongingCheckers.Ellipsoid("ellipsoidChecker")

u.registerObjectBelongingChecker(belongingChecker, pvEllipsoids)
u.applyObjectBelongingChecker(belongingChecker, pv=pvSolvent, correct_every=0, inside="none")


u.setInteraction(dpd, pvSolvent,    pvSolvent)
u.setInteraction(dpd, pvEllipsoids, pvSolvent)

u.registerIntegrator(vv)
u.registerIntegrator(vvEllipsoids)
u.setIntegrator(vv, pvSolvent)
u.setIntegrator(vvEllipsoids, pvEllipsoids)

bb = ymr.Bouncers.Ellipsoid("bounceEllipsoid")
u.registerBouncer(bb)
u.setBouncer(bb, pvEllipsoids, pvSolvent)


dumpEvery = int(tDumpEvery / dt)

if args.withMesh:
    mdump = ymr.Plugins.createDumpMesh("meshDump", pvEllipsoids, dumpEvery, path="ply/")
    u.registerPlugins(mdump)


stats = ymr.Plugins.createStats('stats', "stats.txt", dumpEvery)
u.registerPlugins(stats)

if 0:
    sampleEvery = 2
    binSize     = (1., 1., 1.)
    
    field = ymr.Plugins.createDumpAverage('field', [pvSolvent, pvEllipsoids], sampleEvery, dumpEvery, binSize, [("velocity", "vector_from_float8")], 'h5/field-')
    u.registerPlugins(field)

niter = int(tend/dt)
u.run(niter)

