#!/usr/bin/env python

import udevicex as udx
import argparse
import sys

parser = argparse.ArgumentParser(description='Run the simulation')

parser.add_argument('--debug-lvl', help='Debug level', type=int, default=3)
parser.add_argument('--niters', help='Number of steps to run', type=int, default=50000)

parser.add_argument('--f', help='Periodic force', type=float, default=0.005)

parser.add_argument('--rc', help='Cutoff radius', type=float, default=1.0)
parser.add_argument('--rho', help='Particle density', type=float, default=8)
parser.add_argument('--a', help='a', default=50, type=float)
parser.add_argument('--gamma', help='gamma', default=20, type=float)
parser.add_argument('--kbt', help='kbt', default=0.5, type=float)
parser.add_argument('--dt', help='Time step', default=0.001, type=float)
parser.add_argument('--power', help='Kernel exponent', default=0.5, type=float)

parser.add_argument('--domain', help='Domain size', type=float, nargs=3, default=[64,64,64])
parser.add_argument('--nranks', help='MPI ranks',   type=int,   nargs=3, default=[1,1,1])

parser.add_argument('--with-dumps', help='Enable data-dumps', action='store_true')

parser.add_argument('--dry-run', help="Don't run the simulation, just report the parameters", action='store_true')

args, unknown = parser.parse_known_args()

#====================================================================================
#====================================================================================

def report():
    print('Started with the following parameters: ' + str(args))
    if unknown is not None and len(unknown) > 0:
        print('Some arguments are not recognized and will be ignored: ' + str(unknown))
    print('')
    sys.stdout.flush()


if args.dry_run:
    report()
    quit()

domain = args.domain
rc     = args.rc
    
u = udx.udevicex(tuple(args.nranks), tuple(domain), debug_level=args.debug_lvl, log_filename='log')

if u.isMasterTask():
    report()

#====================================================================================
#====================================================================================

pv = udx.ParticleVectors.ParticleVector('pv', mass = 1)
ic = udx.InitialConditions.Uniform(density=args.rho)
u.registerParticleVector(pv=pv, ic=ic)
    
dpd = udx.Interactions.DPD('dpd', rc, args.a, args.gamma, args.kbt, args.dt, args.power)
u.registerInteraction(dpd)

vveq = udx.Integrators.VelocityVerlet('vveq', args.dt)
vv   = udx.Integrators.VelocityVerlet_withConstForce('vv', args.dt, force=[args.f, 0, 0])

lo = ( -domain[0],  -domain[1],           rc)
hi = (2*domain[0], 2*domain[1], domain[2]-rc)
wall = udx.Walls.Box("plates", low=lo, high=hi, inside=True)
u.registerWall(wall, 0)

frozen = u.makeFrozenWallParticles(pvName="plates", walls=[wall], interaction=dpd, integrator=vveq, density=args.rho)

u.registerIntegrator(vv)
u.setIntegrator(vv, pv)

u.setInteraction(dpd, pv, pv)
u.setInteraction(dpd, frozen, pv)

u.registerPlugins(udx.Plugins.createStats('stats', "stats.txt", every=1000))

if args.with_dumps:
    sampleEvery = 5
    dumpEvery   = 1000
    binSize     = (1., 1., 1.)

    field = udx.Plugins.createDumpAverage('field', [pv], sampleEvery, dumpEvery, binSize, [("velocity", "vector_from_float8")], 'h5/solvent-')
    u.registerPlugins(field)

u.run(args.niters)

