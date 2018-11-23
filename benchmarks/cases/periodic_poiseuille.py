#!/usr/bin/env python

import ymero as ymr
import argparse
import sys

parser = argparse.ArgumentParser(description='Run the simulation')

parser.add_argument('--debug-lvl', help='Debug level', type=int, default=3)
parser.add_argument('--niters', help='Number of steps to run', type=int, default=50000)

parser.add_argument('--f', help='Periodic force', type=float, default=0.01)

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

u = ymr.ymero(tuple(args.nranks), tuple(args.domain), debug_level=args.debug_lvl, log_filename='log')

if u.isMasterTask():
    report()

#====================================================================================
#====================================================================================

pv = ymr.ParticleVectors.ParticleVector('pv', mass = 1)
ic = ymr.InitialConditions.Uniform(density=args.rho)
u.registerParticleVector(pv=pv, ic=ic)
    
dpd = ymr.Interactions.DPD('dpd', 1.0, a=args.a, gamma=args.gamma, kbt=args.kbt, dt=args.dt, power=args.power)
u.registerInteraction(dpd)
u.setInteraction(dpd, pv, pv)

vv = ymr.Integrators.VelocityVerlet_withPeriodicForce('vv', dt=args.dt, force=args.f, direction='x')
u.registerIntegrator(vv)
u.setIntegrator(vv, pv)

u.registerPlugins(ymr.Plugins.createStats('stats', "stats.txt", every=1000))

if args.with_dumps:
    sampleEvery = 5
    dumpEvery   = 1000
    binSize     = (1., 1., 1.)

    field = ymr.Plugins.createDumpAverage('field', [pv], sampleEvery, dumpEvery, binSize, [("velocity", "vector_from_float8")], 'h5/solvent-')
    u.registerPlugins(field)

u.run(args.niters)


# hack: cleanup device because of profiling tools

if u.isComputeTask():

    del u
    del pv
    del ic
    del dpd
    del vv

    if args.with_dumps:
        del field
    
    ymr.destroyCudaContext()
