#!/usr/bin/env python

import udevicex as udx
import argparse
import sys
from membrane_parameters import set_parameters, params2dict
import pickle
from scipy.optimize import fsolve
import math


class Viscosity_getter:
    def __init__(self, folder, a, power):
        self.s = pickle.load(open(folder + 'visc_' + str(float(a)) + '_' + str(float(power)) + '_backup.pckl', 'rb'))
        
    def predict(self, gamma):
        return self.s(gamma)

def get_rbc_params(udx, gamma_in, eta_in, rho):
    prms = udx.Interactions.MembraneParameters()
    set_parameters(prms, gamma_in, eta_in, rho)

    return prms

def get_fsi_gamma(eta, power, rho, rc=1.0):
    return 0.15e2 / 0.83e2 * eta * (8 * power ** 5 + 60 * power ** 4 + 170 * power ** 3 + 225 * power ** 2 + 137 * power + 30) / math.pi / rc ** 5 / rho

#====================================================================================
#====================================================================================   

parser = argparse.ArgumentParser(description='Run the simulation')

parser.add_argument('--debug-lvl', help='Debug level', type=int, default=3)
parser.add_argument('--resource-folder', help='Path to all the required files', type=str, default='./')
parser.add_argument('--niters', help='Number of steps to run', type=int, default=50000)

parser.add_argument('--f', help='Periodic force', type=float, default=0.05)

parser.add_argument('--rho', help='Particle density', type=float, default=8)
parser.add_argument('--a', help='a', default=80, type=float)
parser.add_argument('--gamma', help='gamma', default=20, type=float)
parser.add_argument('--kbt', help='kbt', default=0.5, type=float)
parser.add_argument('--dt', help='Time step', default=0.0005, type=float)
parser.add_argument('--power', help='Kernel exponent', default=0.5, type=float)

parser.add_argument('--lbd', help='RBC to plasma viscosity ratio', default=5.0, type=float)

parser.add_argument('--domain', help='Domain size', type=float, nargs=3, default=[64,64,64])
parser.add_argument('--nranks', help='MPI ranks',   type=int,   nargs=3, default=[1,1,1])

parser.add_argument('--nsubsteps', help='Number of substeps in membrane integration', default=1, type=int)

parser.add_argument('--with-dumps', help='Enable data-dumps', action='store_true')

parser.add_argument('--dry-run', help="Don't run the simulation, just report the parameters", action='store_true')

args, unknown = parser.parse_known_args()

#====================================================================================
#====================================================================================

visc_getter = Viscosity_getter(args.resource_folder, args.a, args.power)

mu_outer = visc_getter.predict(args.gamma)
mu_inner = mu_outer * args.lbd

args.outer_gamma = args.gamma
args.inner_gamma = fsolve(lambda g : visc_getter.predict(g) - mu_inner, mu_inner)[0]
args.outer_fsi_gamma = get_fsi_gamma(mu_outer, args.power, args.rho)
args.inner_fsi_gamma = get_fsi_gamma(mu_inner, args.power, args.rho)

# just in case
args.gamma = None

# RBC parameters
rbc_params = get_rbc_params(udx, args.inner_gamma, mu_inner, args.rho)

#====================================================================================
#====================================================================================

def report():
    print('Started with the following parameters: ' + str(args))
    if unknown is not None and len(unknown) > 0:
        print('Some arguments are not recognized and will be ignored: ' + str(unknown))
    print('Cell parameters: %s' % str(params2dict(rbc_params)))
    print('')
    sys.stdout.flush()


if args.dry_run:
    report()
    quit()

u = udx.udevicex(tuple(args.nranks), tuple(args.domain), debug_level=args.debug_lvl, log_filename='log')

if u.isMasterTask():
    report()

#====================================================================================
#====================================================================================

outer = udx.ParticleVectors.ParticleVector('outer', mass = 1.0)
ic = udx.InitialConditions.Uniform(density=args.rho)
u.registerParticleVector(pv=outer, ic=ic)

# Interactions:
#   DPD
dpd = udx.Interactions.DPD('dpd', rc=1.0, a=args.a, gamma=args.outer_gamma, kbt=args.kbt, dt=args.dt, power=args.power)
u.registerInteraction(dpd)
#   Contact (LJ)
contact = udx.Interactions.LJ('contact', rc=1.0, epsilon=1.0, sigma=0.9, object_aware=True, max_force=750)
u.registerInteraction(contact)
#   Membrane
membrane_int = udx.Interactions.MembraneForces('int_rbc', rbc_params, stressFree=True)
u.registerInteraction(membrane_int)

# Integrator
vv = udx.Integrators.VelocityVerlet_withPeriodicForce('vv', dt=args.dt, force=args.f, direction='x')
u.registerIntegrator(vv)

if args.nsubsteps > 1:
    subvv = udx.Integrators.SubStepMembrane('subvv', args.dt, args.nsubsteps, membrane_int)
    u.registerIntegrator(subvv)


# RBCs
mesh_rbc = udx.ParticleVectors.MembraneMesh(args.resource_folder + 'rbc_mesh.off')
rbcs = udx.ParticleVectors.MembraneVector('rbc', mass=1.0, mesh=mesh_rbc)
u.registerParticleVector(pv=rbcs, ic=udx.InitialConditions.Restart('generated/'))

checker = udx.BelongingCheckers.Mesh('checker')
u.registerObjectBelongingChecker(checker, rbcs)
inner = u.applyObjectBelongingChecker(checker, outer, inside='inner', correct_every=0)

# Bouncer
bouncer = udx.Bouncers.Mesh('bouncer')
u.registerBouncer(bouncer)


# Stitching things with each other
#   dpd
if u.isComputeTask():
    dpd.setSpecificPair(rbcs,  outer,  a=0, gamma=args.outer_fsi_gamma)
    dpd.setSpecificPair(rbcs,  inner,  a=0, gamma=args.inner_fsi_gamma)
    dpd.setSpecificPair(inner, outer,  gamma=0, kbt=0)
    dpd.setSpecificPair(inner, inner,  gamma=args.inner_gamma)

u.setInteraction(dpd, outer,  outer)
u.setInteraction(dpd, inner, outer)
u.setInteraction(dpd, outer, rbcs)
    
u.setInteraction(dpd, inner, inner)
u.setInteraction(dpd, inner, rbcs)

#   contact
u.setInteraction(contact, rbcs, rbcs)

#   membrane
# don't set it if we're using substep
if args.nsubsteps == 1:
    u.setInteraction(membrane_int, rbcs, rbcs)
    
# Integration
for pv in [inner, outer]:
    u.setIntegrator(vv, pv)
    
if args.nsubsteps == 1:
    u.setIntegrator(vv, rbcs)
else:
    u.setIntegrator(subvv, rbcs)

# Membrane bounce
u.setBouncer(bouncer, rbcs, inner)
u.setBouncer(bouncer, rbcs, outer)

#====================================================================================
#====================================================================================

statsEvery=20
u.registerPlugins(udx.Plugins.createStats('stats', "stats.txt", statsEvery))

if args.with_dumps:
    sample_every = 5
    dump_every   = 1000
    bin_size     = (1., 1., 1.)

    field = udx.Plugins.createDumpAverage('field', [outer, inner], sample_every, dump_every, bin_size, [("velocity", "vector_from_float8")], 'h5/solvent-')
    u.registerPlugins(field)
    
    u.registerPlugins(udx.Plugins.createDumpMesh('mesh', ov=rbcs, dump_every=dump_every, path='ply/'))

u.run(args.niters)
