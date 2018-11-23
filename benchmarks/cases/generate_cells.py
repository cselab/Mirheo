#!/usr/bin/env python

import argparse
import numpy as np
import sys
import pickle
import ymero as ymr
from membrane_parameters import set_parameters, params2dict

def gen_ic(domain, cell_volume, hematocrit, extent=(7,7,3)):
    assert(0.0 < hematocrit and hematocrit < 0.7)
    
    norm_extent = np.array(extent) / ((extent[0]*extent[1]*extent[2])**(1/3.0))
    
    domain_vol = domain[0]*domain[1]*domain[2]
    ncells = domain_vol*hematocrit / cell_volume
    
    gap = domain_vol**(1/3.0) / (ncells**(1/3.0) + 1)
    
    nx, ny, nz = [ int(domain[i] / (gap*norm_extent[i])) for i in range(3) ]
    real_ht = nx*ny*nz * cell_volume / domain_vol
    h = [ domain[0]/nx, domain[1]/ny, domain[2]/nz ]
    
    com_q = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                com_q.append( [i*h[0], j*h[1], k*h[2],  1, 0, 0, 0] )
                    
    return real_ht, (nx, ny, nz), com_q
    

def get_rbc_params(ymr, gamma_in, eta_in, rho):
    prms = ymr.Interactions.MembraneParameters()
    set_parameters(prms, gamma_in, eta_in, rho)

    return prms

class Viscosity_getter:
    def __init__(self, folder, a, power):
        self.s = pickle.load(open(folder + 'visc_' + str(float(a)) + '_' + str(float(power)) + '_backup.pckl', 'rb'))
        
    def predict(self, gamma):
        return self.s(gamma)

#====================================================================================
#====================================================================================
    
parser = argparse.ArgumentParser(description='Generate cells with given hematocrit')

parser.add_argument('--debug-lvl', help='Debug level', type=int, default=3)
parser.add_argument('--resource-folder', help='Path to all the required files', type=str, default='./')

parser.add_argument('--domain', help='Domain size', type=float, nargs=3, default=[64,64,64])
parser.add_argument('--nranks', help='MPI ranks',   type=int,   nargs=3, default=[1,1,1])

parser.add_argument('--rho', help='Particle density', type=float, default=8)
parser.add_argument('--a', help='a', default=80, type=float)
parser.add_argument('--gamma', help='gamma', default=20, type=float)
parser.add_argument('--kbt', help='kbt', default=1.5, type=float)
parser.add_argument('--dt', help='Time step', default=0.0005, type=float)
parser.add_argument('--power', help='Kernel exponent', default=0.5, type=float)

parser.add_argument('--lbd', help='RBC to plasma viscosity ratio', default=5.0, type=float)

parser.add_argument('--final-time', help='Final time', type=float, default=20.0)

parser.add_argument('--ht',  help='Hematocrit level', default=0.4, type=float)
parser.add_argument('--vol', help='Volume of a single cell', default=94.0, type=float)

parser.add_argument('--dry-run', help="Don't run the simulation, just report the parameters", action='store_true')

args, unknown = parser.parse_known_args()

#====================================================================================
#====================================================================================

real_ht, ncells, rbcs_ic = gen_ic(args.domain, args.vol, args.ht)

niters = int(args.final_time / args.dt)

visc_getter = Viscosity_getter(args.resource_folder, args.a, args.power)
mu_outer = visc_getter.predict(args.gamma)
mu_inner = mu_outer * args.lbd
params = get_rbc_params(ymr, args.gamma * args.lbd, mu_inner, args.rho)
params.gammaC = 2.0

#====================================================================================
#====================================================================================

def report():
    print('Started with the following parameters: ' + str(args))
    if unknown is not None and len(unknown) > 0:
        print('Some arguments are not recognized and will be ignored: ' + str(unknown))
    print('Outer viscosity: %f, inner: %f' % (mu_outer, mu_inner))
    print('Generated %d cells %s, real hematocrit is %f' % (len(rbcs_ic), str(ncells), real_ht))
    print('Cell parameters: %s' % str(params2dict(params)))
    print('')
    sys.stdout.flush()


if args.dry_run:
    report()
    quit()

u = ymr.ymero(args.nranks, args.domain, debug_level=args.debug_lvl, log_filename='generate', restart_folder="generated/")

if u.isMasterTask():
    report()

#====================================================================================
#====================================================================================

# Interactions:
#   DPD
dpd = ymr.Interactions.DPD('dpd', rc=1.0, a=args.a, gamma=args.gamma, kbt=args.kbt, dt=args.dt, power=args.power)
u.registerInteraction(dpd)
#   Contact (LJ)
contact = ymr.Interactions.LJ('contact', rc=1.0, epsilon=10, sigma=1.0, object_aware=True, max_force=2000)
u.registerInteraction(contact)

#   Membrane
membrane_int = ymr.Interactions.MembraneForces('int_rbc', params, stressFree=False, grow_until=args.final_time*0.5)
u.registerInteraction(membrane_int)

# Integrator
vv = ymr.Integrators.VelocityVerlet('vv', args.dt)
u.registerIntegrator(vv)

# RBCs
mesh_rbc = ymr.ParticleVectors.MembraneMesh(args.resource_folder + 'rbc_mesh.off')
rbcs = ymr.ParticleVectors.MembraneVector('rbc', mass=1.0, mesh=mesh_rbc)
u.registerParticleVector(pv=rbcs, ic=ymr.InitialConditions.Membrane(rbcs_ic, global_scale=0.5), checkpoint_every = niters-5)

# Stitching things with each other
#   contact
u.setInteraction(contact, rbcs, rbcs)
#   membrane
u.setInteraction(membrane_int, rbcs, rbcs)

# Integration
u.setIntegrator(vv, rbcs)

#====================================================================================
#====================================================================================

u.registerPlugins(ymr.Plugins.createStats('stats', every=1000))


u.run(niters)
