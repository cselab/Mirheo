#!/usr/bin/env python

"""Test SW2 forces."""

import mirheo as mir
import numpy
from autograd import grad
import autograd.numpy as np
import sys
import h5py
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

#parameters
rc = 2.0
epsilon = 1.234
A       = 1.234
B       = 1.234
sigma   = 1.234

#Stillinger-Weber 3Body Potential
def sw2_pot(r_i, r_j):
    r_ij = r_i - r_j
   
    r = np.sqrt(np.dot(r_ij,r_ij))

    return A*epsilon*(B*((sigma/r)**4) - 1.0)*np.exp(sigma/(r-rc))

def sw2_for(r_i, r_j):  #this is a sanity check
    r_ij = r_i - r_j
    r2 = np.dot(r_ij,r_ij)
    r  = np.sqrt(r2)
    
    rs2 = (sigma*sigma) / r2
    B_rs4 = B * rs2 * rs2
    r_rc = r - rc
    exp = np.exp(sigma / r_rc)
    A_eps_exp = A * epsilon * exp
    phi = (sigma * (B_rs4 - 1.0))/(r_rc*r_rc*r) + (4.0*B_rs4)/r2

    return phi * A_eps_exp * r_ij


def withinCutOff(r,s):
    rs = r - s
    drs2 = np.dot(rs,rs)
    if drs2 > rc*rc:
        return False
    else:
        return True


def brute_force(particle):
    n = particle.shape[0]
    forces = np.zeros((n,3))
    grad_sw2_pot = grad(sw2_pot)
    #grad_sw2_pot = sw2_for

    #2body
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if withinCutOff(particle[i], particle[j]):
                forces[i,:] += -grad_sw2_pot(particle[i], particle[j])
    return forces


def main():
    n = 32

    #shape of particles = 2.0*np.random.rand(n, 3) + np.full((n, 3), 2.0)
    #if rank == 0: numpy.savetxt('particles-sw3.csv', particles, fmt='%f', delimiter=',')
    particles = np.loadtxt("particles-sw3.csv", delimiter=',')
    vel = np.zeros((n,3))
    
    ranks = (2, 2, 2)   #force Halo intractions
    domain = (8.0, 8.0, 8.0)

    u = mir.Mirheo(ranks, domain, debug_level=3, log_filename='log', no_splash=True)

    pv = mir.ParticleVectors.ParticleVector('pv', mass=1.0)
    ic = mir.InitialConditions.FromArray(pos=particles, vel=vel)
    u.registerParticleVector(pv, ic)

    sw2 = mir.Interactions.Pairwise('sw', rc, kind="SW", epsilon=epsilon, sigma=sigma, A=A, B=B)
    u.registerInteraction(sw2)
    u.setInteraction(sw2, pv, pv)

    vv = mir.Integrators.VelocityVerlet('vv')
    u.registerIntegrator(vv)
    u.setIntegrator(vv, pv)


    dump_every = 1
    u.registerPlugins(mir.Plugins.createForceSaver('forces', pv))
    u.registerPlugins(mir.Plugins.createDumpParticles('force_dump', pv, dump_every, ["forces"], 'h5/sw2-'))

    u.run(2, dt=0.0001)


    if rank == 0:
        f = h5py.File('h5/sw2-00001.h5', 'r')
        mirheo = f['forces'][()]
        brute = brute_force(particles)
        try:    
            #if necessary sort them
            numpy.testing.assert_allclose(np.sort(mirheo, axis=0), np.sort(brute, axis=0), rtol=1e-4) #1e-10 for double precision
        except:
            print("particles:\n", particles)
            print("mirheo positions:\n", f['position'][()])
            print("mirheo force:\n", mirheo)
            print("brute force:\n", brute)
            raise

        print("OK")


main()

# TEST: triplewise.sw2
# cd triplewise
# mir.run --runargs "-n 16" ./sw2.py > sw2.out.txt
