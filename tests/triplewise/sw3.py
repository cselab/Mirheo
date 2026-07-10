#!/usr/bin/env python

"""Test triplewise forces, test halo forces in triplewise & test sw3 working correctly"""

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
lambda_ = 0.1234
epsilon = 0.2234
theta   = 0.3234
gamma   = 0.4234
sigma   = 0.5234

#Stillinger-Weber 3Body Potential
def sw3_pot(r_i, r_j, r_k):
    r_ij_vec = r_i - r_j
    r_jk_vec = r_j - r_k
    r_ki_vec = r_k - r_i

    r_ij = np.sqrt(np.dot(r_ij_vec, r_ij_vec))
    r_jk = np.sqrt(np.dot(r_jk_vec, r_jk_vec))
    r_ki = np.sqrt(np.dot(r_ki_vec, r_ki_vec))

    r_ij_hat = r_ij_vec/r_ij
    r_jk_hat = r_jk_vec/r_jk
    r_ki_hat = r_ki_vec/r_ki

    cos_theta_jik = -np.dot(r_ij_hat, r_ki_hat)
    cos_theta_ijk = -np.dot(r_ij_hat, r_jk_hat)
    cos_theta_ikj = -np.dot(r_ki_hat, r_jk_hat)

    h_jik = lambda_*epsilon*(cos_theta_jik - np.cos(theta))**2 *np.exp(gamma*sigma/(r_ij-rc) + gamma*sigma/(r_ki-rc))
    h_ijk = lambda_*epsilon*(cos_theta_ijk - np.cos(theta))**2 *np.exp(gamma*sigma/(r_ij-rc) + gamma*sigma/(r_jk-rc))
    h_ikj = lambda_*epsilon*(cos_theta_ikj - np.cos(theta))**2 *np.exp(gamma*sigma/(r_ki-rc) + gamma*sigma/(r_jk-rc))

    if(r_ij >= rc):
        if(r_jk < rc and r_ki < rc):
            return h_ikj
        else:                           
            return (0.0, 0.0, 0.0)        
    elif(r_jk >= rc and r_ki >= rc):        
        return (0.0, 0.0, 0.0)
    else:                                   
        if(r_jk < rc and r_ki < rc):  
            return h_jik + h_ijk + h_ikj
        elif(r_jk < rc):
            return h_ijk
        else:                              
            return h_jik


def withinCutOff(r,s):
    rs = r - s
    drs2 = np.dot(rs,rs)
    if drs2 > rc*rc:
        return False
    else:
        return True


def brute_force(particle):
    sw3_grad = grad(sw3_pot)
    n = particle.shape[0]
    forces = np.zeros((n,3))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            for k in range(j+1,n):
                if i == k:
                    continue
                interact01 = withinCutOff(particle[i], particle[j])
                interact12 = withinCutOff(particle[j], particle[k])
                interact20 = withinCutOff(particle[k], particle[i])
                if interact01 and interact12 or interact12 and interact20 or interact20 and interact01:
                    forces[i,:] += -sw3_grad(particle[i],particle[j],particle[k])
                
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

    sw3 = mir.Interactions.Triplewise('interaction', rc=rc, kind='SW', lambda_=lambda_, epsilon=epsilon, theta=theta, gamma=gamma, sigma=sigma)
    u.registerInteraction(sw3)
    u.setInteraction(sw3, pv, pv, pv)

    vv = mir.Integrators.VelocityVerlet('vv')
    u.registerIntegrator(vv)
    u.setIntegrator(vv, pv)


    dump_every = 1
    u.registerPlugins(mir.Plugins.createForceSaver('forces', pv))
    u.registerPlugins(mir.Plugins.createDumpParticles('force_dump', pv, dump_every, ["forces"], 'h5/sw3-'))

    u.run(2, dt=0.0001)


    if rank == 0:
        f = h5py.File('h5/sw3-00001.h5', 'r')
        mirheo = f['forces'][()]
        brute = brute_force(particles)
        try:    
            #if necessary sort them
            numpy.testing.assert_allclose(np.sort(mirheo, axis=0), np.sort(brute, axis=0), rtol=1e-5) #1e-10 for double precision
        except:
            print("particles:\n", particles)
            print("mirheo positions:\n", f['position'][()])
            print("mirheo force:\n", mirheo)
            print("brute force:\n", brute)
            raise

        print("OK")


main()

# TEST: triplewise.sw3
# cd triplewise
# mir.run --runargs "-n 16" ./sw3.py > sw3.out.txt
