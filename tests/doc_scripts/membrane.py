#!/usr/bin/env python

import mirheo as mir

dt = 0.001

ranks  = (1, 1, 1)
domain = (12, 12, 12)

u = mir.Mirheo(ranks, domain, debug_level=3, log_filename='log')

# we need to first create a mesh before initializing the membrane vector
mesh_rbc = mir.ParticleVectors.MembraneMesh("rbc_mesh.off")

# create a MembraneVector with the given mesh
pv_rbc   = mir.ParticleVectors.MembraneVector("rbc", mass=1.0, mesh=mesh_rbc)
# place initial membrane
# we need a position pos and an orientation described by a quaternion q
# here we create only one membrane at the center of the domain
pos_q = [0.5*domain[0], 0.5*domain[1], 0.5*domain[2], # position
         1.0, 0.0, 0.0, 0.0]                          # quaternion
ic_rbc   = mir.InitialConditions.Membrane([pos_q])
u.registerParticleVector(pv_rbc, ic_rbc)

# next we store the parameters in a dictionary
prms_rbc = {
    "x0"     : 0.457,
    "ka_tot" : 4900.0,
    "kv_tot" : 7500.0,
    "ka"     : 5000,
    "ks"     : 0.0444 / 0.000906667,
    "mpow"   : 2.0,
    "gammaC" : 52.0,
    "kBT"    : 0.0,
    "tot_area"   : 62.2242,
    "tot_volume" : 26.6649,
    "kb"     : 44.4444,
    "theta"  : 6.97
}

# now we create the internal interaction
# here we take the WLC model for shear forces and Kantor model for bending forces.
# the parameters are passed in a kwargs style
int_rbc = mir.Interactions.MembraneForces("int_rbc", "wlc", "Kantor", **prms_rbc)

# then we proceed as usual to make th membrane particles evolve in time
vv = mir.Integrators.VelocityVerlet('vv')
u.registerIntegrator(vv)
u.setIntegrator(vv, pv_rbc)
u.registerInteraction(int_rbc)
u.setInteraction(int_rbc, pv_rbc, pv_rbc)

# dump the mesh every 50 steps in ply format to the folder 'ply/'
u.registerPlugins(mir.Plugins.createDumpMesh("mesh_dump", pv_rbc, 50, "ply/"))

u.run(5002, dt=dt)
