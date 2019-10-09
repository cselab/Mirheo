import mirheo as mir
import numpy as np
import trimesh

def recenter(coords, com):
    coords = [[r[0]-com[0], r[1]-com[1], r[2]-com[2]] for r in coords]
    return coords

dt = 0.001
rc = 1.0
mass = 1.0
density = 10
niter = 1000

# the triangle mesh used to create the object
# here we load the file using trimesh for convenience
m = trimesh.load("sphere_mesh.off");

# trimesh is able to compute the inertia tensor
# we assume it is diagonal here
inertia = [row[i] for i, row in enumerate(m.moment_inertia)]

ranks  = (1, 1, 1)
domain = (16, 16, 16)
    
u = mir.Mirheo(ranks, domain, dt, debug_level=3, log_filename='log')

dpd = mir.Interactions.Pairwise('dpd', rc, kind="DPD", a=10.0, gamma=10.0, kBT=0.5, power=0.5)
vv  = mir.Integrators.VelocityVerlet('vv')

# we create here a fake rigid object in the center of the domain with only 2 particles
# those particles will be used to compute the extents in the object belonging, so they
# must be located in the to corners of the bounding box of the object  
# this is only to be able to make use of the belonging checker

bb_hi = m.vertices.max(axis=0).tolist()
bb_lo = m.vertices.min(axis=0).tolist()

coords = [bb_lo, bb_hi]
com_q = [[0.5 * domain[0], 0.5 * domain[1], 0.5 * domain[2],   1, 0, 0, 0]]

mesh = mir.ParticleVectors.Mesh(m.vertices.tolist(), m.faces.tolist())
fake_ov = mir.ParticleVectors.RigidObjectVector('fake_ov', mass, inertia, len(coords), mesh)
fake_ic = mir.InitialConditions.Rigid(com_q, coords)

belonging_checker = mir.BelongingCheckers.Mesh("mesh_checker")

# similarly to wall creation, we freeze particles inside a rigid object
pv_rigid = u.makeFrozenRigidParticles(belonging_checker, fake_ov, fake_ic, [dpd], vv, density, niter)

if u.isMasterTask():
    coords = pv_rigid.getCoordinates()
    coords = recenter(coords, com_q[0])
    np.savetxt("rigid_coords.txt", coords)

    
