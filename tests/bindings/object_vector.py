#!/usr/bin/env python

"""Test ObjectVector channel __cuda_array_interface__ with cupy."""

import unittest
import numpy as np
import mirheo as mir

try:
    import cupy as cp
except ImportError:
    cp = None


class TestObjectVector(unittest.TestCase):
    @unittest.skipIf(cp is None, "Skipping cupy tests, cupy not found.")
    def test_cupy_access(self):
        # radius of the sphere
        R = 1
        # coordinates of the particles composing each sphere
        coords = [[-R, -R, -R],
                  [-R, -R, +R],
                  [-R, +R, -R],
                  [-R, +R, +R],
                  [+R, -R, -R],
                  [+R, -R, +R],
                  [+R, +R, -R],
                  [+R, +R, +R]]

        N = 5
        domain = np.array([10.0, 10.0, 10.0])
        mid = 0.5 * domain
        r0 = np.tile(mid, (N, 1))  # Objects start at the middle of the domain.
        com_q = [mid.tolist() + [1, 0, 0, 0]] * N


        # Mirheo setup with 1 OV and a VV integrator.
        u = mir.Mirheo(nranks=(1, 1, 1), domain=domain, no_splash=True)
        ov = mir.ParticleVectors.RigidEllipsoidVector('ov', mass=1.0, object_size=len(coords), semi_axes=(R,R,R))
        u.registerParticleVector(ov, mir.InitialConditions.Rigid(com_q, coords))
        vv = mir.Integrators.RigidVelocityVerlet('vv')
        u.registerIntegrator(vv)
        u.setIntegrator(vv, ov)

        # Test explicit access of object channels.
        com_extents = cp.asarray(ov.local.per_object['com_extents'])
        self.assertEqual(com_extents.shape, (N, 9)) # com, low, high -> 3 + 3 + 3

        u.run(1, dt=1) # nothing should move; just used to trigger com_extents computation

        cp.testing.assert_array_equal(com_extents, [[0, 0, 0, -R, -R, -R, R, R, R]] * N)




if __name__ == '__main__':
    import sys
    out = unittest.main(argv=[sys.argv[0], '-q'], exit=False)
    if out.result.errors or out.result.failures:
        print("Failed!")


# TEST: bindings.object_vector
# cd bindings
# mir.run -n 1 ./object_vector.py > object_vector.out.txt
