#!/usr/bin/env python

"""Test ParticleVector channel __cuda_array_interface__ with cupy."""

import unittest
import numpy as np
import mirheo as mir

try:
    import cupy as cp
except ImportError:
    cp = None


class TestParticleVector(unittest.TestCase):
    @unittest.skipIf(cp is None, "Skipping cupy tests, cupy not found.")
    def test_cupy_access(self):
        N = 5
        domain = np.array([10.0, 10.0, 10.0])
        mid = 0.5 * domain
        r0 = np.tile(mid, (N, 1))  # Particles start at the middle of the domain.
        v0 = np.zeros((N, 3))

        # Mirheo setup with 1 PV and a VV integrator.
        u = mir.Mirheo(nranks=(1, 1, 1), domain=domain, no_splash=True)
        pv = mir.ParticleVectors.ParticleVector('pv', mass=1.0)
        u.registerParticleVector(pv, mir.InitialConditions.FromArray(r0, v0))
        vv = mir.Integrators.VelocityVerlet('vv')
        u.registerIntegrator(vv)
        u.setIntegrator(vv, pv)

        # Test explicit access.
        r = cp.asarray(pv.local.per_particle['positions'])
        v = cp.asarray(pv.local.per_particle['velocities'])
        f = cp.asarray(pv.local.per_particle['__forces'])
        self.assertEqual(r.shape, (N, 4))  # The 4th component is kept.
        self.assertEqual(v.shape, (N, 4))  # The 4th component is kept.
        self.assertEqual(f.shape, (N, 3))  # The `int` element is stripped away.

        # Test updating velocities and reading positions.
        r = cp.asarray(pv.local.per_particle['positions'])
        v = cp.asarray(pv.local.per_particle['velocities'])
        f = cp.asarray(pv.local.per_particle['__forces'])
        r[:,:3] = 0.0  # Reset positions to the middle of the domain.
        v[:,:3] = cp.array([
            [10, 20, 30],
            [11, 21, 31],
            [12, 22, 32],
            [13, 23, 33],
            [14, 24, 34],
        ])
        f[:] = 0.0  # Reset forces.
        u.run(1, dt=10.0)
        r = cp.asarray(pv.local.per_particle['positions'])
        cp.testing.assert_array_equal(r[:,:3], cp.array(r0 - mid) + 10.0 * v[:,:3])

        # Test forces.
        r = cp.asarray(pv.local.per_particle['positions'])
        v = cp.asarray(pv.local.per_particle['velocities'])
        r[:,:3] = 0.0  # Reset positions to the middle of the domain.
        v[:,:3] = 0.0  # Reset velocities.
        u.registerPlugins(mir.Plugins.createAddForce('add_force', pv, (1.0, 2.0, 3.0)))
        u.run(1, dt=10.0)
        r = cp.asarray(pv.local.per_particle['positions'])
        v = cp.asarray(pv.local.per_particle['velocities'])
        f = cp.asarray(pv.local.per_particle['__forces'])
        cp.testing.assert_array_equal(r[:,:3], [[100.0, 200.0, 300.0]] * N)
        cp.testing.assert_array_equal(v[:,:3], [[10.0, 20.0, 30.0]] * N)
        cp.testing.assert_array_equal(f, [[1.0, 2.0, 3.0]] * N)


if __name__ == '__main__':
    import sys
    out = unittest.main(argv=[sys.argv[0], '-q'], exit=False)
    if out.result.errors or out.result.failures:
        print("Failed!")


# TEST: bindings.particle_vector
# cd bindings
# mir.run -n 1 ./particle_vector.py > particle_vector.out.txt
