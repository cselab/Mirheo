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
        r = cp.asarray(pv.local['positions'])
        v = cp.asarray(pv.local['velocities'])
        f = cp.asarray(pv.local['__forces'])
        self.assertEqual(r.shape, (N, 4))  # The 4th component is kept.
        self.assertEqual(v.shape, (N, 4))  # The 4th component is kept.
        self.assertEqual(f.shape, (N, 3))  # The `int` element is stripped away.

        # Test implicit access.
        self.assertEqual(cp.asarray(pv.r).shape, (N, 3))
        self.assertEqual(cp.asarray(pv.v).shape, (N, 3))
        self.assertEqual(cp.asarray(pv.f).shape, (N, 3))
        self.assertEqual(cp.asarray(pv.local.r).shape, (N, 3))
        self.assertEqual(cp.asarray(pv.local.v).shape, (N, 3))
        self.assertEqual(cp.asarray(pv.local.f).shape, (N, 3))
        cp.asarray(pv.r)[:] = 10.0 * cp.arange(N * 3).reshape((N, 3))
        cp.asarray(pv.v)[:] = 20.0 * cp.arange(N * 3).reshape((N, 3))
        cp.asarray(pv.f)[:] = 30.0 * cp.arange(N * 3).reshape((N, 3))
        cp.testing.assert_array_equal(r[:, :3], pv.r)
        cp.testing.assert_array_equal(v[:, :3], pv.v)
        cp.testing.assert_array_equal(f[:, :3], pv.f)

        # Test updating velocities and reading positions.
        r = cp.asarray(pv.r)
        v = cp.asarray(pv.v)
        f = cp.asarray(pv.f)
        r[:] = 0.0  # Reset positions to the middle of the domain.
        v[:] = cp.array([
            [10, 20, 30],
            [11, 21, 31],
            [12, 22, 32],
            [13, 23, 33],
            [14, 24, 34],
        ])
        f[:] = 0.0  # Reset forces.
        u.run(1, dt=10.0)
        cp.testing.assert_array_equal(pv.r, cp.array(r0 - mid) + 10.0 * v)

        # Test forces.
        r = cp.asarray(pv.r)
        v = cp.asarray(pv.v)
        r[:] = 0.0  # Reset positions to the middle of the domain.
        v[:] = 0.0  # Reset velocities.
        u.registerPlugins(mir.Plugins.createAddForce('add_force', pv, (1.0, 2.0, 3.0)))
        u.run(1, dt=10.0)
        cp.testing.assert_array_equal(pv.r, [[100.0, 200.0, 300.0]] * N)
        cp.testing.assert_array_equal(pv.v, [[10.0, 20.0, 30.0]] * N)
        cp.testing.assert_array_equal(pv.f, [[1.0, 2.0, 3.0]] * N)


if __name__ == '__main__':
    import sys
    out = unittest.main(argv=[sys.argv[0], '-q'], exit=False)
    if out.result.errors or out.result.failures:
        print("Failed!")


# TEST: bindings.particle_vector
# cd bindings
# mir.run -n 1 ./particle_vector.py > particle_vector.out.txt
