#!/usr/bin/env python

"""Test access to global simulation information through mirheo state."""

import numpy as np
import unittest
import mirheo as mir

class TestGlobalSimulationInfo(unittest.TestCase):

    def test_state(self):
        domain = np.array([10.0, 15.0, 20.0])
        u = mir.Mirheo(nranks=(1, 1, 1), domain=domain, no_splash=True)

        # 1. test the time states
        state = u.getState()
        self.assertEqual(state.current_time, 0)

        dt = 1
        num_steps = 5
        u.run(num_steps, dt)

        self.assertEqual(state.current_step, num_steps)
        self.assertEqual(state.current_time, num_steps * dt)

        # 2. test domain info state
        domain_info = u.getState().domain_info

        pos_center_global = domain / 2
        pos_center_local = np.zeros_like(domain)

        np.testing.assert_array_equal(tuple(domain_info.global_size), domain)

        np.testing.assert_array_equal(tuple(domain_info.local_to_global(pos_center_local)),
                                      pos_center_global)

        np.testing.assert_array_equal(tuple(domain_info.global_to_local(pos_center_global)),
                                      pos_center_local)



if __name__ == '__main__':
    import sys
    out = unittest.main(argv=[sys.argv[0], '-q'], exit=False)
    if out.result.errors or out.result.failures:
        print("Failed!")


# TEST: bindings.mirheo_state
# cd bindings
# mir.run -n 1 ./mirheo_state.py > mirheo_state.out.txt
