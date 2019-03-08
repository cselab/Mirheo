#!/usr/bin/env python

# first we need to import the module
import ymero as ymr

dt = 0.001                  # simulation time step
ranks  = (1, 1, 1)          # number of ranks in x, y, z directions
domain = (32.0, 16.0, 16.0) # domain size in x, y, z directions

# create the coordinator
u = ymr.ymero(ranks, domain, dt, debug_level=3, log_filename='log')

u.run(100) # Run 100 time-steps
