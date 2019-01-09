# YMeRo

[![Documentation Status](https://readthedocs.org/projects/ymero/badge/?version=latest)](https://ymero.readthedocs.io/en/latest/?badge=latest)

Computational Microfluidics

YMeRo is a GPU high-performance and high-throughput code aimed at simulation of flows at milli- and microscales.
The code uses Dissipative Particle Dynamics method to describe the liquid and its interaction with cells and other bodies.

For more information, please refer to the online documentation: http://ymero.readthedocs.io/


## Changelog

### v0.9.1

* add sphere initial condition
* add plugin to compute total virial pressure from stresses per particle for a given pv

### v0.9.0

Add common YmrState object.
This changes the interface only slightly due to the python wrapping:
* the time step is now passed to the coordinator constructor
* the time step is not passed to any other object

