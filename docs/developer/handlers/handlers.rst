Handlers
=========

Handlers are intended to manipulate data of \c ParticleVectors.
They are called from the \c Simulation at different moments of the pipeline and compute interactions,
perform integration, bounce-back, etc.

All the handlers take a CUDA stream variable as a required parameter!

.. toctree::
   :glob:

   ./*/interface