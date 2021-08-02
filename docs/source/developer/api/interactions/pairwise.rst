.. _dev-interactions-pairwise:

Pairwise Interactions
=====================

Base class
----------

This is the visible class that is output of the factory function.

.. doxygenclass:: mirheo::BasePairwiseInteraction
   :project: mirheo
   :members:

Implementation
--------------

The factory instantiates one of this templated class.
See below for the requirements on the kernels.

.. doxygenclass:: mirheo::PairwiseInteraction
   :project: mirheo
   :members:

A specific class can be used to compute addtionally the stresses of a given interaction.

.. doxygenclass:: mirheo::PairwiseInteractionWithStress
   :project: mirheo
   :members:


.. _dev-interactions-pairwise-kernels:

Kernels
-------

Interface
^^^^^^^^^

The :any:`mirheo::PairwiseInteraction` takes a functor that describes a pairwise interaction.
This functor may be splitted into two parts:

- a handler, that must be usable on the device.
- a manager, that may store extra information on the host. For simple interactions, this can be the same as the handler class.

The interface of the functor must follow the following requirements:

#. Define a view type to be passed (e.g. :any:`mirheo::PVview`) as well as a particle type to be fetched and the parameter struct used for initialization:

   .. code-block:: c++

      using ViewType = <particle vector view type>
      using ParticleType = <particle type>
      using HandlerType = <type passed to GPU>
      using ParamsType = <struct that contains the parameters of this functor>

#. A generic constructor from the ``ParamsType`` parameters:

   .. code-block:: c++

      PairwiseKernelType(real rc, const ParamsType& p, real dt, long seed=42424242);


#. Setup function (on Host, for manager only)

   .. code-block:: c++

      void setup(LocalParticleVector* lpv1, LocalParticleVector* lpv2, CellList* cl1, CellList* cl2, const MirState *state);

#. Handler function (on Host, for manager only)

   .. code-block:: c++

      const HandlerType& handler() const;

#. Interaction function (output must match with accumulator, see below) (on GPU)

   .. code-block:: c++

      __D__ <OutputType> operator()(const ParticleType dst, int dstId, const ParticleType src, int srcId) const;

#. :ref:`Accumulator <dev-interactions-pairwise-accumulators>` initializer (on GPU)

   .. code-block:: c++

      __D__ <Accumulator> getZeroedAccumulator() const;


#. Fetch functions (see in `fetchers.h` or see the :ref:`docs <dev-interactions-pairwise-kernels-fetchers>`):

   .. code-block:: c++

      __D__ ParticleType read(const ViewType& view, int id) const;
      __D__ ParticleType readNoCache(const ViewType& view, int id) const;

      __D__ void readCoordinates(ParticleType& p, const ViewType& view, int id) const;
      __D__ void readExtraData(ParticleType& p, const ViewType& view, int id) const;

#. Interacting checker to discard pairs not within cutoff:

   .. code-block:: c++

      __D__ bool withinCutoff(const ParticleType& src, const ParticleType& dst) const;

#. Position getter from generic particle type:

   .. code-block:: c++

      __D__ real3 getPosition(const ParticleType& p) const;

.. note::

   To implement a new kernel, the following must be done:
   - satisfy the above interface
   - add a corresponding parameter in parameters.h
   - add it to the variant in parameters.h
   - if necessary, add type traits specialization in type_traits.h


This is the interface for the host calls:

.. doxygenclass:: mirheo::PairwiseKernel
   :project: mirheo
   :members:

The rest is directly implemented in the kernels, as no virtual functions are allowed on the device.

Implemented kernels
^^^^^^^^^^^^^^^^^^^
.. doxygenclass:: mirheo::PairwiseDensity
   :project: mirheo
   :members:

.. doxygenclass:: mirheo::PairwiseDPDHandler
   :project: mirheo
   :members:

.. doxygenclass:: mirheo::PairwiseDPD
   :project: mirheo
   :members:

.. doxygenclass:: mirheo::PairwiseLJ
   :project: mirheo
   :members:

.. doxygenclass:: mirheo::PairwiseMorse
   :project: mirheo
   :members:

.. doxygenclass:: mirheo::PairwiseMDPDHandler
   :project: mirheo
   :members:

.. doxygenclass:: mirheo::PairwiseMDPD
   :project: mirheo
   :members:

.. doxygenclass:: mirheo::PairwiseNorandomDPD
   :project: mirheo
   :members:

.. doxygenclass:: mirheo::PairwiseRepulsiveLJ
   :project: mirheo
   :members:

.. doxygenclass:: mirheo::PairwiseSDPDHandler
   :project: mirheo
   :members:

.. doxygenclass:: mirheo::PairwiseSDPD
   :project: mirheo
   :members:


The above kernels that output a force can be wrapped by the stress wrapper:

.. doxygenclass:: mirheo::PairwiseStressWrapperHandler
   :project: mirheo
   :members:

.. doxygenclass:: mirheo::PairwiseStressWrapper
   :project: mirheo
   :members:


.. _dev-interactions-pairwise-kernels-fetchers:

Fetchers
^^^^^^^^

Fetchers are used to load the correct data needed by the pairwise kernels (e.g. the :any:`mirheo::PairwiseRepulsiveLJ` kernel needs only the positions while the :any:`mirheo::PairwiseSDPD` kernel needs also velocities and number densities).

.. doxygenclass:: mirheo::ParticleFetcher
   :project: mirheo
   :members:

.. doxygenclass:: mirheo::ParticleFetcherWithVelocity
   :project: mirheo
   :members:

.. doxygenclass:: mirheo::ParticleFetcherWithVelocityAndDensity
   :project: mirheo
   :members:

.. doxygenclass:: mirheo::ParticleFetcherWithVelocityDensityAndMass
   :project: mirheo
   :members:


.. _dev-interactions-pairwise-accumulators:

Accumulators
------------

Every :ref:`interaction kernel <dev-interactions-pairwise-kernels>` must initialize an accumulator that is used to add its output quantity.
Depending on the kernel, that quantity may be of different type, and may behave in a different way (e.g. forces and stresses are different).

It must satisfy the following interface requirements (in the following, we denote the type of the local variable as :code:`LType`
and the :ref:`view type<dev-pv-views>` as :code:`ViewType`):

1. A default constructor which initializes the internal local variable
2. Atomic accumulator from local value to destination view:

   .. code-block:: c++

      __D__ void atomicAddToDst(LType, ViewType&, int id) const;

3. Atomic accumulator from local value to source view:

   .. code-block:: c++

      __D__ inline void atomicAddToSrc(LType, ViewType&, int id) const;

4. Accessor of accumulated value:

   .. code-block:: c++

      __D__ inline LType get() const;

5. Function to add a value to the accumulator (from output of pairwise kernel):

   .. code-block:: c++

      __D__ inline void add(LType);

The following accumulators are currently implemented:

.. doxygenclass:: mirheo::DensityAccumulator
   :project: mirheo
   :members:

.. doxygenclass:: mirheo::ForceAccumulator
   :project: mirheo
   :members:

.. doxygenstruct:: mirheo::ForceStress
   :project: mirheo
   :members:

.. doxygenclass:: mirheo::ForceStressAccumulator
   :project: mirheo
   :members:
