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

The :any:`mirheo::PairwiseInteraction` takes a functor that describes a pairwise interaction.
This functor may be splitted into two parts:

- a handler, that must be usable on the device.
- a manager, that may store extra information on the host. For simple interactions, this can be the same as the handler class.

The interface of the functor must follow the following requirements:

1. Define a view type to be passed (e.g. :any:`mirheo::PVview`) as well as a particle type to be fetched

   .. code-block:: c++

      using ViewType = <particle vector view type>
      using ParticleType = <particle type>
      using HandlerType = <type passed to GPU>

2. Setup function (on Host, for manager only)

   .. code-block:: c++

      void setup(LocalParticleVector* lpv1, LocalParticleVector* lpv2, CellList* cl1, CellList* cl2, const MirState *state);
	
3. Handler function (on Host, for manager only)

   .. code-block:: c++

      const HandlerType& handler() const;

4. Interaction function (output must match with accumulator, see below) (on GPU)

   .. code-block:: c++
      
      __D__ <OutputType> operator()(const ParticleType dst, int dstId, const ParticleType src, int srcId) const;

5. :ref:`Accumulator <dev-interactions-pairwise-accumulators>` initializer (on GPU)

   .. code-block:: c++

      __D__ <Accumulator> getZeroedAccumulator() const;


6. Fetch functions (see in `fetchers.h`):

   .. code-block:: c++

      __D__ ParticleType read(const ViewType& view, int id) const;
      __D__ ParticleType readNoCache(const ViewType& view, int id) const;
      
      __D__ void readCoordinates(ParticleType& p, const ViewType& view, int id) const;
      __D__ void readExtraData(ParticleType& p, const ViewType& view, int id) const;
      
7. Interacting checker to discard pairs not within cutoff:

   .. code-block:: c++

      __D__ bool withinCutoff(const ParticleType& src, const ParticleType& dst) const;
	
8. Position getter from generic particle type:

   .. code-block:: c++

      __D__ real3 getPosition(const ParticleType& p) const;

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
