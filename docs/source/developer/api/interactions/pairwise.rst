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

5. Accumulator initializer (on GPU)

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

