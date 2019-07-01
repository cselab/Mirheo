# pairwise interactions

A set of pairwise interaction kernel classes that can be passed to pairwise_kernels functions.
We may distinguish handler from manager when there are extra data to be hold for setup only but not useful when passed to the kernels.

## interface requirements

Need to define a view type to be passed (example: PVview, PVviewWithStresses...) as well as a particle type to be fetched

	using ViewType = <particle vector view type>
	using ParticleType = <particle type>
	using HandlerType = <type passed to GPU>


Setup function (on Host, for manager only)

	void setup(LocalParticleVector* lpv1, LocalParticleVector* lpv2, CellList* cl1, CellList* cl2, float t);
	
Handler function (on Host, for manager only)

	const HandlerType& handler() const;

Interaction function (output must match with accumulator, see below) (on GPU)

	__D__ inline <OutputType> operator()(const ParticleType dst, int dstId, const ParticleType src, int srcId) const;

Accumulator initializer (on GPU)

	__D__ inline <Accumulator> getZeroedAccumulator() const;


Fetch functions (see in `fetchers.h`):

	__D__ inline ParticleType read(const ViewType& view, int id) const;
	__D__ inline ParticleType readNoCache(const ViewType& view, int id) const;
	
	__D__ inline void readCoordinates(ParticleType& p, const ViewType& view, int id) const;
	__D__ inline void readExtraData(ParticleType& p, const ViewType& view, int id) const;

Interacting checker to discard pairs not within cutoff:

	__D__ inline bool withinCutoff(const ParticleType& src, const ParticleType& dst) const;
	
Position getter from generic particle type:
	
	__D__ inline float3 getPosition(const ParticleType& p) const;
