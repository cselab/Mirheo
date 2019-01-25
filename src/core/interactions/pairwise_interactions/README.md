# pairwise interactions

A set of pairwise interaction kernel classes that can be passed to pairwise_kernels functions.

## interface requirements

Need to define a view type to be passed

	using ViewType = particle vector <view type>

Setup function

	void setup(LocalParticleVector* lpv1, LocalParticleVector* lpv2, CellList* cl1, CellList* cl2, float t);
	
Interaction function (output must match with accumulator, see below)

	__D__ inline <OutputType> operator()(const Particle dst, int dstId, const Particle src, int srcId) const;

Accumulator initializer

	__D__ inline <Accumulator> getZeroedAccumulator() const;
