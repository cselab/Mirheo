# accumulator

Wrapper to accumulate outputs of pairwise interactions into views.
This is used in pairwise_kernels.

## implementation

Must contain a (private) variable which stores the accumulated output of the kernel (e.g. force, stress, density...)

## interface requirements

(In the following, we denote the type of the local variable as `LType`
and the view type as `ViewType`)

Default constructor which initialise the local variable  

Atomic accumulator from local value to destination view`

	__D__ inline void atomicAddToDst(LType, ViewType&, int id);

Atomic accumulator from local value to source view

	__D__ inline void atomicAddToSrc(LType, ViewType&, int id);

Accessor ofaccumulated value

	__D__ inline LType get() const;

Accumulator (from output of pairwise kernel)

    __D__ inline void add(LType);
