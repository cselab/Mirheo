#include <cassert>
#include <algorithm>

using namespace std;

#include "rring-buffer.h"

void RRingBuffer::_refill(int s, int e)
{
    assert(e > s && e <= n);
	    
    const int multiple = 2;

    s = s - (s % multiple);
    e = e + (multiple - (e % multiple));
    e = min(e, n);
	    
    curandStatus_t res;
    res = curandGenerateNormal(prng, drsamples + s, e - s, 0, 1);
    assert(res == CURAND_STATUS_SUCCESS);
}
   
RRingBuffer::RRingBuffer(const int n): n(n), s(0), olds(0), c(0), drsamples(NULL)
{
    curandStatus_t res;
    res = curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
    //we could try CURAND_RNG_PSEUDO_MTGP32 or CURAND_RNG_PSEUDO_MT19937
	    
    assert(res == CURAND_STATUS_SUCCESS);
    res = curandSetPseudoRandomGeneratorSeed(prng, 1234ULL);
    assert(res == CURAND_STATUS_SUCCESS);
	    
    cudaMalloc(&drsamples, sizeof(float) * n);
    assert(drsamples != NULL);
    
    update(n);
    assert(s == 0);
}

RRingBuffer::~RRingBuffer()
{
    cudaFree(drsamples);
    curandStatus_t res = curandDestroyGenerator(prng);
    assert(res == CURAND_STATUS_SUCCESS);
}
    
void RRingBuffer::update(const int consumed)
{
    assert(consumed >= 0 && consumed <= n);

    c += consumed;
    assert(c >= 0 && c <= n);
	    
    if (c > 0.45 * n)
    {
	const int c1 = min(olds + c, n) - olds;
	    
	if (c1 > 0)
	    _refill(olds, olds + c1);

	const int c2 = c - c1;

	if (c2 > 0)
	    _refill(0, c2);
	    
	olds = (olds + c) % n;
	s = olds;
	c = 0;
    }
    else
	s = (olds + c) % n;
}
