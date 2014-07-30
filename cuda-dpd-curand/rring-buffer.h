#pragma once

#include <curand.h>

class RRingBuffer
{
    const int n;
    int s, c, olds;
    float * drsamples;
    curandGenerator_t prng;

protected:

    void _refill(int s, int e);
    
public:

    RRingBuffer(const int n);

    ~RRingBuffer();
    
    void update(const int consumed);

    int start() const { return s; }
    float * buffer() const { return drsamples; }
    int nsamples() const { return n; }
};
