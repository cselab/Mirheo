#pragma once

#include <cstdint>

void scan(const uint8_t* in, const int n, int* out, cudaStream_t stream);
