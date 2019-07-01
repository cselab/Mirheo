#pragma once

template <typename TPadding = float4>
__HD__ constexpr static size_t getPaddedSize(size_t datumSize, int n)
{
    size_t size = n * datumSize;
    size_t npads = (size + sizeof(TPadding)-1) / sizeof(TPadding);
    return npads * sizeof(TPadding);
}

template <typename T, typename TPadding = float4>
__HD__ static size_t getPaddedSize(int n)
{
    return getPaddedSize<TPadding>(sizeof(T), n);
}
