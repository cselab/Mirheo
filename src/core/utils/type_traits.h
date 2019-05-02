#pragma once

#include <type_traits>
#include <utility>

#if __cplusplus < 201703L
namespace std
{
template<typename... Ts> struct make_void { typedef void type;};
template<typename... Ts> using void_t = typename make_void<Ts...>::type;
}
#endif


template <typename T, typename = void>
struct is_additionable : std::false_type {};

template <typename T>
struct is_additionable<T, std::void_t<decltype(std::declval<T>() +
                                               std::declval<T>())>>
    : std::true_type {};
    
