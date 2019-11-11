#pragma once

namespace mirheo
{

// template <typename... Ts>
// using void_t = void;
//
// The simple code above does not work properly due to a C++14 defect:
// https://en.cppreference.com/w/cpp/types/void_t
// http://open-std.org/JTC1/SC22/WG21/docs/cwg_defects.html#1558

template<typename... Ts> struct make_void { typedef void type;};
template<typename... Ts> using void_t = typename make_void<Ts...>::type;

} // namespace mirheo
