#pragma once

#include <type_traits>

namespace mirheo
{
#ifndef DOXYGEN_SHOULD_SKIP_THIS

// template <typename... Ts>
// using void_t = void;
//
// The simple code above does not work properly due to a C++14 defect:
// https://en.cppreference.com/w/cpp/types/void_t
// http://open-std.org/JTC1/SC22/WG21/docs/cwg_defects.html#1558

template<typename... Ts> struct make_void { typedef void type;}; ///< implementation helper for void_t
template<typename... Ts> using void_t = typename make_void<Ts...>::type; ///< equivalent to c++17 void_t

/// Utility metafunction that maps a sequence of any types to the value \c false
template<typename... Ts>
struct always_false {
    static constexpr bool value = false;
};

/// Obtains the type T without any top-level const or volatile qualification; remove, if present, reference of T. 
template<typename T>
struct remove_cvref {
    using type = typename std::remove_cv<typename std::remove_reference<T>::type>::type;
};

/// see remove_cvref
template<typename T>
using remove_cvref_t = typename remove_cvref<T>::type;


template<typename T, typename Enable = void>
struct is_dereferenceable {
    static constexpr bool value = false;
};

template<typename T>
struct is_dereferenceable<T, std::enable_if_t<std::is_same<
        decltype(*std::declval<T>()), decltype(*std::declval<T>())>::value>> {
    static constexpr bool value = true;
};

#endif // DOXYGEN_SHOULD_SKIP_THIS

} // namespace mirheo
