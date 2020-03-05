#pragma once

#include "type_traits.h"
#include <string>

namespace mirheo
{

template <typename T>
struct TypeName
{
    static_assert(always_false<T>::value, "TypeName not available.");
    static constexpr const char *name = "UnknownTypeName";
};

/// Set given type's diagnostics/snapshotting name.
#define MIRHEO_TYPE_NAME(TYPE, NAME)              \
    template <>                                   \
    struct TypeName<TYPE> {                       \
        static constexpr const char* name = NAME; \
    }

/// Set given type's diagnostics/snapshotting name to the C++ type name itself.
#define MIRHEO_TYPE_NAME_AUTO(TYPE)               \
    MIRHEO_TYPE_NAME(TYPE, #TYPE)


/** \brief Construct a template class name from the template name and template argument names.

    C-style variadic argument is used as a compile-time-friendly solution.

    Example:
        std::string name = _constructTypeName("Foo", 3, "A", "Boo", "Cookie");
        --> "Foo<A, Boo, Cookie>"
 */
std::string constructTypeName(const char *base, int N, ...);

/** \brief Construct a template instantiation name.

    Example:
        MIRHEO_TYPE_NAME(A, "A");
        MIRHEO_TYPE_NAME(B, "Boo");
        MIRHEO_TYPE_NAME(C, "Cookie");
        std::string name = constructTypeName<A, B, C>("Foo");
        --> "Foo<A, Boo, Cookie>"
 */
template <typename... Args>
inline std::string constructTypeName(const char *templateName)
{
    return constructTypeName(templateName, sizeof...(Args), TypeName<Args>::name...);
}

/** \brief Struct reflection.
 *
    `MemberVars<T>` implements a function `foreach` with which the member
    variables of `T` can be inspected. Useful for serializing and unserializing
    objects.

    Example implementation:

        template <>
        struct MemberVars<LogInfo>
        {
            template <typename Handler, typename Me>
            auto foreach(Handler &&h, Me *me)
            {
                return h.process(
                    h("fileName",     &me->fileName);
                    h("verbosityLvl", &me->verbosityLvl);
                    h("noSplash",     &me->noSplash));
            }
        };

    The `Handler` is a class which implements two functions:
         template <typename MemberVar>
         <non-void type> operator()(const char *name, MemberVar *);
    and
         template <typename ...Args>
         <any type> process()(Args &&...);
    where `Args` are always equal to the non-void type from operator().

    Note: The order of evaluation of the function operator() is unspecified!
          Order-sensitive operations MUST be done in the process() function.
 */
template <typename T>
struct MemberVars
{
    static constexpr bool notImplemented_ = true;
};

/// `MemberVarsAvailable<T>` is true if the struct `MemberVars<T>` has been specialized.
template <typename T, typename Enable = void>
struct MemberVarsAvailable
{
    static constexpr bool value = true;
};

template <typename T>
struct MemberVarsAvailable<T, std::enable_if_t<MemberVars<T>::notImplemented_>>
{
    static constexpr bool value = false;
};

// Implementation detail.
#define MIRHEO_MEMBER_VARS_BEGIN_(TYPE) MIRHEO_MEMBER_VARS_BEGIN2_(TYPE)
#define MIRHEO_MEMBER_VARS_BEGIN2_(TYPE)           \
    template <>                                    \
    struct TypeName<TYPE>                          \
    {                                              \
        static constexpr const char *name = #TYPE; \
    };                                             \
    template <>                                    \
    struct MemberVars<TYPE>                        \
    {                                              \
        template <typename Handler, typename Me>   \
        static auto foreach(Handler &&h, Me *me)   \
        {                                          \
            (void)h;                               \
            (void)me;                              \
            return std::forward<Handler>(h).process(
#define MIRHEO_MEMBER_VARS_0_(TYPE)
#define MIRHEO_MEMBER_VARS_1_(TYPE, A) \
                h(#A, &me->A)
#define MIRHEO_MEMBER_VARS_2_(TYPE, A, B) \
                h(#A, &me->A), \
                h(#B, &me->B)
#define MIRHEO_MEMBER_VARS_3_(TYPE, A, B, C) \
                h(#A, &me->A), \
                h(#B, &me->B), \
                h(#C, &me->C)
#define MIRHEO_MEMBER_VARS_4_(TYPE, A, B, C, D) \
                h(#A, &me->A), \
                h(#B, &me->B), \
                h(#C, &me->C), \
                h(#D, &me->D)
#define MIRHEO_MEMBER_VARS_5_(TYPE, A, B, C, D, E) \
                h(#A, &me->A), \
                h(#B, &me->B), \
                h(#C, &me->C), \
                h(#D, &me->D), \
                h(#E, &me->E)
#define MIRHEO_MEMBER_VARS_6_(TYPE, A, B, C, D, E, F) \
                h(#A, &me->A), \
                h(#B, &me->B), \
                h(#C, &me->C), \
                h(#D, &me->D), \
                h(#E, &me->E), \
                h(#F, &me->F)
#define MIRHEO_MEMBER_VARS_7_(TYPE, A, B, C, D, E, F, G) \
                h(#A, &me->A), \
                h(#B, &me->B), \
                h(#C, &me->C), \
                h(#D, &me->D), \
                h(#E, &me->E), \
                h(#F, &me->F), \
                h(#G, &me->G)
#define MIRHEO_MEMBER_VARS_8_(TYPE, A, B, C, D, E, F, G, H) \
                h(#A, &me->A), \
                h(#B, &me->B), \
                h(#C, &me->C), \
                h(#D, &me->D), \
                h(#E, &me->E), \
                h(#F, &me->F), \
                h(#G, &me->G), \
                h(#H, &me->H)
#define MIRHEO_MEMBER_VARS_END_ \
            );                  \
        }                       \
    }

/// Get the first element of a __VA_ARGS__.
#define MIRHEO_FIRST_HELPER_(FIRST, ...) FIRST
#define MIRHEO_FIRST(...) MIRHEO_FIRST_HELPER_(__VA_ARGS__, dummy)

/// Concatete three tokens.
#define MIRHEO_CONCAT3_(A, B, C)  A##B##C
#define MIRHEO_CONCAT3(A, B, C) MIRHEO_CONCAT3_(A, B, C)

/// Evaluate a macro, assuming a non-zero number of arguments.
#define MIRHEO_EVAL(MACRO, ...) MACRO(__VA_ARGS__)

/// Get the number of arguments, reduced by 1. This way we get around the
/// restriction that __VA_ARGS__ must be non-empty.
#define MIRHEO_NARGS_MINUS_1_(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, N, ...) N
#define MIRHEO_NARGS_MINUS_1(...) \
    MIRHEO_NARGS_MINUS_1_(__VA_ARGS__, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, -1)

/** \brief Generate `MemberVars<TYPE>` for the given type `TYPE`.

    Example:
         namespace mirheo {
             struct S { int a, b; double c; };
             MIRHEO_MEMBER_VARS(3, S, a, b, c);
         } // namespace mirheo

    Note:
         Alternatively, we can simply use std::tie(__VA_ARGS__):
         https://github.com/KonanM/tser/blob/abf343c9ad018ffdbe61d66c0065b198380ed972/include/tser/serialize.hpp#L275-L277
 */
#define MIRHEO_MEMBER_VARS(...)                          \
    MIRHEO_MEMBER_VARS_BEGIN_(MIRHEO_FIRST(__VA_ARGS__)) \
    MIRHEO_EVAL(MIRHEO_CONCAT3(MIRHEO_MEMBER_VARS_, MIRHEO_NARGS_MINUS_1(__VA_ARGS__), _), __VA_ARGS__) \
    MIRHEO_MEMBER_VARS_END_

} // namespace mirheo
