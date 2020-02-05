#pragma once

#include "type_traits.h"

namespace mirheo
{

/**
 * `MemberVars<T>` implements a function `foreach` with which the member
 * variables of `T` can be inspected. Useful for serializing and unserializing
 * objects.
 *
 * Example implementation:
 *
 *      template <>
 *      struct MemberVars<LogInfo>
 *      {
 *          template <typename Handler, typename Me>
 *          auto foreach(Handler &&h, Me *me)
 *          {
 *              return h.process(
 *                  h("fileName",     &me->fileName);
 *                  h("verbosityLvl", &me->verbosityLvl);
 *                  h("noSplash",     &me->noSplash));
 *          }
 *      };
 *
 * The `Handler` is a class which implements two functions:
 *      template <typename MemberVar>
 *      <non-void type> operator()(const char *name, MemberVar *);
 * and
 *      template <typename ...Args>
 *      <any type> process()(Args &&...);
 * where `Args` are always equal to the non-void type from operator().
 *
 * Note: The order of evaluation of the function operator() is unspecified!
 *       Order-sensitive operations MUST be done in the process() function.
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
#define MIRHEO_MEMBER_VARS_BEGIN_(TYPE)            \
    template <>                                    \
    struct MemberVars<TYPE>                        \
    {                                              \
        static constexpr const char *name = #TYPE; \
        template <typename Handler, typename Me>   \
        static auto foreach(Handler &&h, Me *me)   \
        {                                          \
            (void)h;                               \
            (void)me;                              \
            return std::forward<Handler>(h).process(
#define MIRHEO_MEMBER_VARS_DUMP_0_(TYPE)
#define MIRHEO_MEMBER_VARS_DUMP_1_(TYPE, A) \
                h(#A, &me->A)
#define MIRHEO_MEMBER_VARS_DUMP_2_(TYPE, A, B) \
                h(#A, &me->A), \
                h(#B, &me->B)
#define MIRHEO_MEMBER_VARS_DUMP_3_(TYPE, A, B, C) \
                h(#A, &me->A), \
                h(#B, &me->B), \
                h(#C, &me->C)
#define MIRHEO_MEMBER_VARS_DUMP_4_(TYPE, A, B, C, D) \
                h(#A, &me->A), \
                h(#B, &me->B), \
                h(#C, &me->C), \
                h(#D, &me->D)
#define MIRHEO_MEMBER_VARS_DUMP_5_(TYPE, A, B, C, D, E) \
                h(#A, &me->A), \
                h(#B, &me->B), \
                h(#C, &me->C), \
                h(#D, &me->D), \
                h(#E, &me->E)
#define MIRHEO_MEMBER_VARS_DUMP_6_(TYPE, A, B, C, D, E, F) \
                h(#A, &me->A), \
                h(#B, &me->B), \
                h(#C, &me->C), \
                h(#D, &me->D), \
                h(#E, &me->E), \
                h(#F, &me->F)
#define MIRHEO_MEMBER_VARS_DUMP_7_(TYPE, A, B, C, D, E, F, G) \
                h(#A, &me->A), \
                h(#B, &me->B), \
                h(#C, &me->C), \
                h(#D, &me->D), \
                h(#E, &me->E), \
                h(#F, &me->F), \
                h(#G, &me->G)
#define MIRHEO_MEMBER_VARS_DUMP_8_(TYPE, A, B, C, D, E, F, G, H) \
                h(#A, &me->A), \
                h(#B, &me->B), \
                h(#C, &me->C), \
                h(#D, &me->D), \
                h(#E, &me->E), \
                h(#F, &me->F), \
                h(#G, &me->G), \
                h(#H, &me->H)
#define MIRHEO_MEMBER_VARS_END_(TYPE) \
            );                        \
        }                             \
    }

/**
 * Generate `MemberVars<TYPE>` for the given type `TYPE`.
 *
 * Example:
 *      namespace mirheo {
 *          struct S { int a, b; double c; };
 *          MIRHEO_MEMBER_VARS_3(S, a, b, c);
 *      } // namespace mirheo
 *
 *
 * Note:
 *      The implementation which doesn't require manually specifying the number
 *      of arguments can be found here:
 *      https://stackoverflow.com/questions/6707148/foreach-macro-on-macros-arguments
 */
#define MIRHEO_MEMBER_VARS_0(TYPE)   \
    MIRHEO_MEMBER_VARS_BEGIN_(TYPE)  \
    MIRHEO_MEMBER_VARS_DUMP_0_(TYPE) \
    MIRHEO_MEMBER_VARS_END_(TYPE)
#define MIRHEO_MEMBER_VARS_1(TYPE, A)   \
    MIRHEO_MEMBER_VARS_BEGIN_(TYPE)     \
    MIRHEO_MEMBER_VARS_DUMP_1_(TYPE, A) \
    MIRHEO_MEMBER_VARS_END_(TYPE)
#define MIRHEO_MEMBER_VARS_2(TYPE, A, B)   \
    MIRHEO_MEMBER_VARS_BEGIN_(TYPE)        \
    MIRHEO_MEMBER_VARS_DUMP_2_(TYPE, A, B) \
    MIRHEO_MEMBER_VARS_END_(TYPE)
#define MIRHEO_MEMBER_VARS_3(TYPE, A, B, C)   \
    MIRHEO_MEMBER_VARS_BEGIN_(TYPE)           \
    MIRHEO_MEMBER_VARS_DUMP_3_(TYPE, A, B, C) \
    MIRHEO_MEMBER_VARS_END_(TYPE)
#define MIRHEO_MEMBER_VARS_4(TYPE, A, B, C, D)   \
    MIRHEO_MEMBER_VARS_BEGIN_(TYPE)              \
    MIRHEO_MEMBER_VARS_DUMP_4_(TYPE, A, B, C, D) \
    MIRHEO_MEMBER_VARS_END_(TYPE)
#define MIRHEO_MEMBER_VARS_5(TYPE, A, B, C, D, E)   \
    MIRHEO_MEMBER_VARS_BEGIN_(TYPE)                 \
    MIRHEO_MEMBER_VARS_DUMP_5_(TYPE, A, B, C, D, E) \
    MIRHEO_MEMBER_VARS_END_(TYPE)
#define MIRHEO_MEMBER_VARS_6(TYPE, A, B, C, D, E, F)   \
    MIRHEO_MEMBER_VARS_BEGIN_(TYPE)                    \
    MIRHEO_MEMBER_VARS_DUMP_6_(TYPE, A, B, C, D, E, F) \
    MIRHEO_MEMBER_VARS_END_(TYPE)
#define MIRHEO_MEMBER_VARS_7(TYPE, A, B, C, D, E, F, G)   \
    MIRHEO_MEMBER_VARS_BEGIN_(TYPE)                       \
    MIRHEO_MEMBER_VARS_DUMP_7_(TYPE, A, B, C, D, E, F, G) \
    MIRHEO_MEMBER_VARS_END_(TYPE)
#define MIRHEO_MEMBER_VARS_8(TYPE, A, B, C, D, E, F, G, H)   \
    MIRHEO_MEMBER_VARS_BEGIN_(TYPE)                          \
    MIRHEO_MEMBER_VARS_DUMP_8_(TYPE, A, B, C, D, E, F, G, H) \
    MIRHEO_MEMBER_VARS_END_(TYPE)

} // namespace mirheo
