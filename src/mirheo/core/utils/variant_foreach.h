#pragma once

#include <extern/variant/include/mpark/variant.hpp>

namespace mirheo
{

template <typename... Variants>
struct VariantForeachHelper;

template <typename... Args>
struct VariantForeachHelper<mpark::variant<Args...>>
{
    template <typename Visitor, typename ...OtherArgs>
    static void eval(Visitor &vis)
    {
        // https://stackoverflow.com/a/51006031
        using fold_expression = int[];

        // First expand the `OtherArgs` variadic template, then `Args`.
        (void)fold_expression{0, (vis.template operator()<OtherArgs..., Args>(), 0)...};
    }
};

template <typename... Args, typename... Variants>
struct VariantForeachHelper<mpark::variant<Args...>, Variants...>
{
    template <typename Visitor, typename ...OtherArgs>
    static void eval(Visitor &vis)
    {
        using fold_expression = int[];
        (void)fold_expression{0, (VariantForeachHelper<Variants...>::template eval<
                Visitor, OtherArgs..., Args>(vis), 0)...};
    }
};


/** Execute visitor's `eval` template function for each combination of variant types.

    Example:
        struct Visitor
        {
            template <typename A, typename B, typename C>
            void operator()()
            {
                printf("A=%d B=%d C=%d\n", A::value, B::value, c::value);
            }
        };

        struct A1 { static constexpr int value = 1; };
        struct A2 { static constexpr int value = 2; };
        struct B1 { static constexpr int value = 10; };
        struct B2 { static constexpr int value = 20; };
        struct C1 { static constexpr int value = 100; };
        struct C2 { static constexpr int value = 200; };
        void test() {
            // printf 8 combinations of values.
            variantForeach<Visitor,
                           mpark::variant<A1, A2>,
                           mpark::variant<B1, B2>,
                           mpark::variant<C1, C2>>(Visitor{});
        }

    Returns:
        Forwards back the Visitor object.

    Note:
        If necessary, this can be generalized to arbitrary variadic templates.
 */
template <typename Visitor, typename... Variants>
decltype(auto) variantForeach(Visitor &&vis)
{
    VariantForeachHelper<Variants...>::template eval<Visitor>(vis);
    return static_cast<Visitor&&>(vis); // Forward.
}

} // namespace mirheo
