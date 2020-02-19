#pragma once

#include <extern/variant/include/mpark/variant.hpp>

namespace mirheo
{

/// Identity meta type. Equivalent to C++20 std::type_identity<T>.
template <typename T>
struct type_identity {
    using type = T;
};

template <typename... Variants>
struct VariantForeachHelper;

template <>
struct VariantForeachHelper<>
{
    template <typename Visitor, typename ...AllArgs>
    static void eval(Visitor &vis)
    {
        vis(type_identity<AllArgs>{}...);
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

    Visitor is invoked with an instance of an empty wrapper struct, one for each type:
        template <typename T>
        struct type_identity {
            using type = T;
        };

    Example:
        struct A1 { static constexpr int value = 1; };
        struct A2 { static constexpr int value = 2; };
        struct B1 { static constexpr int value = 10; };
        struct B2 { static constexpr int value = 20; };
        struct C1 { static constexpr int value = 100; };
        struct C2 { static constexpr int value = 200; };
        void test() {
            // Prints 8 combinations in row-major order:
            //      1 10 100
            //      1 10 200
            //      1 20 100
            //      1 20 200
            //      2 10 100
            //      2 10 200
            //      2 20 100
            //      2 20 200
            variantForeach<mpark::variant<A1, A2>,
                           mpark::variant<B1, B2>,
                           mpark::variant<C1, C2>>(
                []()(auto a, auto b, auto c)
                {
                    printf("A=%d B=%d C=%d\n",
                           decltype(a)::type::value,
                           decltype(b)::type::value,
                           decltype(c)::type::value);
                });
        }

    Returns:
        Forwards back the Visitor object.

    Note:
        If necessary, this can be generalized to arbitrary variadic templates.
 */
template <typename... Variants, typename Visitor>
Visitor&& variantForeach(Visitor &&vis)
{
    VariantForeachHelper<Variants...>::template eval<Visitor>(vis);
    return static_cast<Visitor&&>(vis); // Forward.
}

} // namespace mirheo
