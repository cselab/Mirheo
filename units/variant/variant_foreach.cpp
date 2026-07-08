#include <mirheo/core/utils/variant_foreach.h>

#include <gtest/gtest.h>

#include <vector>
#include <array>

using namespace mirheo;

struct A1 { static constexpr int value = 1; };
struct A2 { static constexpr int value = 2; };
struct A3 { static constexpr int value = 3; };
struct B1 { static constexpr int value = 10; };
struct B2 { static constexpr int value = 20; };
struct C1 { static constexpr int value = 100; };
struct C2 { static constexpr int value = 200; };
struct D1 { static constexpr int value = 1000; };
struct D2 { static constexpr int value = 2000; };

TEST(Variant, VariantForeach)
{
    std::vector<std::array<int, 4>> v;
    variantForeach<std::variant<A1, A2, A3>,
                   std::variant<B1, B2>,
                   std::variant<C1, C2>,
                   std::variant<D1, D2>>(
            [&v](auto a, auto b, auto c, auto d)
            {
                v.push_back({
                    decltype(a)::type::value,
                    decltype(b)::type::value,
                    decltype(c)::type::value,
                    decltype(d)::type::value,
                });
            });

    /// Test that all 3 * 2 * 2 * 2 == 24 combinations were traversed in the
    /// "row-major" order.
    size_t k = 0;
    for (int a = 1; a <= 2; ++a)
    for (int b = 10; b <= 20; b += 10)
    for (int c = 100; c <= 200; c += 100)
    for (int d = 1000; d <= 2000; d += 1000) {
        ASSERT_LT(k, v.size());
        ASSERT_EQ(v[k], (std::array<int, 4>{a, b, c, d}));
        ++k;
    }
}
