#include <gtest/gtest.h>
#include <mirheo/core/logger.h>
#include <mirheo/core/utils/config.h>

using namespace mirheo;

static std::string removeWhitespace(std::string str) {
    int shift = 0;
    for (size_t i = 0; i < str.size(); ++i) {
        if (str[i] <= ' ')
            ++shift;
        else
            str[i - shift] = str[i];
    }
    str.erase(str.end() - shift, str.end());
    return str;
}

struct Struct1 {
    // Test that the order b, a, c is preserved.
    int b;
    double a;
    std::string c;
};

struct Struct2 {
    int x;
    Struct1 y;  // Test that recursion works.
};

struct Struct3 {
    int z;
    Struct2 w;
};

// Test manual specialization.
template <>
struct mirheo::ConfigDumper<Struct1>
{
    static Config dump(Dumper& dumper, const Struct1& s)
    {
        return Config::Dictionary{
            {"b", dumper(s.b)},
            {"a", dumper(s.a)},
            {"c", dumper(s.c)},
        };
    }
};

// Test dumping through reflection.
template <>
struct mirheo::MemberVars<Struct2>
{
    template <typename Handler, typename Me>
    static auto foreach(Handler &&h, Me *me)
    {
        return h.process(h("x", &me->x), h("y", &me->y));
    }
};

// Test the reflection macro.
namespace mirheo {
MIRHEO_MEMBER_VARS_2(Struct3, z, w);
} // namespace mirheo


TEST(Snapshot, BasicConfigToJSON)
{
    Dumper dumper{MPI_COMM_WORLD, "snapshot/", true};

    // Test basic variant types.
    ASSERT_STREQ(configToJSON(dumper(10)).c_str(), "10");
    ASSERT_STREQ(configToJSON(dumper(12.125)).c_str(), "12.125");
    ASSERT_STREQ(configToJSON(dumper("asdf")).c_str(), "\"asdf\"");
    ASSERT_STREQ(configToJSON(dumper(std::vector<int>{10, 20, 30})).c_str(),
                 "[\n    10,\n    20,\n    30\n]");
    ASSERT_STREQ(configToJSON(dumper(std::map<std::string, int>{{"a", 10}, {"b", 20}})).c_str(),
                 "{\n    \"a\": 10,\n    \"b\": 20\n}");

    // Test escaping special characters.
    ASSERT_STREQ(configToJSON(dumper("\"  '  \\  \f  \b  \r  \n  \t")).c_str(),
                 "\"\\\"  '  \\\\  \\f  \\b  \\r  \\n  \\t\"");
}

/// Test ConfigDumper various interfaces.
TEST(Snapshot, InterfacesForConfigDumper)
{
    Dumper dumper{MPI_COMM_WORLD, "snapshot/", true};

    // Test ConfigDumper<> specialization dump() interface.
    Config config1 = dumper(Struct1{10, 3.125, "hello"});
    ASSERT_STREQ(removeWhitespace(configToJSON(config1)).c_str(),
                 "{\"b\":10,\"a\":3.125,\"c\":\"hello\"}");

    // Test dumping using MemberVars.
    Config config2 = dumper(Struct2{100, Struct1{10, 3.125, "hello"}});
    ASSERT_STREQ(removeWhitespace(configToJSON(config2)).c_str(),
                 "{\"x\":100,\"y\":{\"b\":10,\"a\":3.125,\"c\":\"hello\"}}");

    // Test dumping using MIRHEO_MEMBER_VARS.
    Config config3 = dumper(Struct3{200, Struct2{100, Struct1{10, 3.125, "hello"}}});
    ASSERT_STREQ(removeWhitespace(configToJSON(config3)).c_str(),
                 "{\"z\":200,\"w\":{\"x\":100,\"y\":{\"b\":10,\"a\":3.125,\"c\":\"hello\"}}}");
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    logger.init(MPI_COMM_WORLD, "snapshot.log", 9);

    testing::InitGoogleTest(&argc, argv);
    auto ret = RUN_ALL_TESTS();

    MPI_Finalize();
    return ret;
}
