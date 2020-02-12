#include <gtest/gtest.h>
#include <mirheo/core/interactions/factory.h>
#include <mirheo/core/interactions/pairwise.h>
#include <mirheo/core/logger.h>
#include <mirheo/core/mirheo.h>
#include <mirheo/core/snapshot.h>
#include <mirheo/core/utils/config.h>
#include <mirheo/core/utils/type_traits.h>

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

    friend bool operator==(const Struct1 &x, const Struct1 &y) {
        return x.b == y.b && x.a == y.a && x.c == y.c;
    };
};

struct Struct2 {
    int x;
    Struct1 y;  // Test that recursion works.

    friend bool operator==(const Struct2 &a, const Struct2 &b) {
        return a.x == b.x && a.y == b.y;
    };
};

struct Struct3 {
    int z;
    Struct2 w;
};

// Test manual specialization.
template <>
struct mirheo::TypeLoadSave<Struct1>
{
    static ConfigValue save(Saver& saver, const Struct1& s)
    {
        return ConfigValue::Object{
            {"b", saver(s.b)},
            {"a", saver(s.a)},
            {"c", saver(s.c)},
        };
    }
    static Struct1 load(Loader& loader, const ConfigValue& config)
    {
        return Struct1{
            loader.load<int>(config["b"]),
            loader.load<double>(config["a"]),
            loader.load<std::string>(config["c"]),
        };
    }
};

// Test saving through reflection.
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
    SaverContext context;
    Saver saver{&context};

    // Test basic variant types.
    ASSERT_STREQ(saver(10).toJSONString().c_str(), "10");
    ASSERT_STREQ(saver(12.125).toJSONString().c_str(), "12.125");
    ASSERT_STREQ(saver("asdf").toJSONString().c_str(), "\"asdf\"");
    ASSERT_STREQ(saver(std::vector<int>{10, 20, 30}).toJSONString().c_str(),
                 "[\n    10,\n    20,\n    30\n]");
    ASSERT_STREQ(saver(std::map<std::string, int>{{"a", 10}, {"b", 20}}).toJSONString().c_str(),
                 "{\n    \"a\": 10,\n    \"b\": 20\n}");

    // Test escaping special characters.
    ASSERT_STREQ(saver("\"  '  \\  \f  \b  \r  \n  \t").toJSONString().c_str(),
                 "\"\\\"  '  \\\\  \\f  \\b  \\r  \\n  \\t\"");
}

/// Test TypeLoadSave various interfaces.
TEST(Snapshot, InterfacesForTypeLoadSave)
{
    SaverContext context;
    Saver saver{&context};

    // Test TypeLoadSave<> specialization save() interface.
    ConfigValue config1 = saver(Struct1{10, 3.125, "hello"});
    ASSERT_STREQ(removeWhitespace(config1.toJSONString()).c_str(),
                 "{\"b\":10,\"a\":3.125,\"c\":\"hello\"}");

    // Test saving using MemberVars.
    ConfigValue config2 = saver(Struct2{100, Struct1{10, 3.125, "hello"}});
    ASSERT_STREQ(removeWhitespace(config2.toJSONString()).c_str(),
                 "{\"x\":100,\"y\":{\"b\":10,\"a\":3.125,\"c\":\"hello\"}}");

    // Test saving using MIRHEO_MEMBER_VARS.
    ConfigValue config3 = saver(Struct3{200, Struct2{100, Struct1{10, 3.125, "hello"}}});
    ASSERT_STREQ(removeWhitespace(config3.toJSONString()).c_str(),
                 "{\"z\":200,\"w\":{\"x\":100,\"y\":{\"b\":10,\"a\":3.125,\"c\":\"hello\"}}}");
}

TEST(Snapshot, ParseJSON)
{
    auto testParsing = [](const ConfigValue &config, const std::string &json) {
        std::string a = config.toJSONString();
        std::string b = configFromJSON(json).toJSONString();
        ASSERT_STREQ(a.c_str(), b.c_str());
    };

    testParsing(ConfigValue{10LL}, "10");
    testParsing(ConfigValue{10.125}, "10.125");
    testParsing(ConfigValue{"abc"}, "\"abc\"");
    testParsing(ConfigValue{"a\n\r\b\f\t\"bc"}, R"("a\n\r\b\f\t\"bc")");
    testParsing(ConfigValue::Array{10LL, 20.5, "abc"}, R"([10, 20.5, "abc"])");
    testParsing(ConfigValue::Object{{"b", 10LL}, {"a", 20.5}, {"c", "abc"}},
                R"(  {"b": 10, "a": 20.5, "c": "abc"}  )");
    testParsing(ConfigValue::Object{
            {"b", 10LL},
            {"a", 20.5},
            {"c", ConfigValue::Array{10LL, 20LL, 30LL, "abc"}}},
            R"(  {"b": 10, "a": 20.5, "c": [10, 20, 30, "abc"]}  )");

    ASSERT_EQ(10,   configFromJSON("10").getInt());
    ASSERT_EQ(10.5, configFromJSON("10.5").getFloat());
    ASSERT_EQ(1e10, configFromJSON("1e10").getFloat());
}

template <typename T>
void roundTrip(const T &value) {
    SaverContext context;
    Saver saver{&context};
    ConfigValue saved = saver(value);

    LoaderContext loaderContext{ConfigObject{}, ConfigObject{}};
    Loader loader{&loaderContext};
    auto recovered = loader.load<T>(saved);

    ASSERT_EQ(value, recovered);
}

TEST(Snapshot, DumpUndumpRoundTrip)
{
    roundTrip(10);
    roundTrip(10.5);
    roundTrip(std::string("abc"));
    roundTrip(std::vector<int>{10, 20, 30});
    roundTrip(std::map<std::string, int>{{"c", 10}, {"a", 20}, {"b", 30}});

    using Variant1 = mpark::variant<int, float>;
    roundTrip(Variant1{100});
    roundTrip(Variant1{100.4f});

    using Variant2 = mpark::variant<Struct1, Struct2>;
    roundTrip(Variant2{Struct1{10, 15.5, "abc"}});
    roundTrip(Variant2{Struct2{-20, Struct1{10, 15.5, "abc"}}});
}

TEST(Snapshot, DumpUndumpInteractions)
{
    SaverContext context;
    Saver saver{&context};

    Mirheo mirheo{MPI_COMM_WORLD, {1, 1, 1}, {10.0_r, 10.0_r, 10.0_r}, 0.1_r, {"log", 3, true}, {}, false};
    auto pairwise = InteractionFactory::createPairwiseInteraction(
            mirheo.getState(), "interaction", 1.0, "DPD",
            {{"a", 10.0_r}, {"gamma", 10.0_r}, {"kBT", 1.0_r}, {"power", 0.5_r}});

    {
        saver(pairwise);
        ConfigValue config = saver.getConfig()["Interaction"][0];

        LoaderContext loaderContext{config, ConfigObject{}};
        Loader loader{&loaderContext};
        auto pairwise2 = std::make_shared<PairwiseInteraction>(mirheo.getState(), loader, config.getObject());
        saver(pairwise2);
        ConfigValue config2 = saver.getConfig()["Interaction"][1];
        ASSERT_STREQ(config.toJSONString().c_str(), config2.toJSONString().c_str());
    }
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
