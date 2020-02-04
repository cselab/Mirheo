#pragma once

#include "common.h"  // Forward declarations of Config and ConfigDumper<>.
#include "reflection.h"
#include "type_traits.h"

#include <cassert>
#include <map>
#include <string>
#include <vector>

#include <extern/variant/include/mpark/variant.hpp>
#include <vector_types.h>

namespace mirheo
{

template <typename T, typename Enable>
struct ConfigDumper {
    static_assert(std::is_same<typename remove_cvref<T>::type, T>::value,
                  "Type must be a non-const non-reference type.");
    static_assert(always_false<T>::value, "Not implemented.");

    // Required function(s):
    static Config dump(const T &value);
};

struct Config {
    using Int = long long;
    using Float = double;
    using String = std::string;
    using Dictionary = std::map<std::string, Config>;
    using List = std::vector<Config>;
    using Variant = mpark::variant<Int, Float, String, Dictionary, List>;

    Config(Int value) : value_{value} { }
    Config(Float value) : value_{value} { }
    Config(String value) : value_{std::move(value)} { }
    Config(Dictionary value) : value_{std::move(value)} { }
    Config(List value) : value_{std::move(value)} { }
    Config(const char *str) : value_{std::string(str)} { }
    Config(const Config &) = default;
    Config(Config &&) = default;
    Config& operator=(const Config &) = default;
    Config& operator=(Config &&) = default;

    template <typename T>
    Config(const T &t)
        : value_{ConfigDumper<typename remove_cvref<T>::type>::dump(t).value_}
    { }

    Int getInt() const {
        return mpark::get<Int>(value_);
    }
    Float getFloat() const {
        return mpark::get<Float>(value_);
    }
    const String& getString() const {
        return mpark::get<String>(value_);
    }
    const Dictionary& getDict() const {
        return mpark::get<Dictionary>(value_);
    }
    const List& getList() const {
        return mpark::get<List>(value_);
    }

    Dictionary& getDict() {
        return mpark::get<Dictionary>(value_);
    }
    List& getList() {
        return mpark::get<List>(value_);
    }

    template <typename T>
    inline const T& get() const {
        return mpark::get<T>(value_);
    }
    template <typename T>
    inline const T* get_if() const noexcept {
        return mpark::get_if<T>(&value_);
    }

    template <typename T>
    T read() const; 

    template <typename T>
    void read(T *t) const;

    Variant value_;
};


namespace detail {
    struct ReadItemsHandler {
        struct Dummy { };

        template <typename ...Args>
        static void process(const Args& ...) { }

        template <typename T>
        Dummy operator()(std::string name, const T *t) {
            dict->emplace(std::move(name), ConfigDumper<T>::dump(*t));
            return Dummy{};
        }

        Config::Dictionary *dict;
    };
} // namespace detail

#define MIRHEO_DUMPER_PRIMITIVE(TYPE, ELTYPE)      \
    template <>                                    \
    struct ConfigDumper<TYPE> {                    \
        static Config dump(TYPE x) {               \
            return static_cast<Config::ELTYPE>(x); \
        }                                          \
    }
MIRHEO_DUMPER_PRIMITIVE(bool,               Int);
MIRHEO_DUMPER_PRIMITIVE(int,                Int);
MIRHEO_DUMPER_PRIMITIVE(long,               Int);
MIRHEO_DUMPER_PRIMITIVE(long long,          Int);
MIRHEO_DUMPER_PRIMITIVE(unsigned,           Int);
MIRHEO_DUMPER_PRIMITIVE(unsigned long,      Int);  // This is risky.
MIRHEO_DUMPER_PRIMITIVE(unsigned long long, Int);  // This is risky.
MIRHEO_DUMPER_PRIMITIVE(float,       Float);
MIRHEO_DUMPER_PRIMITIVE(double,      Float);
MIRHEO_DUMPER_PRIMITIVE(std::string, String);
#undef MIRHEO_DUMPER_PRIMITIVE

template <>
struct ConfigDumper<float3> {
    static Config dump(float3 v) {
        return Config::List{(double)v.x, (double)v.y, (double)v.z};
    }
};

/// ConfigDumper for enum types.
template <typename T>
struct ConfigDumper<T, std::enable_if_t<std::is_enum<T>::value>> {
    static Config dump(T t) {
        return static_cast<Config::Int>(t);
    }
};

/// ConfigDumper for structs with reflection information.
template <typename T>
struct ConfigDumper<T, std::enable_if_t<MemberVarsAvailable<T>::value>> {
    static Config dump(const T &t) {
        Config::Dictionary dict;
        MemberVars<T>::foreach(detail::ReadItemsHandler{&dict}, &t);
        return std::move(dict);
    }
};

/// ConfigDumper for std::vector<T>.
template <typename T>
struct ConfigDumper<std::vector<T>> {
    static Config dump(const std::vector<T> &values) {
        Config::List list;
        list.reserve(values.size());
        for (const T &value : values)
            list.push_back(ConfigDumper<T>::dump(value));
        return std::move(list);
    }
};

/// ConfigDumper for std::map<std::string, T>.
template <typename T>
struct ConfigDumper<std::map<std::string, T>> {
    static Config dump(const std::map<std::string, T> &values) {
        Config::Dictionary dict;
        for (const auto &pair : values)
            dict.emplace(pair.first, ConfigDumper<T>::dump(pair.second));
        return std::move(dict);
    }
};

/// ConfigDumper for mpark::variant.
template <typename ...Ts>
struct ConfigDumper<mpark::variant<Ts...>> {
    static Config dump(const mpark::variant<Ts...> &value) {
        auto valueConfig = mpark::visit([](const auto &v) -> Config {
            return ConfigDumper<typename remove_cvref<decltype(v)>::type>::dump(v);
        }, value);
        return Config::Dictionary{
            {"__index", value.index()},
            {"value", std::move(valueConfig)},
        };
    }
};


std::string configToText(const Config &element);
std::string configToJSON(const Config &element);

} // namespace mirheo
