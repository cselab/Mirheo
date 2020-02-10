#pragma once

#include "common.h" // Forward declarations of Config and ConfigDumper<>.
#include "flat_ordered_dict.h"
#include "reflection.h"
#include "type_traits.h"

#include <cassert>
#include <map>
#include <mpi.h>
#include <string>
#include <typeinfo>
#include <vector>

#include <extern/variant/include/mpark/variant.hpp>
#include <vector_types.h>

namespace mirheo
{

constexpr const char ConfigNullRefString[] = "<nullptr>";

// Move this to utils/type_checker.h?
void _typeMismatchError [[noreturn]] (const char *thisTypeName, const char *classTypeName);
template <typename T>
void assertType(T *thisPtr)
{
    if (typeid(*thisPtr) != typeid(T))
        _typeMismatchError(typeid(*thisPtr).name(), typeid(T).name());
}


class Dumper;
class Undumper;

template <typename T, typename Enable>
struct ConfigDumper
{
    static_assert(std::is_same<remove_cvref_t<T>, T>::value,
                  "Type must be a non-const non-reference type.");
    static_assert(always_false<T>::value, "Not implemented.");

    static Config dump(Dumper&, T& value);

    /// Context-free parsing. Only for simple types!
    static T parse(const Config&);

    /// Context-aware undump.
    static T undump(Undumper&, const Config&);
};

class ConfigList : public std::vector<Config>
{
    using Base = std::vector<Config>;
public:
    using Base::Base;


    /// Overwrite operator[] with bound checks.
    Config&       operator[](size_t i)       { return at(i); }
    const Config& operator[](size_t i) const { return at(i); }

    Config& at(size_t i) {
        return i < size() ? Base::operator[](i) : _outOfBound(i, size());
    }
    const Config& at(size_t i) const {
        return i < size() ? Base::operator[](i) : (const Config&)_outOfBound(i, size());
    }

private:
    Config& _outOfBound [[noreturn]] (size_t index, size_t size) const;
};

class ConfigDictionary : public FlatOrderedDict<std::string, Config>
{
    using Base = FlatOrderedDict<std::string, Config>;

public:
    using Base::Base;

    /// Overwrite operator[] with bound checks.
    Config&       operator[](const std::string &key)       { return at(key); }
    const Config& operator[](const std::string &key) const { return at(key); }
    Config&       operator[](const char *key)              { return at(key); }
    const Config& operator[](const char *key) const        { return at(key); }

    Config&       at(const std::string &key);
    const Config& at(const std::string &key) const;
    Config&       at(const char *key);
    const Config& at(const char *key) const;

    /// Get the pointer to the key if it exists, otherwise return a nullptr.
    Config*       get(const std::string &key) &;
    const Config* get(const std::string &key) const&;
    Config*       get(const char *key) &;
    const Config* get(const char *key) const&;
};

class Config
{
public:
    using Int        = long long;
    using Float      = double;
    using String     = std::string;
    using List       = ConfigList;
    using Dictionary = ConfigDictionary;
    using Variant    = mpark::variant<Int, Float, String, List, Dictionary>;

    Config(Int value) : value_{value} {}
    Config(Float value) : value_{value} {}
    Config(String value) : value_{std::move(value)} {}
    Config(Dictionary value) : value_{std::move(value)} {}
    Config(List value) : value_{std::move(value)} {}
    Config(const char *str) : value_{std::string(str)} {}
    Config(const Config&) = default;
    Config(Config&&)      = default;
    Config& operator=(const Config&) = default;
    Config& operator=(Config&&) = default;

    template <typename T>
    Config(const T&)
    {
        static_assert(
            always_false<T>::value,
            "Direct construction of the Config object available only "
            "for variant types (Int, Float, String, Dictionary, List). "
            "Did you mean `dumper(value)` instead of `Config{value}`?");
    }

    std::string toJSONString() const;

    /// Getter functions. Terminate if the underlying type is different. Int
    /// and Float variants accept the other type if the conversion is lossless.
    Int getInt() const;
    Float getFloat() const;
    const String& getString() const;
    const List& getList() const;
    List& getList();
    const Dictionary& getDict() const;
    Dictionary& getDict();

    /// Check if the key exists. Terminates if not a dict.
    bool contains(const std::string &key) const { return getDict().contains(key); }
    bool contains(const char *key)        const { return getDict().contains(key); }

    /// Get the element matching the given key. Terminates if not a dict, or if
    /// the key was not found.
    Config&       operator[](const std::string &key)       { return getDict().at(key); }
    const Config& operator[](const std::string &key) const { return getDict().at(key); }
    Config&       operator[](const char *key)              { return getDict().at(key); }
    const Config& operator[](const char *key) const        { return getDict().at(key); }

    /// Get the list element. Terminates if not a list or if out of range.
    Config&       operator[](size_t i)       { return getList()[i]; }
    const Config& operator[](size_t i) const { return getList()[i]; }
    Config&       operator[](int i)       { return getList()[static_cast<size_t>(i)]; }
    const Config& operator[](int i) const { return getList()[static_cast<size_t>(i)]; }

    /// Get the element if it exists, or null otherwise. Terminates if not a dict.
    Config*       get(const std::string &key) &      { return getDict().get(key); }
    const Config* get(const std::string &key) const& { return getDict().get(key); }
    Config*       get(const char *key) &             { return getDict().get(key); }
    const Config* get(const char *key) const&        { return getDict().get(key); }

    /// Implicit cast to simple types.
    template <typename T>
    operator T() const { return ConfigDumper<T>::parse(*this); }

    /// Implicit cast to specific types.
    operator Config::Int() const { return getInt(); }
    operator Config::Float() const { return getFloat(); }
    operator const std::string&() const { return getString(); }

    /// String concatenation operator.
    friend std::string operator+(const Config& a, const char *b)
    {
        return a.getString() + b;
    }
    friend std::string operator+(const char *a, const Config& b)
    {
        return a + b.getString();
    }
    friend std::string operator+(const Config& a, const std::string& b)
    {
        return a.getString() + b;
    }
    friend std::string operator+(const std::string& a, const Config& b)
    {
        return a + b.getString();
    }

    /// Low-level getter.
    template <typename T>
    inline const T& get() const
    {
        return mpark::get<T>(value_);
    }
    template <typename T>
    inline const T* get_if() const noexcept
    {
        return mpark::get_if<T>(&value_);
    }
    template <typename T>
    inline T* get_if() noexcept
    {
        return mpark::get_if<T>(&value_);
    }

    size_t index() const noexcept { return value_.index(); }

private:
    Variant value_;
};

struct DumpContext
{
    std::string path {"snapshot/"};
    MPI_Comm groupComm {MPI_COMM_NULL};
    std::map<std::string, int> counters;

    bool isGroupMasterTask() const;
};

class Dumper
{
public:
    Dumper(DumpContext context);
    ~Dumper();

    DumpContext& getContext() noexcept { return context_; }
    const Config& getConfig() const noexcept { return config_; }

    /// Dump.
    template <typename T>
    Config operator()(T& t)
    {
        return ConfigDumper<std::remove_const_t<T>>::dump(*this, t);
    }
    template <typename T>
    Config operator()(const T& t)
    {
        return ConfigDumper<T>::dump(*this, t);
    }
    template <typename T>
    Config operator()(T* t)
    {
        return ConfigDumper<std::remove_const_t<T>*>::dump(*this, t);
    }
    Config operator()(const char* t)
    {
        return std::string(t);
    }

    bool isObjectRegistered(const void*) const noexcept;
    const ConfigRefString& getObjectRefString(const void*) const;

    template <typename T>
    const ConfigRefString& registerObject(const T *obj, Config newItem)
    {
        assertType(obj);
        return _registerObject((const void *)obj, std::move(newItem));
    }

private:
    const ConfigRefString& _registerObject(const void *, Config newItem);

    Config config_;
    std::map<const void*, ConfigRefString> descriptions_;
    DumpContext context_;
};

class UndumpContext;
class Undumper
{
public:
    Undumper(UndumpContext *context) : context_(context) {}

    UndumpContext& getContext() noexcept { return *context_; }

    template <typename T>
    T undump(const Config &config)
    {
        return ConfigDumper<T>::undump(*this, config);
    }

private:
    UndumpContext *context_;
};

namespace detail
{
    struct DumpHandler
    {
        template <typename... Args>
        void process(Args&& ...items)
        {
            dict_->reserve(dict_->size() + sizeof...(items));

            // https://stackoverflow.com/a/51006031
            // Note: initializer list preserves the order of evaluation!
            using fold_expression = int[];
            (void)fold_expression{0, (dict_->insert(std::forward<Args>(items)), 0)...};
        }

        template <typename T>
        Config::Dictionary::value_type operator()(std::string name, T *t) const
        {
            return {std::move(name), (*dumper_)(*t)};
        }

        Config::Dictionary *dict_;
        Dumper *dumper_;
    };

    template <typename T>
    struct UndumpHandler {
        template <typename... Args>
        T process(Args ...items) const
        {
            return T{std::move(items)...};
        }

        template <typename Item>
        Item operator()(const std::string &name, const Item *) const
        {
            return un_->undump<Item>(dict_->at(name));
        }

        const Config::Dictionary *dict_;
        Undumper *un_;
    };
} // namespace detail

#define MIRHEO_DUMPER_PRIMITIVE(TYPE, ELTYPE)                                  \
    template <>                                                                \
    struct ConfigDumper<TYPE>                                                  \
    {                                                                          \
        static Config dump(Dumper&, TYPE x)                                    \
        {                                                                      \
            return static_cast<Config::ELTYPE>(x);                             \
        }                                                                      \
        static TYPE parse(const Config &value)                                 \
        {                                                                      \
            return static_cast<TYPE>(value.get##ELTYPE());                     \
        }                                                                      \
        static TYPE undump(Undumper&, const Config &value)                     \
        {                                                                      \
            return static_cast<TYPE>(value.get##ELTYPE());                     \
        }                                                                      \
    }
MIRHEO_DUMPER_PRIMITIVE(bool,               Int);
MIRHEO_DUMPER_PRIMITIVE(int,                Int);
MIRHEO_DUMPER_PRIMITIVE(long,               Int);
MIRHEO_DUMPER_PRIMITIVE(long long,          Int);
MIRHEO_DUMPER_PRIMITIVE(unsigned,           Int);
MIRHEO_DUMPER_PRIMITIVE(unsigned long,      Int);  // This is risky.
MIRHEO_DUMPER_PRIMITIVE(unsigned long long, Int);  // This is risky.
MIRHEO_DUMPER_PRIMITIVE(float,              Float);
MIRHEO_DUMPER_PRIMITIVE(double,             Float);
#undef MIRHEO_DUMPER_PRIMITIVE

template <>
struct ConfigDumper<const char*>
{
    static Config dump(Dumper&, const char *str)
    {
        return std::string(str);
    }
    static const char* parse(const Config&) = delete;
    static const char* undump(Undumper&, const Config&) = delete;
};

template <>
struct ConfigDumper<std::string>
{
    static Config dump(Dumper&, std::string x)
    {
        return std::move(x);
    }
    static const std::string& parse(const Config &config)
    {
        return config.getString();
    }
    static const std::string& undump(Undumper&, const Config &config)
    {
        return config.getString();
    }
};

template <>
struct ConfigDumper<float3>
{
    static Config dump(Dumper&, float3 v);
    static float3 parse(const Config &config);
    static float3 undump(Undumper&, const Config &config)
    {
        return parse(config);
    }
};

/// ConfigDumper for enum types.
template <typename T>
struct ConfigDumper<T, std::enable_if_t<std::is_enum<T>::value>>
{
    static Config dump(Dumper&, T t)
    {
        return static_cast<Config::Int>(t);
    }
    static T parse(const Config &config)
    {
        return static_cast<T>(config.getInt());
    }
    static T undump(Undumper&, const Config &config)
    {
        return parse(config);
    }
};

/// ConfigDumper for structs with reflection information.
template <typename T>
struct ConfigDumper<T, std::enable_if_t<MemberVarsAvailable<std::remove_const_t<T>>::value>>
{
    template <typename TT>  // Const or not.
    static Config dump(Dumper& dumper, TT& t)
    {
        Config::Dictionary dict;
        MemberVars<T>::foreach(detail::DumpHandler{&dict, &dumper}, &t);
        return std::move(dict);
    }
    static T undump(Undumper& un, const Config& config)
    {
        return MemberVars<T>::foreach(
                detail::UndumpHandler<T>{&config.getDict(), &un},
                (const T *)nullptr);
    }
};

/// ConfigDumper for pointer-like (dereferenceable) types. Redirects to the
/// underlying object if not nullptr, otherwise returns a "<nullptr>" string.
template <typename T>
struct ConfigDumper<T, std::enable_if_t<is_dereferenceable<T>::value>>
{
    static Config dump(Dumper& dumper, const T& ptr)
    {
        return ptr ? dumper(*ptr) : Config{ConfigNullRefString};
    }
};

/// ConfigDumper for std::vector<T>.
template <typename T>
struct ConfigDumper<std::vector<T>>
{
    template <typename Vector>  // Const or not.
    static Config dump(Dumper& dumper, Vector& values)
    {
        Config::List list;
        list.reserve(values.size());
        for (auto& value : values)
            list.push_back(dumper(value));
        return std::move(list);
    }
    static std::vector<T> undump(Undumper& un, const Config& config)
    {
        const Config::List& list = config.getList();
        std::vector<T> out;
        out.reserve(list.size());
        for (const Config& item : list)
            out.push_back(un.undump<T>(item));
        return out;
    }
};

/// ConfigDumper for std::map<std::string, T>.
template <typename T>
struct ConfigDumper<std::map<std::string, T>>
{
    template <typename Map>  // Const or not.
    static Config dump(Dumper& dumper, Map& values)
    {
        Config::Dictionary dict;
        dict.reserve(values.size());
        for (auto& pair : values)
            dict.unsafe_insert(pair.first, dumper(pair.second));
        return std::move(dict);
    }
    static std::map<std::string, T> undump(Undumper& un, const Config& config)
    {
        std::map<std::string, T> out;
        for (const auto& pair : config.getDict())
            out.emplace(pair.first, un.undump<T>(pair.second));
        return out;
    }
};

/// ConfigDumper for mpark::variant.
void _variantDumperError [[noreturn]] (size_t index, size_t size);
template <typename... Ts>
struct ConfigDumper<mpark::variant<Ts...>>
{
    using Variant = mpark::variant<Ts...>;

    template <typename T>
    static Variant _undump(Undumper& un, const Config& config)
    {
        return Variant{un.undump<T>(config)};
    }

    template <typename Variant>  // Const or not.
    static Config dump(Dumper& dumper, Variant& value)
    {
        Config::Dictionary dict;
        dict.reserve(2);
        dict.unsafe_insert("__index", static_cast<Config::Int>(value.index()));
        dict.unsafe_insert("value", mpark::visit(dumper, value));
        return dict;
    }
    static Variant undump(Undumper& un, const Config& config)
    {
        const ConfigDictionary& dict = config.getDict();
        size_t index = un.undump<size_t>(dict.at("__index"));
        if (index >= sizeof...(Ts))
            _variantDumperError(index, sizeof...(Ts));

        // Compile an array of _undump functions, one for each type.
        // Pick index-th on runtime.
        using UndumperPtr = Variant(*)(Undumper&, const Config&);
        const UndumperPtr funcs[]{(&_undump<Ts>)...};
        return funcs[index](un, dict.at("value"));
    }
};

Config configFromJSONFile(const std::string& filename);
Config configFromJSON(const std::string& json);

} // namespace mirheo
