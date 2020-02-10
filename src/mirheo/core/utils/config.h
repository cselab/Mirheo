#pragma once

#include "common.h" // Forward declarations of ConfigValue and ConfigDumper<>.
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

/// Reference string used for null pointers.
constexpr const char ConfigNullRefString[] = "<nullptr>";

/// Print the error about mismatched type and throw and exception.
void _typeMismatchError [[noreturn]] (const char *thisTypeName, const char *classTypeName);

/// Throw an exception if pointer's dynamic type is not `T`.
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

    static ConfigValue dump(Dumper&, T& value);

    /// Context-free parsing. Only for simple types!
    static T parse(const ConfigValue&);

    /// Context-aware undump.
    static T undump(Undumper&, const ConfigValue&);
};

class ConfigList : public std::vector<ConfigValue>
{
    using Base = std::vector<ConfigValue>;
public:
    using Base::Base;

    /// Overwrite operator[] with bound checks.
    ConfigValue&       operator[](size_t i)       { return at(i); }
    const ConfigValue& operator[](size_t i) const { return at(i); }

    ConfigValue& at(size_t i) {
        return i < size() ? Base::operator[](i) : _outOfBound(i, size());
    }
    const ConfigValue& at(size_t i) const {
        return i < size() ? Base::operator[](i) : (const ConfigValue&)_outOfBound(i, size());
    }

private:
    ConfigValue& _outOfBound [[noreturn]] (size_t index, size_t size) const;
};

class ConfigObject : public FlatOrderedDict<std::string, ConfigValue>
{
    using Base = FlatOrderedDict<std::string, ConfigValue>;

public:
    using Base::Base;

    /// Overwrite operator[] with bound checks.
    ConfigValue&       operator[](const std::string &key)       { return at(key); }
    const ConfigValue& operator[](const std::string &key) const { return at(key); }
    ConfigValue&       operator[](const char *key)              { return at(key); }
    const ConfigValue& operator[](const char *key) const        { return at(key); }

    ConfigValue&       at(const std::string &key);
    const ConfigValue& at(const std::string &key) const;
    ConfigValue&       at(const char *key);
    const ConfigValue& at(const char *key) const;

    /// Get the pointer to the key if it exists, otherwise return a nullptr.
    ConfigValue*       get(const std::string &key) &;
    const ConfigValue* get(const std::string &key) const&;
    ConfigValue*       get(const char *key) &;
    const ConfigValue* get(const char *key) const&;
};

class ConfigValue
{
public:
    using Int     = long long;
    using Float   = double;
    using String  = std::string;
    using List    = ConfigList;
    using Object  = ConfigObject;
    using Variant = mpark::variant<Int, Float, String, List, Object>;

    ConfigValue(Int value) : value_{value} {}
    ConfigValue(Float value) : value_{value} {}
    ConfigValue(String value) : value_{std::move(value)} {}
    ConfigValue(Object value) : value_{std::move(value)} {}
    ConfigValue(List value) : value_{std::move(value)} {}
    ConfigValue(const char *str) : value_{std::string(str)} {}
    ConfigValue(const ConfigValue&) = default;
    ConfigValue(ConfigValue&&)      = default;
    ConfigValue& operator=(const ConfigValue&) = default;
    ConfigValue& operator=(ConfigValue&&) = default;

    template <typename T>
    ConfigValue(const T&)
    {
        static_assert(
            always_false<T>::value,
            "Direct construction of the ConfigValue object available only "
            "for variant types (Int, Float, String, Object, List). "
            "Did you mean `dumper(value)` instead of `ConfigValue{value}`?");
    }

    std::string toJSONString() const;

    /// Getter functions. Terminate if the underlying type is different. Int
    /// and Float variants accept the other type if the conversion is lossless.
    Int getInt() const;
    Float getFloat() const;
    const String& getString() const;
    const List& getList() const;
    List& getList();
    const Object& getObject() const;
    Object& getObject();

    /// Check if the key exists. Terminates if not an object.
    bool contains(const std::string &key) const { return getObject().contains(key); }
    bool contains(const char *key)        const { return getObject().contains(key); }

    /// Get the element matching the given key. Terminates if not an object, or
    /// if the key was not found.
    ConfigValue&       operator[](const std::string &key)       { return getObject().at(key); }
    const ConfigValue& operator[](const std::string &key) const { return getObject().at(key); }
    ConfigValue&       operator[](const char *key)              { return getObject().at(key); }
    const ConfigValue& operator[](const char *key) const        { return getObject().at(key); }

    /// Get the list element. Terminates if not a list or if out of range.
    ConfigValue&       operator[](size_t i)       { return getList()[i]; }
    const ConfigValue& operator[](size_t i) const { return getList()[i]; }
    ConfigValue&       operator[](int i)       { return getList()[static_cast<size_t>(i)]; }
    const ConfigValue& operator[](int i) const { return getList()[static_cast<size_t>(i)]; }

    /// Get the element if it exists, or null otherwise. Terminates if not an object.
    ConfigValue*       get(const std::string &key) &      { return getObject().get(key); }
    const ConfigValue* get(const std::string &key) const& { return getObject().get(key); }
    ConfigValue*       get(const char *key) &             { return getObject().get(key); }
    const ConfigValue* get(const char *key) const&        { return getObject().get(key); }

    /// Implicit cast to simple types.
    template <typename T>
    operator T() const { return ConfigDumper<T>::parse(*this); }

    /// Implicit cast to specific types.
    operator ConfigValue::Int() const { return getInt(); }
    operator ConfigValue::Float() const { return getFloat(); }
    operator const std::string&() const { return getString(); }

    /// String concatenation operator.
    friend std::string operator+(const ConfigValue& a, const char *b)
    {
        return a.getString() + b;
    }
    friend std::string operator+(const char *a, const ConfigValue& b)
    {
        return a + b.getString();
    }
    friend std::string operator+(const ConfigValue& a, const std::string& b)
    {
        return a.getString() + b;
    }
    friend std::string operator+(const std::string& a, const ConfigValue& b)
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
    const ConfigValue& getConfig() const noexcept { return config_; }

    /// Dump.
    template <typename T>
    ConfigValue operator()(T& t)
    {
        return ConfigDumper<std::remove_const_t<T>>::dump(*this, t);
    }
    template <typename T>
    ConfigValue operator()(const T& t)
    {
        return ConfigDumper<T>::dump(*this, t);
    }
    template <typename T>
    ConfigValue operator()(T* t)
    {
        return ConfigDumper<std::remove_const_t<T>*>::dump(*this, t);
    }
    ConfigValue operator()(const char* t)
    {
        return std::string(t);
    }

    bool isObjectRegistered(const void*) const noexcept;
    const ConfigRefString& getObjectRefString(const void*) const;

    template <typename T>
    const ConfigRefString& registerObject(const T *obj, ConfigValue newItem)
    {
        assertType(obj);
        return _registerObject((const void *)obj, std::move(newItem));
    }

private:
    const ConfigRefString& _registerObject(const void *, ConfigValue newItem);

    ConfigValue config_;
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
    T undump(const ConfigValue &config)
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
            object_->reserve(object_->size() + sizeof...(items));

            // https://stackoverflow.com/a/51006031
            // Note: initializer list preserves the order of evaluation!
            using fold_expression = int[];
            (void)fold_expression{0, (object_->insert(std::forward<Args>(items)), 0)...};
        }

        template <typename T>
        ConfigValue::Object::value_type operator()(std::string name, T *t) const
        {
            return {std::move(name), (*dumper_)(*t)};
        }

        ConfigValue::Object *object_;
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
            return un_->undump<Item>(object_->at(name));
        }

        const ConfigValue::Object *object_;
        Undumper *un_;
    };
} // namespace detail

#define MIRHEO_DUMPER_PRIMITIVE(TYPE, ELTYPE)                                  \
    template <>                                                                \
    struct ConfigDumper<TYPE>                                                  \
    {                                                                          \
        static ConfigValue dump(Dumper&, TYPE x)                               \
        {                                                                      \
            return static_cast<ConfigValue::ELTYPE>(x);                        \
        }                                                                      \
        static TYPE parse(const ConfigValue &value)                            \
        {                                                                      \
            return static_cast<TYPE>(value.get##ELTYPE());                     \
        }                                                                      \
        static TYPE undump(Undumper&, const ConfigValue &value)                \
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
    static ConfigValue dump(Dumper&, const char *str)
    {
        return std::string(str);
    }
    static const char* parse(const ConfigValue&) = delete;
    static const char* undump(Undumper&, const ConfigValue&) = delete;
};

template <>
struct ConfigDumper<std::string>
{
    static ConfigValue dump(Dumper&, std::string x)
    {
        return std::move(x);
    }
    static const std::string& parse(const ConfigValue &config)
    {
        return config.getString();
    }
    static const std::string& undump(Undumper&, const ConfigValue &config)
    {
        return config.getString();
    }
};

template <>
struct ConfigDumper<float3>
{
    static ConfigValue dump(Dumper&, float3 v);
    static float3 parse(const ConfigValue &config);
    static float3 undump(Undumper&, const ConfigValue &config)
    {
        return parse(config);
    }
};

/// ConfigDumper for enum types.
template <typename T>
struct ConfigDumper<T, std::enable_if_t<std::is_enum<T>::value>>
{
    static ConfigValue dump(Dumper&, T t)
    {
        return static_cast<ConfigValue::Int>(t);
    }
    static T parse(const ConfigValue &config)
    {
        return static_cast<T>(config.getInt());
    }
    static T undump(Undumper&, const ConfigValue &config)
    {
        return parse(config);
    }
};

/// ConfigDumper for structs with reflection information.
template <typename T>
struct ConfigDumper<T, std::enable_if_t<MemberVarsAvailable<std::remove_const_t<T>>::value>>
{
    template <typename TT>  // Const or not.
    static ConfigValue dump(Dumper& dumper, TT& t)
    {
        ConfigValue::Object object;
        MemberVars<T>::foreach(detail::DumpHandler{&object, &dumper}, &t);
        return std::move(object);
    }
    static T undump(Undumper& un, const ConfigValue& config)
    {
        return MemberVars<T>::foreach(
                detail::UndumpHandler<T>{&config.getObject(), &un},
                (const T *)nullptr);
    }
};

/// ConfigDumper for pointer-like (dereferenceable) types. Redirects to the
/// underlying object if not nullptr, otherwise returns a "<nullptr>" string.
template <typename T>
struct ConfigDumper<T, std::enable_if_t<is_dereferenceable<T>::value>>
{
    static ConfigValue dump(Dumper& dumper, const T& ptr)
    {
        return ptr ? dumper(*ptr) : ConfigValue{ConfigNullRefString};
    }
};

/// ConfigDumper for std::vector<T>.
template <typename T>
struct ConfigDumper<std::vector<T>>
{
    template <typename Vector>  // Const or not.
    static ConfigValue dump(Dumper& dumper, Vector& values)
    {
        ConfigValue::List list;
        list.reserve(values.size());
        for (auto& value : values)
            list.push_back(dumper(value));
        return std::move(list);
    }
    static std::vector<T> undump(Undumper& un, const ConfigValue& config)
    {
        const ConfigValue::List& list = config.getList();
        std::vector<T> out;
        out.reserve(list.size());
        for (const ConfigValue& item : list)
            out.push_back(un.undump<T>(item));
        return out;
    }
};

/// ConfigDumper for std::map<std::string, T>.
template <typename T>
struct ConfigDumper<std::map<std::string, T>>
{
    template <typename Map>  // Const or not.
    static ConfigValue dump(Dumper& dumper, Map& values)
    {
        ConfigValue::Object object;
        object.reserve(values.size());
        for (auto& pair : values)
            object.unsafe_insert(pair.first, dumper(pair.second));
        return std::move(object);
    }
    static std::map<std::string, T> undump(Undumper& un, const ConfigValue& config)
    {
        std::map<std::string, T> out;
        for (const auto& pair : config.getObject())
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
    static Variant _undump(Undumper& un, const ConfigValue& config)
    {
        return Variant{un.undump<T>(config)};
    }

    template <typename Variant>  // Const or not.
    static ConfigValue dump(Dumper& dumper, Variant& value)
    {
        ConfigValue::Object object;
        object.reserve(2);
        object.unsafe_insert("__index", static_cast<ConfigValue::Int>(value.index()));
        object.unsafe_insert("value", mpark::visit(dumper, value));
        return object;
    }
    static Variant undump(Undumper& un, const ConfigValue& config)
    {
        const ConfigObject& object = config.getObject();
        size_t index = un.undump<size_t>(object.at("__index"));
        if (index >= sizeof...(Ts))
            _variantDumperError(index, sizeof...(Ts));

        // Compile an array of _undump functions, one for each type.
        // Pick index-th on runtime.
        using UndumperPtr = Variant(*)(Undumper&, const ConfigValue&);
        const UndumperPtr funcs[]{(&_undump<Ts>)...};
        return funcs[index](un, object.at("value"));
    }
};

ConfigValue configFromJSONFile(const std::string& filename);
ConfigValue configFromJSON(const std::string& json);

} // namespace mirheo
