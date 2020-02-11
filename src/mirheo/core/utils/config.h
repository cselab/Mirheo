#pragma once

#include "common.h" // Forward declarations of ConfigValue and TypeLoadSave<>.
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

/// Extract the object name from ConfigRefString.
std::string parseNameFromRefString(const ConfigRefString &ref);

/// Print the error about mismatched type and throw and exception.
void _typeMismatchError [[noreturn]] (const char *thisTypeName, const char *classTypeName);

/// Throw an exception if object's dynamic type is not `T`.
template <typename T>
void assertType(T *thisPtr)
{
    if (typeid(*thisPtr) != typeid(T))
        _typeMismatchError(typeid(*thisPtr).name(), typeid(T).name());
}

class Saver;
class Loader;

template <typename T, typename Enable>
struct TypeLoadSave
{
    static_assert(std::is_same<remove_cvref_t<T>, T>::value,
                  "Type must be a non-const non-reference type.");
    static_assert(always_false<T>::value, "Not implemented.");

    static ConfigValue save(Saver&, T& value);

    /// Context-free parsing. Only for simple types!
    static T parse(const ConfigValue&);

    /// Context-aware load.
    static T load(Loader&, const ConfigValue&);
};

class ConfigArray : public std::vector<ConfigValue>
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

/** Generic configuration JSON-like value class.

    It can represent integers, floats, strings, arrays and objects (dictionaries).
  */
class ConfigValue
{
public:
    using Int     = long long;
    using Float   = double;
    using String  = std::string;
    using Array    = ConfigArray;
    using Object  = ConfigObject;
    using Variant = mpark::variant<Int, Float, String, Array, Object>;

    ConfigValue(Int value) : value_{value} {}
    ConfigValue(Float value) : value_{value} {}
    ConfigValue(String value) : value_{std::move(value)} {}
    ConfigValue(Object value) : value_{std::move(value)} {}
    ConfigValue(Array value) : value_{std::move(value)} {}
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
            "for variant types (Int, Float, String, Object, Array). "
            "Did you mean `saver(value)` instead of `ConfigValue{value}`?");
    }

    std::string toJSONString() const;

    /// Getter functions. Terminate if the underlying type is different. Int
    /// and Float variants accept the other type if the conversion is lossless.
    Int getInt() const;
    Float getFloat() const;
    const String& getString() const;
    const Array& getArray() const;
    Array& getArray();
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

    /// Get the array element. Terminates if not an array or if out of range.
    ConfigValue&       operator[](size_t i)       { return getArray()[i]; }
    const ConfigValue& operator[](size_t i) const { return getArray()[i]; }
    ConfigValue&       operator[](int i)       { return getArray()[static_cast<size_t>(i)]; }
    const ConfigValue& operator[](int i) const { return getArray()[static_cast<size_t>(i)]; }

    /// Get the element if it exists, or nullptr otherwise. Terminates if not an object.
    ConfigValue*       get(const std::string &key) &      { return getObject().get(key); }
    const ConfigValue* get(const std::string &key) const& { return getObject().get(key); }
    ConfigValue*       get(const char *key) &             { return getObject().get(key); }
    const ConfigValue* get(const char *key) const&        { return getObject().get(key); }

    /// Implicit cast to simple types. Risky since it implicitly enables weird operator overloads.
    template <typename T>
    operator T() const { return TypeLoadSave<T>::parse(*this); }

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

class Saver
{
public:
    Saver(DumpContext context);
    ~Saver();

    DumpContext& getContext() noexcept { return context_; }
    const ConfigValue& getConfig() const noexcept { return config_; }

    /// Dump.
    template <typename T>
    ConfigValue operator()(T& t)
    {
        return TypeLoadSave<std::remove_const_t<T>>::save(*this, t);
    }
    template <typename T>
    ConfigValue operator()(const T& t)
    {
        return TypeLoadSave<T>::save(*this, t);
    }
    template <typename T>
    ConfigValue operator()(T* t)
    {
        return TypeLoadSave<std::remove_const_t<T>*>::save(*this, t);
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
    std::map<const void*, ConfigRefString> refStrings_;
    DumpContext context_;
};

class LoaderContext;
class Loader
{
public:
    Loader(LoaderContext *context) : context_(context) {}

    LoaderContext& getContext() noexcept { return *context_; }

    template <typename T>
    T load(const ConfigValue &config)
    {
        return TypeLoadSave<T>::load(*this, config);
    }

private:
    LoaderContext *context_;
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
            return {std::move(name), (*saver_)(*t)};
        }

        ConfigValue::Object *object_;
        Saver *saver_;
    };

    template <typename T>
    struct LoadHandler {
        template <typename... Args>
        T process(Args ...items) const
        {
            return T{std::move(items)...};
        }

        template <typename Item>
        Item operator()(const std::string &name, const Item *) const
        {
            return un_->load<Item>(object_->at(name));
        }

        const ConfigValue::Object *object_;
        Loader *un_;
    };
} // namespace detail

#define MIRHEO_DUMPER_PRIMITIVE(TYPE, ELTYPE)                                  \
    template <>                                                                \
    struct TypeLoadSave<TYPE>                                                  \
    {                                                                          \
        static ConfigValue save(Saver&, TYPE x)                                \
        {                                                                      \
            return static_cast<ConfigValue::ELTYPE>(x);                        \
        }                                                                      \
        static TYPE parse(const ConfigValue &value)                            \
        {                                                                      \
            return static_cast<TYPE>(value.get##ELTYPE());                     \
        }                                                                      \
        static TYPE load(Loader&, const ConfigValue &value)                    \
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
struct TypeLoadSave<const char*>
{
    static ConfigValue save(Saver&, const char *str)
    {
        return std::string(str);
    }
    static const char* parse(const ConfigValue&) = delete;
    static const char* load(Loader&, const ConfigValue&) = delete;
};

template <>
struct TypeLoadSave<std::string>
{
    static ConfigValue save(Saver&, std::string x)
    {
        return std::move(x);
    }
    static const std::string& parse(const ConfigValue &config)
    {
        return config.getString();
    }
    static const std::string& load(Loader&, const ConfigValue &config)
    {
        return config.getString();
    }
};

template <>
struct TypeLoadSave<float3>
{
    static ConfigValue save(Saver&, float3 v);
    static float3 parse(const ConfigValue &config);
    static float3 load(Loader&, const ConfigValue &config)
    {
        return parse(config);
    }
};

/// TypeLoadSave for enum types.
template <typename T>
struct TypeLoadSave<T, std::enable_if_t<std::is_enum<T>::value>>
{
    static ConfigValue save(Saver&, T t)
    {
        return static_cast<ConfigValue::Int>(t);
    }
    static T parse(const ConfigValue &config)
    {
        return static_cast<T>(config.getInt());
    }
    static T load(Loader&, const ConfigValue &config)
    {
        return parse(config);
    }
};

/// TypeLoadSave for structs with reflection information.
template <typename T>
struct TypeLoadSave<T, std::enable_if_t<MemberVarsAvailable<std::remove_const_t<T>>::value>>
{
    template <typename TT>  // Const or not.
    static ConfigValue save(Saver& saver, TT& t)
    {
        ConfigObject object;
        MemberVars<T>::foreach(detail::DumpHandler{&object, &saver}, &t);
        return std::move(object);
    }
    static T load(Loader& loader, const ConfigValue& config)
    {
        return MemberVars<T>::foreach(
                detail::LoadHandler<T>{&config.getObject(), &loader},
                (const T *)nullptr);
    }
};

/// TypeLoadSave for pointer-like (dereferenceable) types. Redirects to the
/// underlying object if not nullptr, otherwise returns a "<nullptr>" string.
template <typename T>
struct TypeLoadSave<T, std::enable_if_t<is_dereferenceable<T>::value>>
{
    static ConfigValue save(Saver& saver, const T& ptr)
    {
        return ptr ? saver(*ptr) : ConfigValue{ConfigNullRefString};
    }
};

/// TypeLoadSave for std::vector<T>.
template <typename T>
struct TypeLoadSave<std::vector<T>>
{
    template <typename Vector>  // Const or not.
    static ConfigValue save(Saver& saver, Vector& values)
    {
        ConfigValue::Array array;
        array.reserve(values.size());
        for (auto& value : values)
            array.push_back(saver(value));
        return std::move(array);
    }
    static std::vector<T> load(Loader& loader, const ConfigValue& config)
    {
        const ConfigValue::Array& array = config.getArray();
        std::vector<T> out;
        out.reserve(array.size());
        for (const ConfigValue& item : array)
            out.push_back(loader.load<T>(item));
        return out;
    }
};

/// TypeLoadSave for std::map<std::string, T>.
template <typename T>
struct TypeLoadSave<std::map<std::string, T>>
{
    template <typename Map>  // Const or not.
    static ConfigValue save(Saver& saver, Map& values)
    {
        ConfigValue::Object object;
        object.reserve(values.size());
        for (auto& pair : values)
            object.unsafe_insert(pair.first, saver(pair.second));
        return std::move(object);
    }
    static std::map<std::string, T> load(Loader& loader, const ConfigValue& config)
    {
        std::map<std::string, T> out;
        for (const auto& pair : config.getObject())
            out.emplace(pair.first, loader.load<T>(pair.second));
        return out;
    }
};

/// TypeLoadSave for mpark::variant.
void _variantDumperError [[noreturn]] (size_t index, size_t size);
template <typename... Ts>
struct TypeLoadSave<mpark::variant<Ts...>>
{
    using Variant = mpark::variant<Ts...>;

    template <typename T>
    static Variant _load(Loader& loader, const ConfigValue& config)
    {
        return Variant{loader.load<T>(config)};
    }

    template <typename Variant>  // Const or not.
    static ConfigValue save(Saver& saver, Variant& value)
    {
        ConfigValue::Object object;
        object.reserve(2);
        object.unsafe_insert("__index", static_cast<ConfigValue::Int>(value.index()));
        object.unsafe_insert("value", mpark::visit(saver, value));
        return object;
    }
    static Variant load(Loader& loader, const ConfigValue& config)
    {
        const ConfigObject& object = config.getObject();
        size_t index = loader.load<size_t>(object.at("__index"));
        if (index >= sizeof...(Ts))
            _variantDumperError(index, sizeof...(Ts));

        // Compile an array of _load functions, one for each type.
        // Pick index-th on runtime.
        using LoaderPtr = Variant(*)(Loader&, const ConfigValue&);
        const LoaderPtr funcs[]{(&_load<Ts>)...};
        return funcs[index](loader, object.at("value"));
    }
};

ConfigValue configFromJSONFile(const std::string& filename);
ConfigValue configFromJSON(const std::string& json);

} // namespace mirheo
