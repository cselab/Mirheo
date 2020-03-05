#include "config.h"
#include "strprintf.h"
#include <mirheo/core/logger.h>

#include <cassert>
#include <sstream>

/* Implementation notice:
 *
 * Currently we manually implement a JSON representation, exporting and importing.
 * We considered using 3rd party libraries, but they were not desirable for
 * various reasons. The most promising one seems to be
 * https://github.com/Tencent/rapidjson
 * due to its very high type safety (e.g. differentiates between signed and
 * unsigned integers).
 *
 * The problem is its compilation procedure, based on old-style CMake. The
 * implementation below and for the whole ConfigValue class can be replaced
 * with this or some other JSON library if needed and found suitable.
 */

namespace mirheo
{

std::string parseNameFromRefString(const ConfigRefString &ref)
{
    // Format: "<TYPENAME with name=NAME>".
    size_t pos = ref.find("with name=");
    if (pos == std::string::npos)
        die("Unrecognized or unnamed reference format: %s", ref.c_str());
    pos += 4 + 1 + 5;
    return ref.substr(pos, ref.size() - pos - 1);
}

/// Create a string that refers to an object located elsewhere in the JSON file.
static inline ConfigRefString createRefString(const char *typeName, const char *objectName)
{
    return objectName ? strprintf("<%s with name=%s>", typeName, objectName)
                      : strprintf("<%s>", typeName);
}

std::string readWholeFile(const std::string& filename)
{
    FileWrapper file(filename, "r");

    // https://stackoverflow.com/questions/14002954/c-programming-how-to-read-the-whole-file-contents-into-a-buffer
    fseek(file.get(), 0, SEEK_END);
    long size = ftell(file.get());
    fseek(file.get(), 0, SEEK_SET);  /* same as rewind(f); */

    std::string output(size, '_');
    fread(&output[0], 1, size, file.get());
    return output;
}

void writeToFile(const std::string& filename, const std::string& content)
{
    FileWrapper f(filename, "w");
    fwrite(content.data(), 1, content.size(), f.get());
}


void _typeMismatchError [[noreturn]] (const char *thisTypeName, const char *classTypeName)
{
    die("Missing implementation of a virtual member function. Var type=%s class type=%s",
        thisTypeName, classTypeName);
}

void _typeLoadSaveNotImplemented [[noreturn]] (const char *func, const char *typeName)
{
    die("Type %s does not support loading or saving. Invoked function: %s",
        typeName, func);
}

static std::string doubleToString(double x)
{
    char str[32];
    sprintf(str, "%.17g", x);
    return str;
}

static std::string stringToJSON(const std::string& input)
{
    std::string output;
    output.reserve(2 + input.size());
    output += '"';
    for (char c : input) {
        switch (c) {
        case '"': output += "\\\""; break;
        case '\b': output += "\\b"; break;
        case '\f': output += "\\f"; break;
        case '\n': output += "\\n"; break;
        case '\r': output += "\\r"; break;
        case '\t': output += "\\t"; break;
        case '\\': output += "\\\\"; break;
        default:
            output += c;
        }
    }
    output += '"';
    return output;
}

namespace
{
    class ConfigExporter
    {
    public:
        enum class Tag
        {
            StartObject,
            EndObject,
            StartArray,
            EndArray,
            StartObjectItem,
            EndObjectItem,
            StartArrayItem,
            EndArrayItem,
            Dummy
        };

        // For simplicity, we merge Int, Float and String tokens into std::string.
        using Token = mpark::variant<std::string, Tag>;

        void process(const ConfigValue& element);
        std::string exportJSON();

    private:
        std::vector<Token> tokens_;
    };
} // anonymous namespace

void ConfigExporter::process(const ConfigValue& element)
{
    if (auto *ll = element.get_if<long long>()) {
        tokens_.push_back(std::to_string(*ll));
    } else if (auto *d = element.get_if<double>()) {
        tokens_.push_back(doubleToString(*d));
    } else if (auto *s = element.get_if<std::string>()) {
        tokens_.push_back(stringToJSON(*s));
    } else if (auto *obj = element.get_if<ConfigValue::Object>()) {
        tokens_.push_back(Tag::StartObject);
        for (const auto &pair : *obj) {
            tokens_.push_back(Tag::StartObjectItem);
            tokens_.push_back(stringToJSON(pair.first));
            process(pair.second);
            tokens_.push_back(Tag::EndObjectItem);
        }
        tokens_.push_back(Tag::EndObject);
    } else if (auto *array = element.get_if<ConfigValue::Array>()) {
        tokens_.push_back(Tag::StartArray);
        for (const ConfigValue& el : *array) {
            tokens_.push_back(Tag::StartArrayItem);
            process(el);
            tokens_.push_back(Tag::EndArrayItem);
        }
        tokens_.push_back(Tag::EndArray);
    } else {
        assert(false);
    }
}

std::string ConfigExporter::exportJSON()
{
    std::ostringstream stream;
    std::string nlindent {'\n'};

    enum class ObjectType { Object, Array };

    auto push = [&]() { nlindent += "    "; };
    auto pop  = [&]() { nlindent.erase(nlindent.size() - 4); };

    size_t numTokens = tokens_.size();
    tokens_.push_back("dummy");
    for (size_t i = 0; i < numTokens; ++i) {
        const Token& token     = tokens_[i];
        const Token& nextToken = tokens_[i + 1];
        if (auto *s = mpark::get_if<std::string>(&token)) {
            stream << *s;
            continue;
        }
        Tag tag = mpark::get<Tag>(token);
        Tag nextTag;
        if (const Tag *_nextTag = mpark::get_if<Tag>(&nextToken))
            nextTag = *_nextTag;
        else
            nextTag = Tag::Dummy;

        switch (tag) {
        case Tag::StartObject:
            if (nextTag == Tag::EndObject) {
                stream << "{}";
                ++i;
                break;
            }
            stream << '{';
            push();
            break;
        case Tag::EndObject:
            pop();
            stream << nlindent << '}';
            break;
        case Tag::StartArray:
            if (nextTag == Tag::EndArray) {
                stream << "[]";
                ++i;
                break;
            }
            stream << '[';
            push();
            break;
        case Tag::EndArray:
            pop();
            stream << nlindent << ']';
            break;
        case Tag::StartObjectItem:
            stream << nlindent;
            stream << mpark::get<std::string>(nextToken); // Key.
            stream << ": ";
            ++i;
            break;
        case Tag::StartArrayItem:
            stream << nlindent;
            break;
        case Tag::EndObjectItem:
        case Tag::EndArrayItem:
            if (nextTag == Tag::EndObject || nextTag == Tag::EndArray)
                break;
            stream << ',';
            break;
        default:
            assert(false);
        }
    }
    return std::move(stream).str();
}

ConfigValue& ConfigObject::at(const std::string &key)
{
    auto it = find(key);
    if (it == end())
        die("Key \"%s\" not found in %s", key.c_str(), ConfigValue{*this}.toJSONString().c_str());
    return it->second;
}
const ConfigValue& ConfigObject::at(const std::string &key) const
{
    auto it = find(key);
    if (it == end())
        die("Key \"%s\" not found in %s", key.c_str(), ConfigValue{*this}.toJSONString().c_str());
    return it->second;
}
ConfigValue& ConfigObject::at(const char *key)
{
    return at(std::string(key));
}
const ConfigValue& ConfigObject::at(const char *key) const
{
    return at(std::string(key));
}

ConfigValue* ConfigObject::get(const std::string &key) &
{
    auto it = find(key);
    return it != end() ? &it->second : nullptr;
}
const ConfigValue* ConfigObject::get(const std::string &key) const&
{
    auto it = find(key);
    return it != end() ? &it->second : nullptr;
}
ConfigValue* ConfigObject::get(const char *key) &
{
    return get(std::string(key));
}
const ConfigValue* ConfigObject::get(const char *key) const&
{
    return get(std::string(key));
}


std::string ConfigValue::toJSONString() const
{
    ConfigExporter writer;
    writer.process(*this);
    return writer.exportJSON();
}

std::string ConfigValue::toString() const
{
    if (auto *v = get_if<Int>())
        return std::to_string(*v);
    if (auto *v = get_if<Float>())
        return doubleToString(*v);
    if (auto *v = get_if<String>())
        return *v;

    std::stringstream stream;
    if (auto *v = get_if<Array>()) {
        stream << '[';
        for (size_t i = 0; i < v->size(); ++i) {
            if (i > 0)
                stream << ", ";
            stream << (*v)[i].toJSONString();
        }
        stream << ']';
    } else if (auto *obj = get_if<Object>()) {
        stream << '{';
        for (auto it = obj->begin(); it != obj->end(); ++it) {
            if (it != obj->begin())
                stream << ", ";
            stream << stringToJSON(it->first) << ": " << it->second.toJSONString();
        }
        stream << '}';
    } else {
        die("Invalid ConfigValue.");
    }
    return std::move(stream).str();
}

ConfigValue::Int ConfigValue::getInt() const
{
    if (const Int *v = get_if<Int>())
        return *v;
    if (const Float *v = get_if<Float>()) {
        if ((Float)(Int)*v == *v)  // Accept only if lossless.
            return (Int)*v;
    }
    die("getInt on a non-int object:\n%s", toJSONString().c_str());
}

ConfigValue::Float ConfigValue::getFloat() const
{
    if (const Float *v = get_if<Float>())
        return *v;
    if (const Int *v = get_if<Int>())
        if ((Int)(Float)*v == *v)
            return static_cast<double>(*v);
    die("getFloat on a non-float object:\n%s", toJSONString().c_str());
}

const ConfigValue::String& ConfigValue::getString() const {
    if (const String *v = get_if<String>())
        return *v;
    die("getString on a non-string object:\n%s", toJSONString().c_str());
}

const ConfigValue::Array& ConfigValue::getArray() const
{
    if (auto *array = get_if<Array>())
        return *array;
    die("getArray on a non-array object:\n%s", toJSONString().c_str());
}

ConfigValue::Array& ConfigValue::getArray()
{
    if (auto *array = get_if<Array>())
        return *array;
    die("getArray on a non-array object:\n%s", toJSONString().c_str());
}

const ConfigValue::Object& ConfigValue::getObject() const &
{
    if (auto *obj = get_if<Object>())
        return *obj;
    die("getObject on a non-dictionary object:\n%s", toJSONString().c_str());
}

ConfigValue::Object& ConfigValue::getObject() &
{
    if (auto *obj = get_if<Object>())
        return *obj;
    die("getObject on a non-dictionary object:\n%s", toJSONString().c_str());
}

ConfigValue::Object ConfigValue::getObject() &&
{
    if (auto *obj = get_if<Object>())
        return std::move(*obj);
    die("getObject on a non-dictionary object:\n%s", toJSONString().c_str());
}

ConfigValue& ConfigArray::_outOfBound [[noreturn]] (size_t index) const
{
    die("Index %zu out of range (size=%zu):\n%s",
        index, size(), ConfigValue{*this}.toJSONString().c_str());
}


Saver::Saver(SaverContext *context) : config_{ConfigObject{}}, context_{context}
{}

Saver::~Saver() = default;

bool Saver::isObjectRegistered(const void *ptr) const noexcept
{
    return refStrings_.find(ptr) != refStrings_.end();
}
const std::string& Saver::getObjectRefString(const void *ptr) const
{
    if (!isObjectRegistered(ptr)) {
        die("Object %p not registered. Could be the pointer was shifted due to "
            "multiple inheritance. Specialize TypeLoadSave for this type.", ptr);
    }
    return refStrings_.find(ptr)->second;
}

const std::string& Saver::_registerObject(const void *ptr, ConfigValue object)
{
    if (isObjectRegistered(ptr))
        die("Object %p already registered.", ptr);

    auto *newObject = object.get_if<ConfigValue::Object>();
    if (newObject == nullptr)
        die("Expected a dictionary, instead got:\n%s", object.toJSONString().c_str());

    // Get the category name and remove it from the dictionary.
    auto itCategory = newObject->find("__category");
    if (itCategory == newObject->end()) {
        die("Key \"%s\" not found in the config:\n%s",
            "__category", object.toJSONString().c_str());
    }
    std::string category = std::move(itCategory)->second.getString();
    newObject->erase(itCategory);

    // Find the category in the master object. Add an empty array if not found.
    auto it = config_.find(category);
    if (it == config_.end())
        it = config_.emplace(category, ConfigValue::Array{}).first;

    // Get the object name, if it exists.
    auto itName = newObject->find("name");
    const char *name =
        itName != newObject->end() ? itName->second.getString().c_str() : nullptr;

    // Get the object type.
    auto itType = newObject->find("__type");
    if (itType == newObject->end()) {
        die("Key \"%s\" not found in the config:\n%s",
            "__type", object.toJSONString().c_str());
    }

    // Genreate the refstring before moving the object, just in case.
    const char *type = itType->second.getString().c_str();
    ConfigRefString ref = createRefString(type, name);
    it->second.getArray().emplace_back(std::move(object));

    return refStrings_.emplace(ptr, std::move(ref)).first->second;
}

ConfigValue TypeLoadSave<float2>::save(Saver&, float2 v)
{
    return ConfigValue::Array{(double)v.x, (double)v.y};
}
float2 TypeLoadSave<float2>::parse(const ConfigValue &config)
{
    const auto& array = config.getArray();
    if (array.size() != 2)
        die("Expected 2 elements, got %zu.", array.size());
    return float2{(float)array[0], (float)array[1]};
}

ConfigValue TypeLoadSave<float3>::save(Saver&, float3 v)
{
    return ConfigValue::Array{(double)v.x, (double)v.y, (double)v.z};
}
float3 TypeLoadSave<float3>::parse(const ConfigValue &config)
{
    const auto& array = config.getArray();
    if (array.size() != 3)
        die("Expected 3 elements, got %zu.", array.size());
    return float3{(float)array[0], (float)array[1], (float)array[2]};
}

ConfigValue TypeLoadSave<double2>::save(Saver&, double2 v)
{
    return ConfigValue::Array{v.x, v.y};
}
double2 TypeLoadSave<double2>::parse(const ConfigValue &config)
{
    const auto& array = config.getArray();
    if (array.size() != 2)
        die("Expected 2 elements, got %zu.", array.size());
    return double2{(double)array[0], (double)array[1]};
}

ConfigValue TypeLoadSave<double3>::save(Saver&, double3 v)
{
    return ConfigValue::Array{v.x, v.y, v.z};
}
double3 TypeLoadSave<double3>::parse(const ConfigValue &config)
{
    const auto& array = config.getArray();
    if (array.size() != 3)
        die("Expected 3 elements, got %zu.", array.size());
    return double3{(double)array[0], (double)array[1], (double)array[2]};
}

void _variantLoadIndexError [[noreturn]] (size_t index, size_t size)
{
    die("Variant index %zu out of range (size=%zu).", index, size);
}
ConfigValue _variantSave(ConfigValue::Int index, ConfigValue value)
{
    ConfigValue config{ConfigObject{}};
    ConfigObject *object = config.get_if<ConfigObject>();
    object->reserve(2);
    object->unsafe_insert("__index", index);
    object->unsafe_insert("value", std::move(value));
    return config;
}


namespace {
    /**
     * Parse JSON and return a ConfigValue.
     *
     * Usage:
     *      ConfigValue config = JSonParser{"[10, 20, 30.5"]}.parse();
     */
    class JSONParser {
    public:
        JSONParser(const char *s) : str_(s) {}

        ConfigValue parse() {
            Token token = nextToken(TT::AnyValue);
            if (token == TT::OpenBrace) {
                ConfigValue::Object obj;
                for (;;) {
                    if (peekToken(TT::AnyValue, TT::ClosedBrace) == TT::ClosedBrace)
                        break;
                    std::string key = mpark::get<ConfigValue::String>(nextToken(TT::String).value);
                    nextToken(TT::Colon);
                    obj.emplace(key, parse());
                    if (peekToken(TT::Comma, TT::ClosedBrace) == TT::Comma) {
                        nextToken(TT::Comma);  // Consume.
                        continue;
                    }
                }
                nextToken(TT::ClosedBrace);  // Consume.
                return ConfigValue{std::move(obj)};
            } else if (token == TT::OpenSquare) {
                ConfigValue::Array array;
                for (;;) {
                    if (peekToken(TT::AnyValue, TT::ClosedSquare) == TT::ClosedSquare)
                        break;
                    array.push_back(parse());
                    if (peekToken(TT::ClosedSquare, TT::Comma) == TT::Comma) {
                        nextToken(TT::Comma);  // Consume.
                        continue;
                    }
                }
                nextToken(TT::ClosedSquare);  // Consume.
                return ConfigValue{std::move(array)};
            } else if (token == TT::Int) {
                return mpark::get<ConfigValue::Int>(token.value);
            } else if (token == TT::Float) {
                return mpark::get<ConfigValue::Float>(token.value);
            } else if (token == TT::String) {
                return std::move(mpark::get<ConfigValue::String>(token.value));
            }
            die("Unreachable.");
        }

    private:
        enum class TT {  // TokenType
            Unspecified = 0,
            OpenBrace = 1,
            ClosedBrace = 2,
            OpenSquare = 4,
            ClosedSquare = 8,
            Comma = 16,
            Colon = 32,
            String = 64,
            Int = 128,    // long long
            Float = 256,  // double

            AnyValue = OpenBrace | OpenSquare | String | Int | Float,
        };

        struct Token {
            mpark::variant<ConfigValue::Int, ConfigValue::Float, ConfigValue::String> value {ConfigValue::Int{0}};
            TT type {TT::Unspecified};

            Token(TT t) : type{t} {}
            Token(long long v) : value{v}, type{TT::Int} {}
            Token(double v) : value{v}, type{TT::Float} {}
            Token(std::string v) : value{std::move(v)}, type{TT::String} {}

            inline bool operator==(TT other) const noexcept {
                return type == other;
            }
        };

        /// Look at the next token without consuming it, and check that its
        /// type matches one of the given masks.
        template <typename... TT_>
        const Token& peekToken(TT_ ...options)
        {
            int tt = 0;
            // https://stackoverflow.com/a/51006031
            using fold_expression = int[];
            (void)fold_expression{0, (tt |= static_cast<int>(options))...};
            return _peekToken(tt);
        }

        const Token& _peekToken(int restriction) {
            if (currentToken_ == TT::Unspecified)
                currentToken_ = _readToken();
            if (((int)currentToken_.type & restriction) == 0) {
                die("Token type=%d  restriction=%d:\n%s",
                (int)currentToken_.type, restriction, str_);
            }
            return currentToken_;
        }

        /// Consume the token and return it. Check if its type matches the given mask.
        Token nextToken(TT restriction) {
            _peekToken(static_cast<int>(restriction));
            Token out = std::move(currentToken_);
            currentToken_ = TT::Unspecified;
            return out;
        }

        /// Parse next token.
        Token _readToken() {
            for (;;) {
                char c = *str_;  // Peek.
                switch (c) {
                    case '\0':
                        die("Unexpected end of json.");
                    case '\t':
                    case '\n':
                    case '\r':
                    case ' ': ++str_; continue;
                    case '{': return (++str_, TT::OpenBrace);
                    case '}': return (++str_, TT::ClosedBrace);
                    case '[': return (++str_, TT::OpenSquare);
                    case ']': return (++str_, TT::ClosedSquare);
                    case ',': return (++str_, TT::Comma);
                    case ':': return (++str_, TT::Colon);
                    case '\"': return _readString();
                }
                if ((c >= '0' && c <= '9') || c == '-')
                    return _readNumber();
                die("Unexpected character [%c] at:\n%s", c, str_);
            }
        }

        /// Parse next string.
        Token _readString() {
            std::string out;
            assert(*str_ == '"');
            ++str_;
            for (;;) {
                char c = *str_++;
                if (c == '\0')
                    die("Unexpected end of string.");
                if (c == '\\') {
                    char d = *str_++;
                    switch (d) {
                        case '"': d = '"'; break;
                        case 'b': d = '\b'; break;
                        case 'f': d = '\f'; break;
                        case 'n': d = '\n'; break;
                        case 'r': d = '\r'; break;
                        case 't': d = '\t'; break;
                        case '\\': d = '\\'; break;
                        default:
                            die("Unexpected escape character [\\%c].", d);
                    }
                    out.push_back(d);
                } else if (c == '"') {
                    break;
                } else {
                    out.push_back(c);
                }
            }
            return out;
        }

        /// Parse an integer or a float. Currently, this function will
        /// incorrectly parse "1230" as a double.
        Token _readNumber() {
            double d;
            int n;
            if (1 == sscanf(str_, "%lf%n", &d, &n)) {
                str_ += n;
                return d;
            }
            long long ll;
            if (1 == sscanf(str_, "%lld%n", &ll, &n)) {
                str_ += n;
                return ll;
            }
            die("Error parsing a number:\n%s", str_);
        }

        const char *str_;

        Token currentToken_ {TT::Unspecified};
    };
} // anonymous namespace


ConfigValue configFromJSON(const std::string& json)
{
    return JSONParser{json.c_str()}.parse();
}

ConfigValue configFromJSONFile(const std::string& filename)
{
    return configFromJSON(readWholeFile(filename));
}

} // namespace mirheo
