#include "config.h"
#include "strprintf.h"
#include <mirheo/core/logger.h>

#include <cassert>
#include <sstream>

namespace mirheo
{

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
    output.push_back('"');
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
            output.push_back(c);
        }
    }
    output.push_back('"');
    return output;
}

namespace {
    class ConfigToJSON {
    public:
        enum class Tag {
            StartDict,
            EndDict,
            StartList,
            EndList,
            StartDictItem,
            EndDictItem,
            StartListItem,
            EndListItem,
            Dummy
        };

        // For simplicity, we merge Int, Float and String tokens into std::string.
        using Token = mpark::variant<std::string, Tag>;

        void process(const Config& element);
        std::string generate();

    private:
        std::vector<Token> tokens_;
    };
} // anonymous namespace

void ConfigToJSON::process(const Config& element)
{
    if (auto *v = element.get_if<long long>()) {
        tokens_.push_back(std::to_string(*v));
    } else if (auto *v = element.get_if<double>()) {
        tokens_.push_back(doubleToString(*v));
    } else if (auto *v = element.get_if<std::string>()) {
        tokens_.push_back(stringToJSON(*v));
    } else if (auto *dict = element.get_if<Config::Dictionary>()) {
        tokens_.push_back(Tag::StartDict);
        for (const auto &pair : *dict) {
            tokens_.push_back(Tag::StartDictItem);
            tokens_.push_back(stringToJSON(pair.first));
            process(pair.second);
            tokens_.push_back(Tag::EndDictItem);
        }
        tokens_.push_back(Tag::EndDict);
    } else if (auto *list = element.get_if<Config::List>()) {
        tokens_.push_back(Tag::StartList);
        for (const Config& el : *list) {
            tokens_.push_back(Tag::StartListItem);
            process(el);
            tokens_.push_back(Tag::EndListItem);
        }
        tokens_.push_back(Tag::EndList);
    } else {
        assert(false);
    }
}

std::string ConfigToJSON::generate()
{
    std::ostringstream stream;
    std::string nlindent {'\n'};

    enum class ObjectType { Dict, List };

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
        case Tag::StartDict:
            if (nextTag == Tag::EndDict) {
                stream << "{}";
                ++i;
                break;
            }
            stream << '{';
            push();
            break;
        case Tag::EndDict:
            pop();
            stream << nlindent << '}';
            break;
        case Tag::StartList:
            if (nextTag == Tag::EndList) {
                stream << "[]";
                ++i;
                break;
            }
            stream << '[';
            push();
            break;
        case Tag::EndList:
            pop();
            stream << nlindent << ']';
            break;
        case Tag::StartDictItem:
            stream << nlindent;
            stream << mpark::get<std::string>(nextToken); // Key.
            stream << ": ";
            ++i;
            break;
        case Tag::StartListItem:
            stream << nlindent;
            break;
        case Tag::EndDictItem:
        case Tag::EndListItem:
            if (nextTag == Tag::EndDict || nextTag == Tag::EndList)
                break;
            stream << ',';
            break;
        default:
            assert(false);
        }
    }
    return std::move(stream).str();
}

std::string Config::toJSONString() const
{
    ConfigToJSON writer;
    writer.process(*this);
    return writer.generate();
}

Config::Int Config::getInt() const
{
    if (const Int *v = get_if<Int>())
        return *v;
    if (const Float *v = get_if<Float>()) {
        if ((Float)(Int)*v == *v)  // Accept only if lossless.
            return (Int)*v;
    }
    die("getInt on a non-int object:\n%s", toJSONString().c_str());
}

Config::Float Config::getFloat() const
{
    if (const Float *v = get_if<Float>())
        return *v;
    if (const Int *v = get_if<Int>())
        if ((Int)(Float)*v == *v)
            return static_cast<double>(*v);
    die("getFloat on a non-float object:\n%s", toJSONString().c_str());
}

const Config::String& Config::getString() const {
    if (const String *v = get_if<String>())
        return *v;
    die("getString on a non-string object:\n%s", toJSONString().c_str());
}

const Config::List& Config::getList() const
{
    if (auto *list = get_if<List>())
        return *list;
    die("getList on a non-list object:\n%s", toJSONString().c_str());
}

Config::List& Config::getList()
{
    if (auto *list = get_if<List>())
        return *list;
    die("getList on a non-list object:\n%s", toJSONString().c_str());
}

const Config::Dictionary& Config::getDict() const
{
    if (auto *dict = get_if<Dictionary>())
        return *dict;
    die("getDict on a non-dictionary object:\n%s", toJSONString().c_str());
}

Config::Dictionary& Config::getDict()
{
    if (auto *dict = get_if<Dictionary>())
        return *dict;
    die("getDict on a non-dictionary object:\n%s", toJSONString().c_str());
}


Config& Config::at(const std::string &key)
{
    auto &dict = getDict();
    auto it = dict.find(key);
    if (it == dict.end())
        die("Key \"%s\" not found in\n%s", key.c_str(), toJSONString().c_str());
    return it->second;
}

const Config& Config::at(const std::string &key) const
{
    auto &dict = getDict();
    auto it = dict.find(key);
    if (it == dict.end())
        die("Key \"%s\" not found in\n%s", key.c_str(), toJSONString().c_str());
    return it->second;
}

Config& Config::at(const char *key)
{
    return at(std::string(key));
}

const Config& Config::at(const char *key) const
{
    return at(std::string(key));
}

Config& Config::at(size_t i)
{
    auto &list = getList();
    if (i > list.size())
        die("Index %zu out of range (size=%zu):\n%s", i, list.size(), toJSONString().c_str());
    return list[i];
}

const Config& Config::at(size_t i) const
{
    auto &list = getList();
    if (i > list.size())
        die("Index %zu out of range (size=%zu):\n%s", i, list.size(), toJSONString().c_str());
    return list[i];
}


bool DumpContext::isGroupMasterTask() const
{
    int rank;
    MPI_Comm_rank(groupComm, &rank);
    return rank == 0;
}


Dumper::Dumper(DumpContext context) :
    config_{Config::Dictionary{}}, context_{std::move(context)}
{}

Dumper::~Dumper() = default;

bool Dumper::isObjectRegistered(const void *ptr) const noexcept
{
    return descriptions_.find(ptr) != descriptions_.end();
}
const std::string& Dumper::getObjectDescription(const void *ptr) const
{
    assert(isObjectRegistered(ptr));
    return descriptions_.find(ptr)->second;
}

const std::string& Dumper::registerObject(const void *ptr, Config object)
{
    assert(!isObjectRegistered(ptr));

    auto *newDict = object.get_if<Config::Dictionary>();
    if (newDict == nullptr)
        die("Expected a dictionary, instead got:\n%s", object.toJSONString().c_str());

    // Get the category name and remove it from the dictionary.
    auto itCategory = newDict->find("__category");
    if (itCategory == newDict->end()) {
        die("Key \"%s\" not found in the config:\n%s",
            "__category", object.toJSONString().c_str());
    }
    std::string category = std::move(itCategory)->second.getString();
    newDict->erase(itCategory);

    // Find the category in the master dict. Add an empty list if not found.
    auto& dict = config_.getDict();
    auto it = dict.find(category);
    if (it == dict.end())
        it = dict.emplace(category, Config::List{}).first;

    // Get the object name, if it exists.
    auto itName = newDict->find("name");
    const char *name =
        itName != newDict->end() ? itName->second.getString().c_str() : nullptr;

    // Get the object type.
    auto itType = newDict->find("__type");
    if (itType == newDict->end()) {
        die("Key \"%s\" not found in the config:\n%s",
            "__type", object.toJSONString().c_str());
    }

    const char *type = itType->second.getString().c_str();
    std::string ref = name ? strprintf("<%s with name=%s>", type, name)
                           : strprintf("<%s>", type);
    it->second.getList().emplace_back(std::move(object));

    return descriptions_.emplace(ptr, std::move(ref)).first->second;
}


Undumper::Undumper(UndumpContext context) :
    context_{std::move(context)}
{}

Undumper::~Undumper() = default;


Config ConfigDumper<float3>::dump(Dumper&, float3 v)
{
    return Config::List{(double)v.x, (double)v.y, (double)v.z};
}
float3 ConfigDumper<float3>::undump(Undumper& un, const Config &config)
{
    const auto& list = config.getList();
    if (list.size() != 3)
        die("Expected 3 elements, got %zu.", list.size());
    return float3{un.undump<float>(list[0]),
                  un.undump<float>(list[1]),
                  un.undump<float>(list[2])};
}


/// Read an return the content of a file as a string.
/// Terminates if the file is not found.
static std::string readWholeFile(const std::string& filename)
{
    FileWrapper file;
    if (file.open(filename, "r") != FileWrapper::Status::Success)
        die("Error reading config JSON file \"%s\".", filename.c_str());

    // https://stackoverflow.com/questions/14002954/c-programming-how-to-read-the-whole-file-contents-into-a-buffer
    fseek(file.get(), 0, SEEK_END);
    long size = ftell(file.get());
    fseek(file.get(), 0, SEEK_SET);  /* same as rewind(f); */

    std::string output(size, '_');
    fread(&output[0], 1, size, file.get());
    return output;
}


namespace {
    /**
     * Parse JSON and return a Config.
     *
     * Usage:
     *      Config config = JSonParser{"[10, 20, 30.5"]}.parse();
     */
    class JSONParser {
    public:
        JSONParser(const char *s) : str_(s) {}

        Config parse() {
            Token token = nextToken(TT::AnyValue);
            if (token == TT::OpenBrace) {
                Config::Dictionary dict;
                for (;;) {
                    if (peekToken(TT::AnyValue, TT::ClosedBrace) == TT::ClosedBrace)
                        break;
                    std::string key = mpark::get<Config::String>(nextToken(TT::String).value);
                    nextToken(TT::Colon);
                    dict.emplace(key, parse());
                    if (peekToken(TT::Comma, TT::ClosedBrace) == TT::Comma) {
                        nextToken(TT::Comma);  // Consume.
                        continue;
                    }
                }
                nextToken(TT::ClosedBrace);  // Consume.
                return Config{std::move(dict)};
            } else if (token == TT::OpenSquare) {
                Config::List list;
                for (;;) {
                    if (peekToken(TT::AnyValue, TT::ClosedSquare) == TT::ClosedSquare)
                        break;
                    list.push_back(parse());
                    if (peekToken(TT::ClosedSquare, TT::Comma) == TT::Comma) {
                        nextToken(TT::Comma);  // Consume.
                        continue;
                    }
                }
                nextToken(TT::ClosedSquare);  // Consume.
                return Config{std::move(list)};
            } else if (token == TT::Int) {
                return mpark::get<Config::Int>(token.value);
            } else if (token == TT::Float) {
                return mpark::get<Config::Float>(token.value);
            } else if (token == TT::String) {
                return std::move(mpark::get<Config::String>(token.value));
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
            mpark::variant<Config::Int, Config::Float, Config::String> value {Config::Int{0}};
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


Config configFromJSON(const std::string& json)
{
    return JSONParser{json.c_str()}.parse();
}

Config configFromJSONFile(const std::string& filename)
{
    return configFromJSON(readWholeFile(filename));
}

} // namespace mirheo
