#include "config.h"
#include <sstream>
#include <cassert>

namespace mirheo
{

namespace {

std::string doubleToString(double x)
{
    char str[32];
    sprintf(str, "%.17g", x);
    return str;
}

std::string stringToJSON(const std::string &input)
{
    std::string output;
    output.reserve(2 + input.size());
    output.push_back('"');
    for (char c : input) {
        switch (c) {
            case '"':
            case '/':
            case '\b':
            case '\f':
            case '\n':
            case '\r':
            case '\t':
            case '\\':
                output.push_back('\\');
        }
        output.push_back(c);
    }
    output.push_back('"');
    return output;
}


struct ConfigToJSON {
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
    std::vector<Token> tokens;

    void process(const Config &element) {
        if (auto *v = element.get_if<long long>()) {
            tokens.push_back(std::to_string(*v));
        } else if (auto *v = element.get_if<double>()) {
            tokens.push_back(doubleToString(*v));
        } else if (auto *v = element.get_if<std::string>()) {
            tokens.push_back(stringToJSON(*v));
        } else if (auto *dict = element.get_if<Config::Dictionary>()) {
            tokens.push_back(Tag::StartDict);
            for (const auto &pair : *dict) {
                tokens.push_back(Tag::StartDictItem);
                tokens.push_back(stringToJSON(pair.first));
                process(pair.second);
                tokens.push_back(Tag::EndDictItem);
            }
            tokens.push_back(Tag::EndDict);
        } else if (auto *list = element.get_if<Config::List>()) {
            tokens.push_back(Tag::StartList);
            for (const Config &el : *list) {
                tokens.push_back(Tag::StartListItem);
                process(el);
                tokens.push_back(Tag::EndListItem);
            }
            tokens.push_back(Tag::EndList);
        } else {
            assert(false);
        }
    }

    std::string generate() {
        std::ostringstream stream;
        std::string nlindent {'\n'};

        enum class ObjectType { Dict, List };

        auto push = [&]() {
            nlindent += "    ";
        };
        auto pop = [&]() {
            nlindent.erase(nlindent.size() - 4);
        };

        size_t numTokens = tokens.size();
        tokens.push_back("dummy");
        for (size_t i = 0; i < numTokens; ++i) {
            const Token &token = tokens[i];
            const Token &nextToken = tokens[i + 1];
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
                    if (nextTag == Tag::EndDict) {
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
                    stream << mpark::get<std::string>(nextToken);  // Key.
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
};

} // anonymous namespace

std::string configToJSON(const Config &config) {
    ConfigToJSON writer;
    writer.process(config);
    return writer.generate();
}

} // namespace mirheo
