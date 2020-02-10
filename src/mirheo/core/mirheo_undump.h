#pragma once

#include <mirheo/core/utils/config.h>

#include <tuple>

namespace mirheo
{

class Mirheo;
class Mesh;
class MembraneMesh;
class ParticleVector;
class Interaction;

std::string _parseNameFromReference(const std::string &ref);
void _unknownReferenceError [[noreturn]] (const std::string &ref);

template <typename T>
struct UndumpContextGetPtr;

class UndumpContext {
public:
    UndumpContext(std::string snapshotPath);
    UndumpContext(Config compute, Config postprocess,
                  std::string snapshotPath = "snapshot/");
    ~UndumpContext();

    template <typename T>
    decltype(auto) get(const std::string &ref)
    {
        return UndumpContextGetPtr<T>::get(this, ref);
    }

    template <typename T>
    std::map<std::string, std::shared_ptr<T>>& getContainer()
    {
        return std::get<std::map<std::string, std::shared_ptr<T>>>(objects_);
    }

    decltype(auto) getAllObjects() { return objects_; }
    const std::string& getPath() const { return path_; }
    const Config& getComp() const { return compConfig_; }
    const Config& getPost() const { return postConfig_; }

private:

    std::tuple<
        std::map<std::string, std::shared_ptr<Mesh>>,
        std::map<std::string, std::shared_ptr<ParticleVector>>,
        std::map<std::string, std::shared_ptr<Interaction>>> objects_;
    std::string path_;
    Config compConfig_;
    Config postConfig_;
};


template <typename T>
struct UndumpContextGetPtr {
    static const std::shared_ptr<T>& get(UndumpContext *context, const std::string &ref) {
        const auto &container = std::get<std::map<std::string, std::shared_ptr<T>>>(
                context->getAllObjects());
        auto it = container.find(_parseNameFromReference(ref));
        if (it == container.end())
            _unknownReferenceError(ref);
        return it->second;
    }
};

template <>
struct UndumpContextGetPtr<MembraneMesh> {
    static std::shared_ptr<MembraneMesh> get(UndumpContext*, const std::string &ref);
};


void importSnapshot(Mirheo *mir, Undumper& un);

} // namespace mirheo
