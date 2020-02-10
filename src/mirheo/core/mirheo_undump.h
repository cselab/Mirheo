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
    UndumpContext(ConfigValue compute, ConfigValue postprocess,
                  std::string snapshotPath = "snapshot/");
    ~UndumpContext();

    const ConfigObject& getCompObjectConfig(const std::string& category,
                                            const std::string& name);

    template <typename T>
    decltype(auto) getShared(const std::string& ref)
    {
        return UndumpContextGetPtr<T>::getShared(this, ref);
    }
    template <typename T>
    std::unique_ptr<T> getUnique(const std::string& ref)
    {
        return UndumpContextGetPtr<T>::getUnique(this, ref);
    }

    template <typename T>
    std::map<std::string, std::shared_ptr<T>>& getContainerShared()
    {
        return std::get<std::map<std::string, std::shared_ptr<T>>>(shared_);
    }
    template <typename T>
    std::map<std::string, std::unique_ptr<T>>& getContainerUnique()
    {
        return std::get<std::map<std::string, std::unique_ptr<T>>>(unique_);
    }

    const std::string& getPath() const { return path_; }
    const ConfigObject& getComp() const { return compConfig_.getObject(); }
    const ConfigObject& getPost() const { return postConfig_.getObject(); }

private:

    std::tuple<
        std::map<std::string, std::shared_ptr<Mesh>>,
        std::map<std::string, std::shared_ptr<ParticleVector>>,
        std::map<std::string, std::shared_ptr<Interaction>>> shared_;
    std::tuple<
        std::map<std::string, std::unique_ptr<Interaction>>> unique_;
    std::string path_;
    ConfigValue compConfig_;
    ConfigValue postConfig_;
};


template <typename T>
struct UndumpContextGetPtr {
    static const std::shared_ptr<T>& getShared(UndumpContext *context, const std::string& ref) {
        const auto& container = context->getContainerShared<T>();
        auto it = container.find(_parseNameFromReference(ref));
        if (it == container.end())
            _unknownReferenceError(ref);
        return it->second;
    }
    static std::unique_ptr<T> getUnique(UndumpContext *context, const std::string& ref)
    {
        if (ref == ConfigNullRefString)
            return nullptr;
        auto& container = context->getContainerUnique<T>();
        auto it = container.find(_parseNameFromReference(ref));
        if (it == container.end())
            _unknownReferenceError(ref);
        return std::move(it->second);
    }
};

template <>
struct UndumpContextGetPtr<MembraneMesh> {
    static std::shared_ptr<MembraneMesh> getShared(UndumpContext*, const std::string &ref);
};


void importSnapshot(Mirheo *mir, Undumper& un);

} // namespace mirheo
