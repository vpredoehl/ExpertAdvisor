#include "SemanticWorkerRegistry.hpp"

#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

#include <sys/stat.h>
#include <unistd.h>

namespace EA::Scheduler
{
namespace
{

[[noreturn]] void Fail(const std::string& diagnostic)
{
    throw std::invalid_argument(diagnostic);
}

struct JsonValue
{
    using Object = std::map<std::string, JsonValue>;
    using Array = std::vector<JsonValue>;
    std::variant<std::nullptr_t, bool, long long, std::string, Object, Array>
        value;
};

class JsonParser final
{
public:
    explicit JsonParser(std::string text) : text_{std::move(text)} {}

    JsonValue parse()
    {
        JsonValue result = parseValue();
        whitespace();
        if (position_ != text_.size()) error("trailing_content");
        return result;
    }

private:
    [[noreturn]] void error(const char* reason) const
    {
        Fail("semantic_worker_registry_malformed:" + std::string{reason} +
             ":offset=" + std::to_string(position_));
    }

    void whitespace()
    {
        while (position_ < text_.size() &&
               std::isspace(static_cast<unsigned char>(text_[position_])))
            ++position_;
    }

    char take()
    {
        if (position_ == text_.size()) error("unexpected_end");
        return text_[position_++];
    }

    void expect(char expected)
    {
        whitespace();
        if (take() != expected) error("unexpected_token");
    }

    JsonValue parseValue()
    {
        whitespace();
        if (position_ == text_.size()) error("missing_value");
        switch (text_[position_])
        {
            case '{': return parseObject();
            case '[': return parseArray();
            case '"': return JsonValue{parseString()};
            case 't': return parseLiteral("true", JsonValue{true});
            case 'f': return parseLiteral("false", JsonValue{false});
            case 'n': return parseLiteral("null", JsonValue{nullptr});
            default:
                if (text_[position_] == '-' ||
                    std::isdigit(static_cast<unsigned char>(text_[position_])))
                    return JsonValue{parseInteger()};
                error("invalid_value");
        }
    }

    JsonValue parseLiteral(const char* literal, JsonValue result)
    {
        const std::size_t length = std::strlen(literal);
        if (text_.substr(position_, length) != literal)
            error("invalid_literal");
        position_ += length;
        return result;
    }

    JsonValue parseObject()
    {
        expect('{');
        JsonValue::Object object;
        whitespace();
        if (position_ < text_.size() && text_[position_] == '}')
        {
            ++position_;
            return JsonValue{std::move(object)};
        }
        while (true)
        {
            whitespace();
            if (position_ == text_.size() || text_[position_] != '"')
                error("object_key_required");
            std::string key = parseString();
            expect(':');
            if (!object.emplace(key, parseValue()).second)
                error("duplicate_key");
            whitespace();
            const char separator = take();
            if (separator == '}') break;
            if (separator != ',') error("object_separator_required");
        }
        return JsonValue{std::move(object)};
    }

    JsonValue parseArray()
    {
        expect('[');
        JsonValue::Array array;
        whitespace();
        if (position_ < text_.size() && text_[position_] == ']')
        {
            ++position_;
            return JsonValue{std::move(array)};
        }
        while (true)
        {
            array.push_back(parseValue());
            whitespace();
            const char separator = take();
            if (separator == ']') break;
            if (separator != ',') error("array_separator_required");
        }
        return JsonValue{std::move(array)};
    }

    static void AppendUtf8(std::string& output, unsigned int scalar)
    {
        if (scalar <= 0x7fU) output.push_back(static_cast<char>(scalar));
        else if (scalar <= 0x7ffU)
        {
            output.push_back(static_cast<char>(0xc0U | (scalar >> 6U)));
            output.push_back(static_cast<char>(0x80U | (scalar & 0x3fU)));
        }
        else
        {
            output.push_back(static_cast<char>(0xe0U | (scalar >> 12U)));
            output.push_back(static_cast<char>(0x80U | ((scalar >> 6U) & 0x3fU)));
            output.push_back(static_cast<char>(0x80U | (scalar & 0x3fU)));
        }
    }

    std::string parseString()
    {
        if (take() != '"') error("string_required");
        std::string result;
        while (position_ < text_.size())
        {
            const unsigned char character =
                static_cast<unsigned char>(take());
            if (character == '"') return result;
            if (character < 0x20U) error("control_character_in_string");
            if (character != '\\')
            {
                result.push_back(static_cast<char>(character));
                continue;
            }
            const char escaped = take();
            switch (escaped)
            {
                case '"': result.push_back('"'); break;
                case '\\': result.push_back('\\'); break;
                case '/': result.push_back('/'); break;
                case 'b': result.push_back('\b'); break;
                case 'f': result.push_back('\f'); break;
                case 'n': result.push_back('\n'); break;
                case 'r': result.push_back('\r'); break;
                case 't': result.push_back('\t'); break;
                case 'u':
                {
                    unsigned int scalar = 0;
                    for (int digit = 0; digit < 4; ++digit)
                    {
                        const char hex = take();
                        scalar <<= 4U;
                        if (hex >= '0' && hex <= '9') scalar += hex - '0';
                        else if (hex >= 'a' && hex <= 'f') scalar += hex - 'a' + 10;
                        else if (hex >= 'A' && hex <= 'F') scalar += hex - 'A' + 10;
                        else error("invalid_unicode_escape");
                    }
                    if (scalar >= 0xd800U && scalar <= 0xdfffU)
                        error("unicode_surrogate_unsupported");
                    AppendUtf8(result, scalar);
                    break;
                }
                default: error("invalid_escape");
            }
        }
        error("unterminated_string");
    }

    long long parseInteger()
    {
        const std::size_t begin = position_;
        if (text_[position_] == '-') ++position_;
        if (position_ == text_.size()) error("invalid_integer");
        if (text_[position_] == '0') ++position_;
        else
        {
            if (!std::isdigit(static_cast<unsigned char>(text_[position_])))
                error("invalid_integer");
            while (position_ < text_.size() &&
                   std::isdigit(static_cast<unsigned char>(text_[position_])))
                ++position_;
        }
        if (position_ < text_.size() &&
            (text_[position_] == '.' || text_[position_] == 'e' ||
             text_[position_] == 'E'))
            error("integer_required");
        try
        {
            return std::stoll(text_.substr(begin, position_ - begin));
        }
        catch (const std::exception&)
        {
            error("integer_out_of_range");
        }
    }

    std::string text_;
    std::size_t position_ = 0;
};

std::string ReadFile(const std::filesystem::path& path, const char* diagnostic)
{
    std::ifstream input{path, std::ios::binary};
    if (!input) Fail(std::string{diagnostic} + ":" + path.string());
    std::ostringstream contents;
    contents << input.rdbuf();
    if (!input.good() && !input.eof())
        Fail(std::string{diagnostic} + ":" + path.string());
    return contents.str();
}

const JsonValue::Object& Object(const JsonValue& value, const std::string& field)
{
    const auto* object = std::get_if<JsonValue::Object>(&value.value);
    if (!object) Fail("semantic_worker_registry_malformed:" + field + "_object_required");
    return *object;
}

const JsonValue::Array& Array(const JsonValue& value, const std::string& field)
{
    const auto* array = std::get_if<JsonValue::Array>(&value.value);
    if (!array) Fail("semantic_worker_registry_malformed:" + field + "_array_required");
    return *array;
}

const JsonValue& Required(
    const JsonValue::Object& object, const std::string& field)
{
    const auto found = object.find(field);
    if (found == object.end())
        Fail("semantic_worker_registry_malformed:missing_" + field);
    return found->second;
}

void RequireOnlyFields(
    const JsonValue::Object& object, const std::set<std::string>& allowed,
    const std::string& context)
{
    for (const auto& [field, value] : object)
    {
        (void)value;
        if (!allowed.contains(field))
            Fail("semantic_worker_registry_malformed:unknown_" + context +
                 "_field=" + field);
    }
}

long long Integer(const JsonValue& value, const std::string& field)
{
    const auto* integer = std::get_if<long long>(&value.value);
    if (!integer)
        Fail("semantic_worker_registry_malformed:" + field + "_integer_required");
    return *integer;
}

std::string String(const JsonValue& value, const std::string& field)
{
    const auto* string = std::get_if<std::string>(&value.value);
    if (!string)
        Fail("semantic_worker_registry_malformed:" + field + "_string_required");
    return *string;
}

int PositiveInt(const JsonValue& value, const std::string& field)
{
    const long long parsed = Integer(value, field);
    if (parsed <= 0 || parsed > std::numeric_limits<int>::max())
        Fail("semantic_worker_registry_malformed:" + field + "_invalid");
    return static_cast<int>(parsed);
}

std::size_t PositiveSize(const JsonValue& value, const std::string& field)
{
    const long long parsed = Integer(value, field);
    if (parsed <= 0)
        Fail("semantic_worker_registry_malformed:" + field + "_invalid");
    return static_cast<std::size_t>(parsed);
}

bool LowerHex(const std::string& text, std::size_t length)
{
    return text.size() == length &&
        std::all_of(text.begin(), text.end(), [](unsigned char character)
        {
            return (character >= '0' && character <= '9') ||
                   (character >= 'a' && character <= 'f');
        });
}

std::set<std::string> Capabilities(const JsonValue& value)
{
    std::set<std::string> result;
    for (const auto& item : Array(value, "capabilities"))
    {
        const std::string capability = String(item, "capability");
        if (capability != "train" && capability != "infer" &&
            capability != "analyze")
            Fail("semantic_worker_capability_mismatch:unknown=" + capability);
        if (!result.insert(capability).second)
            Fail("semantic_worker_registry_malformed:duplicate_capability=" +
                 capability);
    }
    if (result.empty())
        Fail("semantic_worker_capability_mismatch:empty");
    return result;
}

std::string Sha256(const std::filesystem::path& path)
{
    const std::string contents = ReadFile(path, "semantic_worker_artifact_unreadable");
    std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest{};
    if (contents.size() > std::numeric_limits<CC_LONG>::max())
        Fail("semantic_worker_artifact_too_large:" + path.string());
    CC_SHA256(contents.data(), static_cast<CC_LONG>(contents.size()), digest.data());
    static constexpr char hexadecimal[] = "0123456789abcdef";
    std::string result;
    result.reserve(digest.size() * 2U);
    for (const unsigned char byte : digest)
    {
        result.push_back(hexadecimal[byte >> 4U]);
        result.push_back(hexadecimal[byte & 0x0fU]);
    }
    return result;
}

std::filesystem::path CanonicalExisting(
    const std::filesystem::path& path, const char* diagnostic)
{
    std::error_code error;
    const auto canonical = std::filesystem::canonical(path, error);
    if (error) Fail(std::string{diagnostic} + ":" + path.string());
    return canonical;
}

bool IsWithin(
    const std::filesystem::path& child, const std::filesystem::path& parent)
{
    auto childIt = child.begin();
    for (auto parentIt = parent.begin(); parentIt != parent.end();
         ++parentIt, ++childIt)
    {
        if (childIt == child.end() || *childIt != *parentIt) return false;
    }
    return true;
}

std::filesystem::path ResolveArtifactPath(
    const std::filesystem::path& root, const std::string& relative,
    const char* diagnostic)
{
    const std::filesystem::path supplied{relative};
    if (supplied.empty() || supplied.is_absolute() || supplied != supplied.lexically_normal())
        Fail("semantic_worker_registry_malformed:artifact_path_not_canonical=" + relative);
    const auto expected = root / supplied;
    const auto resolved = CanonicalExisting(expected, diagnostic);
    if (!IsWithin(resolved, root))
        Fail("semantic_worker_registry_malformed:artifact_path_escapes_root=" + relative);
    if (resolved != expected)
        Fail("semantic_worker_registry_malformed:artifact_path_not_canonical=" + relative);
    return resolved;
}

SemanticWorkerRuntimePackage ParseRuntime(
    const JsonValue& value, const std::filesystem::path& root)
{
    const auto& object = Object(value, "runtime");
    RequireOnlyFields(object, {"identity", "directory", "manifest"},
                      "runtime");

    SemanticWorkerRuntimePackage runtime;
    runtime.identity = String(Required(object, "identity"), "identity");
    if (!LowerHex(runtime.identity, 64U))
        Fail("semantic_worker_registry_malformed:runtime_identity_invalid");

    const std::string expectedPrefix = "runtime/" + runtime.identity;
    const std::string directoryRelative =
        String(Required(object, "directory"), "directory");
    const std::string manifestRelative =
        String(Required(object, "manifest"), "manifest");
    if (directoryRelative != expectedPrefix ||
        manifestRelative != expectedPrefix + "/manifest.json")
        Fail("semantic_worker_registry_malformed:runtime_path_mismatch:identity=" +
             runtime.identity);

    const auto directory = ResolveArtifactPath(
        root, directoryRelative, "semantic_worker_runtime_missing");
    const auto manifest = ResolveArtifactPath(
        root, manifestRelative, "semantic_worker_runtime_manifest_missing");
    struct stat directoryStatus {};
    if (::lstat(directory.c_str(), &directoryStatus) != 0 ||
        !S_ISDIR(directoryStatus.st_mode))
        Fail("semantic_worker_runtime_missing:" + directory.string());
    if (Sha256(manifest) != runtime.identity)
        Fail("semantic_worker_runtime_manifest_hash_mismatch:identity=" +
             runtime.identity);

    const JsonValue manifestValue = JsonParser{ReadFile(
        manifest, "semantic_worker_runtime_manifest_unreadable")}.parse();
    const auto& manifestObject = Object(manifestValue, "runtime_manifest");
    RequireOnlyFields(manifestObject,
        {"schema_version", "storage", "resources"}, "runtime_manifest");
    if (Integer(Required(manifestObject, "schema_version"), "schema_version") !=
            kSemanticWorkerRuntimeManifestSchemaVersion ||
        String(Required(manifestObject, "storage"), "storage") != "immutable")
        Fail("semantic_worker_runtime_manifest_mismatch:identity=" +
             runtime.identity);

    for (const auto& item : Array(
             Required(manifestObject, "resources"), "resources"))
    {
        const auto& resourceObject = Object(item, "runtime_resource");
        RequireOnlyFields(resourceObject,
            {"built_identity", "runtime_name", "sha256"},
            "runtime_resource");
        SemanticWorkerRuntimeResource resource;
        resource.builtIdentity = String(
            Required(resourceObject, "built_identity"), "built_identity");
        resource.runtimeName = String(
            Required(resourceObject, "runtime_name"), "runtime_name");
        resource.sha256 = String(
            Required(resourceObject, "sha256"), "sha256");
        if (!LowerHex(resource.sha256, 64U))
            Fail("semantic_worker_runtime_manifest_mismatch:resource_hash");
        const bool expectedDefault =
            resource.builtIdentity == "default.metallib" &&
            resource.runtimeName == "default.metallib";
        const bool expectedMetaNN =
            resource.builtIdentity == "MetaNN_metal.metallib" &&
            resource.runtimeName == "MetaNN.metallib";
        if (!expectedDefault && !expectedMetaNN)
            Fail("semantic_worker_runtime_manifest_mismatch:resource=" +
                 resource.runtimeName);
        resource.canonicalPath = ResolveArtifactPath(
            root,
            expectedPrefix + "/" + resource.runtimeName,
            "semantic_worker_runtime_dependency_missing").string();
        struct stat resourceStatus {};
        if (::lstat(resource.canonicalPath.c_str(), &resourceStatus) != 0 ||
            !S_ISREG(resourceStatus.st_mode))
            Fail("semantic_worker_runtime_dependency_unresolvable:resource=" +
                 resource.runtimeName);
        if (Sha256(resource.canonicalPath) != resource.sha256)
            Fail("semantic_worker_runtime_dependency_hash_mismatch:resource=" +
                 resource.runtimeName);
        if (!runtime.resources.emplace(
                resource.runtimeName, std::move(resource)).second)
            Fail("semantic_worker_runtime_manifest_mismatch:duplicate_resource");
    }
    if (runtime.resources.size() != 2U ||
        !runtime.resources.contains("default.metallib") ||
        !runtime.resources.contains("MetaNN.metallib"))
        Fail("semantic_worker_runtime_manifest_mismatch:required_resources");

    runtime.canonicalDirectoryPath = directory.string();
    runtime.canonicalManifestPath = manifest.string();
    return runtime;
}

SemanticWorkerArtifact ParseWorker(
    const JsonValue& value, const std::filesystem::path& root,
    const int registrySchemaVersion)
{
    const auto& object = Object(value, "worker");
    RequireOnlyFields(object,
        registrySchemaVersion == kSemanticWorkerRegistrySchemaVersion
            ? std::set<std::string>{"semantic_layout", "worker_role", "artifact_manifest_schema_version", "worker_rule", "model_input_width",
         "source_commit", "sha256", "executable", "manifest",
         "runtime_identity", "capabilities"}
            : std::set<std::string>{"semantic_layout", "worker_rule", "model_input_width",
         "source_commit", "sha256", "executable", "manifest",
         "runtime_identity", "capabilities"},
        "worker");
    SemanticWorkerArtifact worker;
    worker.semanticLayoutVersion = PositiveInt(
        Required(object, "semantic_layout"), "semantic_layout");
    worker.modelInputWidth = PositiveSize(
        Required(object, "model_input_width"), "model_input_width");
    const std::string kind = String(
        Required(object, "worker_rule"), "worker_rule");
    if (kind == "current") worker.kind = SemanticWorkerArtifactKind::Current;
    else if (kind == "historical")
        worker.kind = SemanticWorkerArtifactKind::Historical;
    else Fail("semantic_worker_registry_malformed:worker_rule_invalid");
    worker.sourceCommit = String(Required(object, "source_commit"), "source_commit");
    worker.sha256 = String(Required(object, "sha256"), "sha256");
    worker.runtimeIdentity = String(
        Required(object, "runtime_identity"), "runtime_identity");
    if (!LowerHex(worker.sourceCommit, 40U))
        Fail("semantic_worker_registry_malformed:source_commit_invalid");
    if (!LowerHex(worker.sha256, 64U))
        Fail("semantic_worker_registry_malformed:sha256_invalid");
    if (!LowerHex(worker.runtimeIdentity, 64U))
        Fail("semantic_worker_registry_malformed:runtime_identity_invalid");
    worker.capabilities = Capabilities(Required(object, "capabilities"));

    if (registrySchemaVersion == kSemanticWorkerRegistrySchemaVersion)
    {
        const std::string role = String(Required(object, "worker_role"), "worker_role");
        if (role == "infer") worker.role = SemanticWorkerRole::Infer;
        else if (role == "train") worker.role = SemanticWorkerRole::Train;
        else Fail("semantic_worker_registry_malformed:worker_role_invalid");
        worker.artifactManifestSchemaVersion = static_cast<int>(Integer(
            Required(object, "artifact_manifest_schema_version"),
            "artifact_manifest_schema_version"));
        if (worker.artifactManifestSchemaVersion !=
                kLegacySemanticWorkerArtifactManifestSchemaVersion &&
            worker.artifactManifestSchemaVersion !=
                kSemanticWorkerArtifactManifestSchemaVersion)
            Fail("semantic_worker_registry_malformed:artifact_manifest_schema_version_invalid");
    }
    else
    {
        // Registry v2 predates role-aware artifacts.  Its single binding is
        // explicitly interpreted as the immutable inference binding.
        worker.artifactManifestSchemaVersion =
            kLegacySemanticWorkerArtifactManifestSchemaVersion;
    }

    const std::string roleComponent = worker.role == SemanticWorkerRole::Infer
        ? "infer" : "train";
    const bool roleAwareArtifact = registrySchemaVersion ==
        kSemanticWorkerRegistrySchemaVersion && worker.artifactManifestSchemaVersion ==
        kSemanticWorkerArtifactManifestSchemaVersion;
    const std::string expectedPrefix = roleAwareArtifact
        ? "layout" + std::to_string(worker.semanticLayoutVersion) + "/" +
              roleComponent + "/" + worker.sourceCommit + "/" + worker.sha256 + "/"
        : "layout" + std::to_string(worker.semanticLayoutVersion) + "/" +
              worker.sourceCommit + "/" + worker.sha256 + "/";
    const std::string executableRelative =
        String(Required(object, "executable"), "executable");
    const std::string manifestRelative =
        String(Required(object, "manifest"), "manifest");
    const std::string executableIdentity = roleAwareArtifact
        ? (worker.role == SemanticWorkerRole::Infer ? "lstm-infer-worker" : "LSTM_Release")
        : "LSTM_Release";
    if (executableRelative != expectedPrefix + executableIdentity ||
        manifestRelative != expectedPrefix + "manifest.json")
        Fail("semantic_worker_registry_malformed:content_addressed_path_mismatch:layout=" +
             std::to_string(worker.semanticLayoutVersion));

    const auto executable = ResolveArtifactPath(
        root, executableRelative, "semantic_worker_artifact_missing");
    const auto manifest = ResolveArtifactPath(
        root, manifestRelative, "semantic_worker_manifest_missing");
    struct stat executableStatus {};
    if (::lstat(executable.c_str(), &executableStatus) != 0 ||
        !S_ISREG(executableStatus.st_mode) ||
        ::access(executable.c_str(), X_OK) != 0)
        Fail("semantic_worker_artifact_not_executable:" + executable.string());
    worker.canonicalExecutablePath = executable.string();
    worker.canonicalManifestPath = manifest.string();

    if (Sha256(executable) != worker.sha256)
        Fail("semantic_worker_hash_mismatch:layout=" +
             std::to_string(worker.semanticLayoutVersion));

    const JsonValue manifestValue = JsonParser{ReadFile(
        manifest, "semantic_worker_manifest_unreadable")}.parse();
    const auto& manifestObject = Object(manifestValue, "manifest");
    RequireOnlyFields(manifestObject,
        roleAwareArtifact
            ? std::set<std::string>{"schema_version", "semantic_layout", "storage",
         "model_input_width", "source_commit", "sha256", "executable_identity",
         "worker_role", "capabilities"}
            : std::set<std::string>{"schema_version", "semantic_layout", "storage",
         "model_input_width", "source_commit", "sha256",
         "executable_identity", "capabilities"},
        "manifest");
    if (Integer(Required(manifestObject, "schema_version"), "schema_version") !=
            worker.artifactManifestSchemaVersion ||
        PositiveInt(Required(manifestObject, "semantic_layout"), "semantic_layout") !=
            worker.semanticLayoutVersion ||
        PositiveSize(Required(manifestObject, "model_input_width"), "model_input_width") !=
            worker.modelInputWidth ||
        String(Required(manifestObject, "storage"), "storage") != "immutable" ||
        String(Required(manifestObject, "source_commit"), "source_commit") !=
            worker.sourceCommit ||
        String(Required(manifestObject, "sha256"), "sha256") != worker.sha256 ||
        String(Required(manifestObject, "executable_identity"), "executable_identity") !=
            executableIdentity ||
        (roleAwareArtifact &&
         String(Required(manifestObject, "worker_role"), "worker_role") != roleComponent) ||
        Capabilities(Required(manifestObject, "capabilities")) !=
            worker.capabilities)
        Fail("semantic_worker_manifest_mismatch:layout=" +
             std::to_string(worker.semanticLayoutVersion));
    return worker;
}

} // namespace

std::string ValidateAndCanonicalizeWorkerExecutable(
    const std::string& configuredPath, const std::string& optionName)
{
    if (configuredPath.empty() || configuredPath.front() != '/')
        throw std::invalid_argument(optionName + " requires an absolute executable path");
    errno = 0;
    char* resolved = ::realpath(configuredPath.c_str(), nullptr);
    if (resolved == nullptr)
        throw std::invalid_argument(optionName + " path cannot be canonicalized: " +
                                    std::string{std::strerror(errno)});
    std::string canonicalPath{resolved};
    std::free(resolved);
    struct stat status {};
    if (::stat(canonicalPath.c_str(), &status) != 0 ||
        !S_ISREG(status.st_mode) || ::access(canonicalPath.c_str(), X_OK) != 0)
        throw std::invalid_argument(
            optionName + " requires an existing executable regular file");
    return canonicalPath;
}

SemanticWorkerRegistry SemanticWorkerRegistry::Load(
    const SemanticWorkerRegistryLoadRequest& request)
{
    if (request.registryPath.empty())
        Fail("semantic_worker_registry_missing:path_empty");
    const auto registryPath = CanonicalExisting(
        request.registryPath, "semantic_worker_registry_missing");
    struct stat registryStatus {};
    if (::stat(registryPath.c_str(), &registryStatus) != 0 ||
        !S_ISREG(registryStatus.st_mode))
        Fail("semantic_worker_registry_missing:" + registryPath.string());
    const auto root = registryPath.parent_path();
    const JsonValue parsed = JsonParser{ReadFile(
        registryPath, "semantic_worker_registry_unreadable")}.parse();
    const auto& object = Object(parsed, "registry");
    RequireOnlyFields(object,
        {"schema_version", "current_layout", "runtimes", "workers"},
        "registry");
    const int registrySchemaVersion = static_cast<int>(
        Integer(Required(object, "schema_version"), "schema_version"));
    if (registrySchemaVersion != kSemanticWorkerRegistrySchemaVersion &&
        registrySchemaVersion != kLegacySemanticWorkerRegistrySchemaVersion)
        Fail("semantic_worker_registry_schema_unsupported");

    SemanticWorkerRegistry registry;
    registry.canonicalRegistryPath_ = registryPath.string();
    registry.currentLayoutVersion_ = PositiveInt(
        Required(object, "current_layout"), "current_layout");
    for (const auto& item : Array(Required(object, "runtimes"), "runtimes"))
    {
        SemanticWorkerRuntimePackage runtime = ParseRuntime(item, root);
        if (!registry.runtimes_.emplace(
                runtime.identity, std::move(runtime)).second)
            Fail("semantic_worker_registry_duplicate_runtime");
    }
    if (registry.runtimes_.empty())
        Fail("semantic_worker_registry_malformed:runtimes_empty");
    std::size_t currentEntries = 0;
    for (const auto& item : Array(Required(object, "workers"), "workers"))
    {
        SemanticWorkerArtifact worker = ParseWorker(item, root, registrySchemaVersion);
        if (worker.kind == SemanticWorkerArtifactKind::Current)
            ++currentEntries;
        if (!registry.workers_.emplace(
                worker.semanticLayoutVersion, std::move(worker)).second)
            Fail("semantic_worker_registry_duplicate_layout");
    }
    const auto current = registry.workers_.find(registry.currentLayoutVersion_);
    if (currentEntries != 1U || current == registry.workers_.end() ||
        current->second.kind != SemanticWorkerArtifactKind::Current)
        Fail("semantic_worker_registry_current_rule_invalid");
    if (registry.currentLayoutVersion_ !=
            request.expectedCurrentSemanticLayoutVersion ||
        current->second.modelInputWidth != request.expectedCurrentModelInputWidth ||
        !current->second.capabilities.contains("infer") ||
        (registrySchemaVersion == kLegacySemanticWorkerRegistrySchemaVersion &&
         (!current->second.capabilities.contains("train") ||
          !current->second.capabilities.contains("analyze"))))
        Fail("semantic_worker_capability_mismatch:current_rule");
    for (const auto& [layout, worker] : registry.workers_)
    {
        if (!worker.capabilities.contains("infer"))
            Fail("semantic_worker_capability_mismatch:layout=" +
                 std::to_string(layout) + ":infer_required");
        if (!registry.runtimes_.contains(worker.runtimeIdentity))
            Fail("semantic_worker_runtime_identity_unavailable:layout=" +
                 std::to_string(layout) + ":identity=" +
                 worker.runtimeIdentity);
        const auto validation = registry.validateRuntimeForExecutable(
            worker.canonicalExecutablePath);
        if (!validation.ready)
            Fail(validation.diagnostic);
    }

    if (request.legacyLayout6ExecutableAssertion)
    {
        const auto legacy = registry.workers_.find(6);
        if (legacy == registry.workers_.end())
            Fail("legacy_layout6_worker_assertion_without_registry_entry");
        const std::string asserted = ValidateAndCanonicalizeWorkerExecutable(
            *request.legacyLayout6ExecutableAssertion,
            "--legacy-layout6-infer-worker");
        if (asserted != legacy->second.canonicalExecutablePath)
            Fail("legacy_layout6_worker_registry_conflict:registry=" +
                 legacy->second.canonicalExecutablePath + ":flag=" + asserted);
    }
    return registry;
}

const std::string& SemanticWorkerRegistry::canonicalRegistryPath() const noexcept
{
    return canonicalRegistryPath_;
}

const SemanticWorkerArtifact& SemanticWorkerRegistry::currentWorker() const
{
    const auto found = workers_.find(currentLayoutVersion_);
    if (found == workers_.end())
        throw std::logic_error("semantic worker registry has no current worker");
    return found->second;
}

const SemanticWorkerArtifact* SemanticWorkerRegistry::find(
    int semanticLayoutVersion) const noexcept
{
    const auto found = workers_.find(semanticLayoutVersion);
    return found == workers_.end() ? nullptr : &found->second;
}

SemanticWorkerRuntimeValidation
SemanticWorkerRegistry::validateRuntimeForExecutable(
    const std::string& canonicalExecutablePath) const
{
    const SemanticWorkerArtifact* selectedWorker = nullptr;
    for (const auto& [layout, worker] : workers_)
    {
        (void)layout;
        if (worker.canonicalExecutablePath == canonicalExecutablePath)
        {
            selectedWorker = &worker;
            break;
        }
    }
    if (selectedWorker == nullptr)
        return {false,
                "semantic_worker_runtime_unregistered_executable:path=" +
                    canonicalExecutablePath,
                {}};

    const auto runtime = runtimes_.find(selectedWorker->runtimeIdentity);
    if (runtime == runtimes_.end())
        return {false,
                "semantic_worker_runtime_identity_unavailable:layout=" +
                    std::to_string(selectedWorker->semanticLayoutVersion) +
                    ":identity=" + selectedWorker->runtimeIdentity,
                {}};

    const std::filesystem::path workerDirectory =
        std::filesystem::path{canonicalExecutablePath}.parent_path();
    for (const auto& [runtimeName, resource] : runtime->second.resources)
    {
        const std::filesystem::path presented = workerDirectory / runtimeName;
        struct stat linkStatus {};
        if (::lstat(presented.c_str(), &linkStatus) != 0)
            return {false,
                    "semantic_worker_runtime_dependency_missing:resource=" +
                        runtimeName + ":worker=" + canonicalExecutablePath,
                    {}};
        if (!S_ISLNK(linkStatus.st_mode))
            return {false,
                    "semantic_worker_runtime_dependency_unresolvable:resource=" +
                        runtimeName + ":worker=" + canonicalExecutablePath,
                    {}};

        std::error_code error;
        const auto observedLink = std::filesystem::read_symlink(presented, error);
        const auto expectedLink = std::filesystem::relative(
            resource.canonicalPath, workerDirectory, error);
        if (error || observedLink != expectedLink)
            return {false,
                    "semantic_worker_runtime_dependency_unresolvable:resource=" +
                        runtimeName + ":worker=" + canonicalExecutablePath,
                    {}};
        const auto resolved = std::filesystem::canonical(presented, error);
        if (error || resolved != std::filesystem::path{resource.canonicalPath})
            return {false,
                    "semantic_worker_runtime_dependency_unresolvable:resource=" +
                        runtimeName + ":worker=" + canonicalExecutablePath,
                    {}};
        try
        {
            if (Sha256(resolved) != resource.sha256)
                return {false,
                        "semantic_worker_runtime_dependency_hash_mismatch:resource=" +
                            runtimeName + ":worker=" + canonicalExecutablePath,
                        {}};
        }
        catch (const std::invalid_argument&)
        {
            return {false,
                    "semantic_worker_runtime_dependency_unresolvable:resource=" +
                        runtimeName + ":worker=" + canonicalExecutablePath,
                    {}};
        }
    }
    return {true, "semantic_worker_runtime_ready", workerDirectory.string()};
}

SemanticWorkerSelection SemanticWorkerRegistry::selectInferenceWorker(
    const PersistedWorkerSemanticIdentity& persisted) const
{
    if (!persisted.inputWidth && !persisted.layoutVersion)
        return {false, "semantic_worker_identity_unavailable", {}, 0, 0, {}};
    if (!persisted.inputWidth || !persisted.layoutVersion)
        return {false, "semantic_worker_identity_incomplete", {}, 0, 0, {}};
    const SemanticWorkerArtifact* worker = find(*persisted.layoutVersion);
    if (worker == nullptr)
        return {false,
                "semantic_worker_layout_unsupported:layout=" +
                    std::to_string(*persisted.layoutVersion),
                {}, 0, 0, {}};
    if (!worker->capabilities.contains("infer") ||
        *persisted.inputWidth != worker->modelInputWidth)
        return {false, "semantic_worker_incompatible", {},
                worker->semanticLayoutVersion, worker->modelInputWidth, {}};
    WorkerSemanticCapability capability;
    capability.layoutVersion = worker->semanticLayoutVersion;
    capability.maximumInputWidth = worker->modelInputWidth;
    const auto admission = EvaluateSemanticWorkerAdmission(
        "infer", persisted, capability);
    if (!admission.admissible)
        return {false, admission.diagnostic, {}, 0, 0, {}};
    return {true, "semantic_worker_compatible",
            worker->canonicalExecutablePath,
            worker->semanticLayoutVersion,
            worker->modelInputWidth,
            worker->kind == SemanticWorkerArtifactKind::Current
                ? "current_published_semantic_worker"
                : "immutable_historical_semantic_worker"};
}

} // namespace EA::Scheduler
