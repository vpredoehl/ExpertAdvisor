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
#include <tuple>
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

int NonNegativeInt(const JsonValue& value, const std::string& field)
{
    const long long parsed = Integer(value, field);
    if (parsed < 0 || parsed > std::numeric_limits<int>::max())
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
            capability != "analyze" &&
            capability != kTrainFeatureAblationCapability)
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
        registrySchemaVersion >= kSemanticWorkerRegistrySchemaVersion
            ? std::set<std::string>{"semantic_layout", "worker_role", "artifact_manifest_schema_version", "worker_rule", "model_input_width",
         "source_commit", "sha256", "executable", "manifest",
         "runtime_identity", "capabilities", "selection_priority"}
            : registrySchemaVersion >= kRoleAwareSemanticWorkerRegistrySchemaVersion
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
    if (registrySchemaVersion >= kSemanticWorkerRegistrySchemaVersion)
        worker.selectionPriority = NonNegativeInt(
            Required(object, "selection_priority"), "selection_priority");

    if (registrySchemaVersion >= kRoleAwareSemanticWorkerRegistrySchemaVersion)
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

    if (worker.capabilities.contains(kTrainFeatureAblationCapability) &&
        (worker.role != SemanticWorkerRole::Train ||
         !worker.capabilities.contains("train")))
    {
        Fail("semantic_worker_capability_mismatch:train_feature_ablation_requires_train_role");
    }

    const std::string roleComponent = worker.role == SemanticWorkerRole::Infer
        ? "infer" : "train";
    const bool roleAwareArtifact = registrySchemaVersion >=
        kRoleAwareSemanticWorkerRegistrySchemaVersion && worker.artifactManifestSchemaVersion ==
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
        registrySchemaVersion != kPreviousSemanticWorkerRegistrySchemaVersion &&
        registrySchemaVersion != kRoleAwareSemanticWorkerRegistrySchemaVersion &&
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
    std::map<SemanticWorkerRole, std::size_t> currentEntries;
    std::set<std::tuple<int, std::size_t, SemanticWorkerRole, int>> priorities;
    std::set<std::tuple<int, SemanticWorkerRole, std::string, std::string>> identities;
    std::set<std::tuple<int, std::size_t, SemanticWorkerRole, std::string>> executablePaths;
    for (const auto& item : Array(Required(object, "workers"), "workers"))
    {
        SemanticWorkerArtifact worker = ParseWorker(item, root, registrySchemaVersion);
        if (worker.kind == SemanticWorkerArtifactKind::Current)
            ++currentEntries[worker.role];
        auto& candidates = registry.workers_[
            std::make_pair(worker.semanticLayoutVersion, worker.role)];
        if (worker.role == SemanticWorkerRole::Infer && !candidates.empty())
            Fail("semantic_worker_registry_duplicate_layout_role");
        if (registrySchemaVersion < kSemanticWorkerRegistrySchemaVersion &&
            !candidates.empty())
            Fail("semantic_worker_registry_duplicate_layout_role");
        const auto identity = std::make_tuple(
            worker.semanticLayoutVersion, worker.role, worker.sourceCommit,
            worker.sha256);
        if (!identities.insert(identity).second)
            Fail("semantic_worker_registry_duplicate_artifact_identity");
        const auto executablePath = std::make_tuple(
            worker.semanticLayoutVersion, worker.modelInputWidth, worker.role,
            worker.canonicalExecutablePath);
        if (!executablePaths.insert(executablePath).second)
            Fail("semantic_worker_registry_duplicate_executable_path");
        if (registrySchemaVersion >= kSemanticWorkerRegistrySchemaVersion &&
            !priorities.insert(std::make_tuple(
                worker.semanticLayoutVersion, worker.modelInputWidth,
                worker.role, worker.selectionPriority)).second)
            Fail("semantic_worker_registry_duplicate_selection_priority");
        candidates.push_back(std::move(worker));
    }
    const auto currentInference = registry.workers_.find(
        {registry.currentLayoutVersion_, SemanticWorkerRole::Infer});
    if (currentEntries[SemanticWorkerRole::Infer] != 1U ||
        currentInference == registry.workers_.end() ||
        currentInference->second.size() != 1U ||
        currentInference->second.front().kind != SemanticWorkerArtifactKind::Current)
        Fail("semantic_worker_registry_current_rule_invalid");
    if (registrySchemaVersion >= kPreviousSemanticWorkerRegistrySchemaVersion)
    {
        const auto currentTraining = registry.workers_.find(
            {registry.currentLayoutVersion_, SemanticWorkerRole::Train});
        if (currentEntries[SemanticWorkerRole::Train] != 1U ||
            currentTraining == registry.workers_.end() ||
            std::count_if(currentTraining->second.begin(),
                          currentTraining->second.end(),
                          [](const SemanticWorkerArtifact& worker)
                          {
                              return worker.kind ==
                                  SemanticWorkerArtifactKind::Current;
                          }) != 1)
            Fail("semantic_worker_registry_training_reference_current_rule_invalid");
    }
    if (registry.currentLayoutVersion_ !=
            request.expectedCurrentSemanticLayoutVersion ||
        currentInference->second.front().modelInputWidth != request.expectedCurrentModelInputWidth ||
        !currentInference->second.front().capabilities.contains("infer") ||
        (registrySchemaVersion == kLegacySemanticWorkerRegistrySchemaVersion &&
         (!currentInference->second.front().capabilities.contains("train") ||
          !currentInference->second.front().capabilities.contains("analyze"))))
        Fail("semantic_worker_capability_mismatch:current_rule");
    for (const auto& [key, candidates] : registry.workers_)
    {
        const int layout = key.first;
        for (const auto& worker : candidates)
        {
            const char* requiredCapability = worker.role == SemanticWorkerRole::Infer
                ? "infer" : "train";
            if (!worker.capabilities.contains(requiredCapability))
                Fail("semantic_worker_capability_mismatch:layout=" +
                     std::to_string(layout) + ":role_required=" + requiredCapability);
            if (!registry.runtimes_.contains(worker.runtimeIdentity))
                Fail("semantic_worker_runtime_identity_unavailable:layout=" +
                     std::to_string(layout) + ":identity=" +
                     worker.runtimeIdentity);
            const auto validation = registry.validateRuntimeForExecutable(
                worker.canonicalExecutablePath);
            if (!validation.ready)
                Fail(validation.diagnostic);
        }
    }

    if (request.legacyLayout6ExecutableAssertion)
    {
        const auto legacy = registry.workers_.find({6, SemanticWorkerRole::Infer});
        if (legacy == registry.workers_.end() || legacy->second.size() != 1U)
            Fail("legacy_layout6_worker_assertion_without_registry_entry");
        const std::string asserted = ValidateAndCanonicalizeWorkerExecutable(
            *request.legacyLayout6ExecutableAssertion,
            "--legacy-layout6-infer-worker");
        if (asserted != legacy->second.front().canonicalExecutablePath)
            Fail("legacy_layout6_worker_registry_conflict:registry=" +
                 legacy->second.front().canonicalExecutablePath + ":flag=" + asserted);
    }
    return registry;
}

const std::string& SemanticWorkerRegistry::canonicalRegistryPath() const noexcept
{
    return canonicalRegistryPath_;
}

const SemanticWorkerArtifact& SemanticWorkerRegistry::currentWorker() const
{
    const auto found = workers_.find(
        {currentLayoutVersion_, SemanticWorkerRole::Train});
    if (found != workers_.end())
    {
        const auto current = std::find_if(
            found->second.begin(), found->second.end(),
            [](const SemanticWorkerArtifact& worker)
            { return worker.kind == SemanticWorkerArtifactKind::Current; });
        if (current != found->second.end()) return *current;
    }
    const auto legacy = workers_.find(
        {currentLayoutVersion_, SemanticWorkerRole::Infer});
    if (legacy != workers_.end() && legacy->second.size() == 1U)
        return legacy->second.front();
    throw std::logic_error("semantic worker registry has no current worker");
}

const SemanticWorkerArtifact* SemanticWorkerRegistry::find(
    int semanticLayoutVersion) const noexcept
{
    return find(semanticLayoutVersion, SemanticWorkerRole::Infer);
}

const SemanticWorkerArtifact* SemanticWorkerRegistry::findByCanonicalExecutable(
    const std::string& canonicalExecutablePath) const noexcept
{
    for (const auto& [key, candidates] : workers_)
    {
        (void)key;
        for (const auto& artifact : candidates)
            if (artifact.canonicalExecutablePath == canonicalExecutablePath)
                return &artifact;
    }
    return nullptr;
}

const SemanticWorkerArtifact* SemanticWorkerRegistry::find(
    int semanticLayoutVersion, SemanticWorkerRole role) const noexcept
{
    const auto found = workers_.find({semanticLayoutVersion, role});
    return found == workers_.end() || found->second.size() != 1U
        ? nullptr : &found->second.front();
}

const std::vector<SemanticWorkerArtifact>* SemanticWorkerRegistry::findCandidates(
    int semanticLayoutVersion, SemanticWorkerRole role) const noexcept
{
    const auto found = workers_.find({semanticLayoutVersion, role});
    return found == workers_.end() ? nullptr : &found->second;
}

SemanticWorkerRuntimeValidation
SemanticWorkerRegistry::validateRuntimeForExecutable(
    const std::string& canonicalExecutablePath) const
{
    const SemanticWorkerArtifact* selectedWorker = nullptr;
    for (const auto& [key, candidates] : workers_)
    {
        (void)key;
        for (const auto& worker : candidates)
        {
            if (worker.canonicalExecutablePath == canonicalExecutablePath)
            {
                selectedWorker = &worker;
                break;
            }
        }
        if (selectedWorker != nullptr) break;
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

namespace
{
SemanticWorkerSelection SelectWorkerForRole(
    const SemanticWorkerRegistry& registry,
    const PersistedWorkerSemanticIdentity& persisted,
    SemanticWorkerRole role,
    const SemanticWorkerCapabilities& requiredCapabilities)
{
    if (!persisted.inputWidth && !persisted.layoutVersion)
        return {false, "semantic_worker_identity_unavailable", {}, 0, 0, {}};
    if (!persisted.inputWidth || !persisted.layoutVersion)
        return {false, "semantic_worker_identity_incomplete", {}, 0, 0, {}};
    const std::vector<SemanticWorkerArtifact>* candidates =
        registry.findCandidates(*persisted.layoutVersion, role);
    // Schema-v2 registries predate explicit roles.  Their combined current
    // artifact is stored under the inference-role key, but currentWorker()
    // exposes it as the training/reference executable.  Permit only that
    // exact current identity; an unsupported historical train identity must
    // still fail closed rather than falling back.
    const SemanticWorkerArtifact* legacyWorker = nullptr;
    if (candidates == nullptr && role == SemanticWorkerRole::Train)
    {
        const SemanticWorkerArtifact& legacyCurrent = registry.currentWorker();
        if (legacyCurrent.semanticLayoutVersion == *persisted.layoutVersion &&
            legacyCurrent.capabilities.contains("train"))
        {
            legacyWorker = &legacyCurrent;
        }
    }
    if (candidates == nullptr && legacyWorker == nullptr)
        return {false,
                "semantic_worker_layout_unsupported:layout=" +
                    std::to_string(*persisted.layoutVersion),
                {}, 0, 0, {}};
    const char* phase = role == SemanticWorkerRole::Infer ? "infer" : "train";
    std::vector<const SemanticWorkerArtifact*> exact;
    if (legacyWorker != nullptr)
        exact.push_back(legacyWorker);
    else
    {
        for (const auto& candidate : *candidates)
        {
            if (candidate.modelInputWidth == *persisted.inputWidth &&
                candidate.capabilities.contains(phase))
                exact.push_back(&candidate);
        }
    }
    if (exact.empty())
        return {false, "semantic_worker_incompatible", {},
                *persisted.layoutVersion, *persisted.inputWidth, {}};

    std::vector<const SemanticWorkerArtifact*> eligible;
    for (const auto* candidate : exact)
        if (std::all_of(requiredCapabilities.begin(), requiredCapabilities.end(),
                        [&](const std::string& capability)
                        { return candidate->capabilities.contains(capability); }))
            eligible.push_back(candidate);
    if (eligible.empty())
    {
        const std::string required = requiredCapabilities.empty()
            ? phase : *requiredCapabilities.begin();
        return {false,
                "semantic_worker_capability_incompatible:layout=" +
                    std::to_string(*persisted.layoutVersion) +
                    ":role=" + phase + ":required=" + required,
                {}, *persisted.layoutVersion, *persisted.inputWidth, {}};
    }

    SemanticWorkerCapabilities baseline = requiredCapabilities;
    baseline.insert(phase);
    const auto excessCount = [&baseline](const SemanticWorkerArtifact* candidate)
    {
        std::size_t result = 0;
        for (const auto& capability : candidate->capabilities)
            if (!baseline.contains(capability)) ++result;
        return result;
    };
    const auto minimumExcessCandidate = std::min_element(
        eligible.begin(), eligible.end(),
        [&excessCount](const auto* left, const auto* right)
        { return excessCount(left) < excessCount(right); });
    const std::size_t minimumExcess = excessCount(*minimumExcessCandidate);
    eligible.erase(std::remove_if(eligible.begin(), eligible.end(),
        [&excessCount, minimumExcess](const auto* candidate)
        { return excessCount(candidate) != minimumExcess; }), eligible.end());
    const auto lowestPriorityCandidate = std::min_element(
        eligible.begin(), eligible.end(),
        [](const auto* left, const auto* right)
        { return left->selectionPriority < right->selectionPriority; });
    const int lowestPriority = (*lowestPriorityCandidate)->selectionPriority;
    eligible.erase(std::remove_if(eligible.begin(), eligible.end(),
        [lowestPriority](const SemanticWorkerArtifact* candidate)
        { return candidate->selectionPriority != lowestPriority; }), eligible.end());
    if (eligible.size() != 1U)
        return {false,
                "semantic_worker_selection_ambiguous:layout=" +
                    std::to_string(*persisted.layoutVersion) +
                    ":width=" + std::to_string(*persisted.inputWidth) +
                    ":role=" + phase,
                {}, *persisted.layoutVersion, *persisted.inputWidth, {}};
    const SemanticWorkerArtifact* worker = eligible.front();
    WorkerSemanticCapability capability;
    capability.layoutVersion = worker->semanticLayoutVersion;
    capability.maximumInputWidth = worker->modelInputWidth;
    const auto admission = EvaluateSemanticWorkerAdmission(
        phase, persisted, capability);
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
} // namespace

SemanticWorkerSelection SemanticWorkerRegistry::selectInferenceWorker(
    const PersistedWorkerSemanticIdentity& persisted) const
{
    return SelectWorkerForRole(
        *this, persisted, SemanticWorkerRole::Infer, {});
}

SemanticWorkerSelection SemanticWorkerRegistry::selectTrainingReferenceWorker(
    const PersistedWorkerSemanticIdentity& persisted,
    const SemanticWorkerCapabilities& requiredCapabilities) const
{
    // A genuinely fresh legacy experiment has no model-bearing identity yet.
    // Preserve that explicit legacy admission contract by choosing the
    // published current training artifact without manufacturing persisted
    // width/layout values.  Missing identity for a resume/model remains
    // fail-closed in SelectWorkerForRole.
    if (!persisted.inputWidth && !persisted.layoutVersion &&
        !persisted.modelIdentityExpected)
    {
        const SemanticWorkerArtifact& worker = currentWorker();
        // Schema-v2 registries represent the combined current executable as
        // the inference-role fallback returned by currentWorker().  Its
        // explicit train capability is the authoritative legacy role proof.
        if (!worker.capabilities.contains("train"))
        {
            return {false, "semantic_worker_incompatible", {},
                    worker.semanticLayoutVersion, worker.modelInputWidth, {}};
        }
        for (const std::string& requiredCapability : requiredCapabilities)
        {
            if (!worker.capabilities.contains(requiredCapability))
                return {false,
                        "semantic_worker_capability_incompatible:layout=" +
                            std::to_string(worker.semanticLayoutVersion) +
                            ":role=train:required=" + requiredCapability,
                        {}, worker.semanticLayoutVersion,
                        worker.modelInputWidth, {}};
        }
        return {true, "semantic_worker_compatible",
                worker.canonicalExecutablePath,
                worker.semanticLayoutVersion,
                worker.modelInputWidth,
                "current_published_semantic_worker"};
    }
    // A control whose persisted mask is empty has no capability requirement of
    // its own. Once an exact layout/width TRAIN candidate has explicitly
    // qualified the canonical ablation implementation, route that control
    // through the same capability-qualified selection domain as a paired
    // ablation arm. This is deliberately a general superset rule, not a
    // study-specific preference: it prevents a no-mask control and a masked
    // treatment from silently separating when a later, narrower plain-TRAIN
    // artifact is appended to the registry. The normal deterministic
    // capability/priority selection below still chooses the one artifact.
    // Layouts with no ablation-qualified candidate retain their historical
    // empty-mask routing behavior.
    SemanticWorkerCapabilities effectiveCapabilities = requiredCapabilities;
    if (effectiveCapabilities.empty() && persisted.inputWidth &&
        persisted.layoutVersion)
    {
        if (const auto* candidates = findCandidates(
                *persisted.layoutVersion, SemanticWorkerRole::Train))
        {
            const bool ablationCapableCandidate = std::any_of(
                candidates->begin(), candidates->end(),
                [&](const SemanticWorkerArtifact& candidate)
                {
                    return candidate.modelInputWidth == *persisted.inputWidth &&
                        candidate.capabilities.contains("train") &&
                        candidate.capabilities.contains(
                            kTrainFeatureAblationCapability);
                });
            if (ablationCapableCandidate)
                effectiveCapabilities.insert(kTrainFeatureAblationCapability);
        }
    }
    return SelectWorkerForRole(
        *this, persisted, SemanticWorkerRole::Train, effectiveCapabilities);
}

} // namespace EA::Scheduler
