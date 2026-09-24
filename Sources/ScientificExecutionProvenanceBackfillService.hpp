#pragma once

// Deliberately separate from evaluation repositories.  This is a one-off
// historical repair boundary; timestamp/path inference is prohibited from
// normal evidence loading.

#include "SchedulerCore/SemanticWorkerRegistry.hpp"

#include <CommonCrypto/CommonDigest.h>
#include <pqxx/pqxx>

#include <array>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace EA::ScientificExecutionProvenanceBackfill
{
enum class Outcome { proposed, missing, ambiguous, conflict, unresolvedIdentity, alreadyBound };

struct Options {
    bool apply = false;
    bool modeSpecified = false;
    bool confirmed = false;
    std::optional<long long> experimentId;
    std::string registryPath = "Builds/SemanticWorkers/registry.json";
};

inline bool IsCommand(int argc, const char* const argv[])
{
    for (int i = 1; i < argc; ++i)
        if (std::string_view{argv[i]} == "--backfill-scientific-execution-provenance")
            return true;
    return false;
}

inline Options Parse(int argc, const char* const argv[])
{
    Options result;
    bool command = false;
    for (int i = 1; i < argc; ++i) {
        const std::string value{argv[i]};
        if (value == "--backfill-scientific-execution-provenance") {
            if (command) throw std::invalid_argument("backfill command specified more than once");
            command = true;
        } else if (value == "--dry-run" || value == "--apply") {
            if (result.modeSpecified) throw std::invalid_argument("backfill requires exactly one of --dry-run or --apply");
            result.apply = value == "--apply"; result.modeSpecified = true;
        } else if (value == "--yes") {
            result.confirmed = true;
        } else if (value.rfind("--experiment-id=", 0) == 0) {
            if (result.experimentId) throw std::invalid_argument("--experiment-id specified more than once");
            result.experimentId = std::stoll(value.substr(16));
        } else if (value.rfind("--semantic-worker-registry=", 0) == 0) {
            result.registryPath = value.substr(27);
        } else {
            throw std::invalid_argument("unsupported scientific provenance backfill option: " + value);
        }
    }
    if (!command || !result.modeSpecified)
        throw std::invalid_argument("backfill requires exactly one of --dry-run or --apply");
    if (result.apply != result.confirmed)
        throw std::invalid_argument("backfill --apply requires --yes; --yes is invalid with --dry-run");
    if (result.experimentId && *result.experimentId <= 0)
        throw std::invalid_argument("--experiment-id must be positive");
    return result;
}

inline bool LowerHex(const std::string& value, std::size_t size)
{
    if (value.size() != size) return false;
    for (const unsigned char c : value)
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
    return true;
}

inline std::string FileSha256(const std::filesystem::path& path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in) return {};
    CC_SHA256_CTX context; CC_SHA256_Init(&context);
    std::array<char, 32768> bytes{};
    while (in.read(bytes.data(), bytes.size()) || in.gcount() > 0)
        CC_SHA256_Update(&context, bytes.data(), static_cast<CC_LONG>(in.gcount()));
    if (!in.eof()) return {};
    std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest{};
    CC_SHA256_Final(digest.data(), &context);
    static constexpr char hex[] = "0123456789abcdef";
    std::string result; result.reserve(digest.size() * 2);
    for (unsigned char byte : digest) { result += hex[byte >> 4U]; result += hex[byte & 15U]; }
    return result;
}

inline std::optional<std::string> JsonString(const std::string& text, const char* key)
{
    const std::regex expression{"\\\"" + std::string{key} + "\\\"\\s*:\\s*\\\"([^\\\"]*)\\\""};
    std::smatch match;
    return std::regex_search(text, match, expression) ? std::optional<std::string>{match[1].str()} : std::nullopt;
}
inline std::optional<int> JsonInt(const std::string& text, const char* key)
{
    const std::regex expression{"\\\"" + std::string{key} + "\\\"\\s*:\\s*([0-9]+)"};
    std::smatch match;
    if (!std::regex_search(text, match, expression)) return std::nullopt;
    try { return std::stoi(match[1].str()); } catch (...) { return std::nullopt; }
}
inline std::string ReadText(const std::filesystem::path& path)
{
    std::ifstream in{path}; std::ostringstream out; out << in.rdbuf(); return in ? out.str() : std::string{};
}

inline bool IsWithin(const std::filesystem::path& value, const std::filesystem::path& root)
{
    const auto mismatch = std::mismatch(root.begin(), root.end(), value.begin(), value.end());
    return mismatch.first == root.end();
}

// Publication installs exactly these two relative links (see
// PublishSemanticWorker.install_runtime_links).  Treating a random symlink in
// a package as provenance would allow the retained runtime inventory to be
// mistaken for a worker-generation binding.
inline std::optional<std::string> RecoverPublishedRuntimeIdentity(
    const std::filesystem::path& artifactRoot,
    const std::filesystem::path& workerDirectory)
{
    std::error_code ec;
    std::optional<std::string> runtimeIdentity;
    const auto root = std::filesystem::canonical(artifactRoot, ec);
    if (ec || root.filename() != "SemanticWorkers" || root.parent_path().filename() != "Builds")
        return std::nullopt;
    for (const char* name : {"default.metallib", "MetaNN.metallib"}) {
        const auto link = workerDirectory / name;
        const auto status = std::filesystem::symlink_status(link, ec);
        if (ec || !std::filesystem::is_symlink(status)) return std::nullopt;
        const auto target = std::filesystem::canonical(link, ec);
        if (ec || !std::filesystem::is_regular_file(target, ec)) return std::nullopt;
        const auto runtimeDirectory = target.parent_path();
        if (runtimeDirectory.parent_path() != root / "runtime" ||
            !LowerHex(runtimeDirectory.filename().string(), 64) ||
            !IsWithin(target, root / "runtime")) return std::nullopt;
        const auto expected = std::filesystem::relative(target, workerDirectory, ec);
        const auto observed = std::filesystem::read_symlink(link, ec);
        if (ec || observed != expected) return std::nullopt;
        const std::string identity = runtimeDirectory.filename().string();
        if (runtimeIdentity && *runtimeIdentity != identity) return std::nullopt;
        runtimeIdentity = identity;
        const auto manifest = runtimeDirectory / "manifest.json";
        const std::string text = ReadText(manifest);
        if (text.empty() || FileSha256(manifest) != identity ||
            !std::regex_search(text, std::regex{"\\\"schema_version\\\"\\s*:\\s*1"}) ||
            !std::regex_search(text, std::regex{"\\\"storage\\\"\\s*:\\s*\\\"immutable\\\""}))
            return std::nullopt;
        // The immutable package contract contains the two named, hashed
        // resources.  Hash them here; the manifest hash binds their names and
        // declared digests to runtime_identity.
        const auto digest = FileSha256(target);
        if (digest.empty()) return std::nullopt;
        const std::string built = name == std::string{"MetaNN.metallib"}
            ? "MetaNN_metal.metallib" : name;
        const std::regex declaration{"\\{\\s*\\\"built_identity\\\"\\s*:\\s*\\\"" + built +
            "\\\"\\s*,\\s*\\\"runtime_name\\\"\\s*:\\s*\\\"" + name +
            "\\\"\\s*,\\s*\\\"sha256\\\"\\s*:\\s*\\\"" + digest + "\\\"\\s*\\}"};
        if (!std::regex_search(text, declaration)) return std::nullopt;
        if (name == std::string{"default.metallib"}) {
            const auto other = runtimeDirectory / "MetaNN.metallib";
            if (!std::filesystem::is_regular_file(other, ec) || FileSha256(other).empty()) return std::nullopt;
        }
    }
    return runtimeIdentity;
}

struct Identity {
    int layout = 0;
    int width = 0;
    std::string role;
    std::string sourceCommit;
    std::string sha256;
    std::string runtimeIdentity;
    std::string executable;
    std::string manifest;
    std::string source;
};
struct Attempt {
    long long id = 0;
    std::string executable;
    std::string reservedAt, completedAt, reconciliation;
    std::optional<int> layout, width;
    std::optional<std::string> role, commit, sha, runtime, manifest;
};
struct Artifact {
    std::string type; long long id = 0, experimentId = 0;
    std::optional<long long> existing;
    std::string timestamp;
    std::string phase, role, kind, capacity;
};

inline bool Successful(const pqxx::row& r, const std::string& phase)
{
    if (r["lifecycle_state"].is_null() || r["completed_at"].is_null() ||
        r["lifecycle_state"].as<std::string>() != "completed") return false;
    if (!r["exit_code"].is_null() && r["exit_code"].as<int>() == 0) return true;
    if (r["reconciliation_result"].is_null()) return false;
    const auto recovered = r["reconciliation_result"].as<std::string>();
    // Missing-process recovery is valid for train, infer, and analyze when
    // the repository found phase-specific durable completion evidence.  The
    // historical failed-result recovery is, by construction, infer-only.
    return recovered == "process_missing_result_recovered" ||
        (phase == "infer" &&
         recovered == "historical_completed_inference_result_recovered");
}

inline std::optional<Identity> ResolvePathIdentity(const Attempt& attempt,
                                                   const Artifact& artifact,
                                                   const std::optional<Scheduler::SemanticWorkerRegistry>& registry)
{
    std::error_code ec;
    if (!attempt.executable.empty()) {
        const auto executable = std::filesystem::canonical(attempt.executable, ec);
        if (!ec && std::filesystem::is_regular_file(executable, ec) && registry) {
        if (const auto* entry = registry->findByCanonicalExecutable(executable.string()); entry != nullptr) {
            const std::string role = entry->role == Scheduler::SemanticWorkerRole::Train ? "train" : "infer";
            if (role == artifact.role)
                return Identity{entry->semanticLayoutVersion, static_cast<int>(entry->modelInputWidth), role,
                                entry->sourceCommit, entry->sha256, entry->runtimeIdentity,
                                entry->canonicalExecutablePath, entry->canonicalManifestPath, "registry"};
        }
        }
    }
    // Retained persisted artifact metadata is authoritative only when it is
    // internally complete; it is never synthesized from a path.
    if (attempt.layout && *attempt.layout > 0 && attempt.width && *attempt.width > 0 &&
        attempt.role && attempt.commit && attempt.sha && attempt.runtime && attempt.manifest &&
        !attempt.executable.empty() && *attempt.role == artifact.role && LowerHex(*attempt.commit, 40) &&
        LowerHex(*attempt.sha, 64) && LowerHex(*attempt.runtime, 64))
        return Identity{*attempt.layout, *attempt.width, *attempt.role, *attempt.commit, *attempt.sha,
                        *attempt.runtime, attempt.executable, attempt.manifest.value_or(""), "retained_artifact_metadata"};
    if (attempt.executable.empty()) return std::nullopt;
    ec.clear();
    const auto executable = std::filesystem::canonical(attempt.executable, ec);
    if (ec || !std::filesystem::is_regular_file(executable, ec)) return std::nullopt;
    // Legacy-only recovery: demand the exact immutable SemanticWorkers layout
    // shape, a matching manifest, and byte-for-byte executable hash.  A
    // manifest alone is not enough; runtime identity is deliberately absent
    // unless a runtime package symlink proves it.
    const auto manifest = executable.parent_path() / "manifest.json";
    const std::string text = ReadText(manifest);
    const auto layout = JsonInt(text, "semantic_layout");
    const auto width = JsonInt(text, "model_input_width");
    const auto commit = JsonString(text, "source_commit");
    const auto sha = JsonString(text, "sha256");
    const auto identity = JsonString(text, "executable_identity");
    const auto storage = JsonString(text, "storage");
    if (!layout || !width || !commit || !sha || !identity || !storage || *storage != "immutable" ||
        !LowerHex(*commit, 40) || !LowerHex(*sha, 64) || FileSha256(executable) != *sha) return std::nullopt;
    const std::string requiredExecutable = artifact.role == "train" ? "LSTM_Release" : "lstm-infer-worker";
    if (*identity != requiredExecutable || executable.filename() != requiredExecutable) return std::nullopt;
    const auto parent = executable.parent_path();
    const auto commitDirectory = parent.parent_path().filename().string();
    const auto shaDirectory = parent.filename().string();
    const bool roleAware = parent.parent_path().parent_path().filename() == "infer";
    const auto layoutDirectory = roleAware
        ? parent.parent_path().parent_path().parent_path().filename().string()
        : parent.parent_path().parent_path().filename().string();
    if (commitDirectory != *commit || shaDirectory != *sha ||
        layoutDirectory != "layout" + std::to_string(*layout) ||
        (artifact.role == "infer" && !roleAware) || (artifact.role == "train" && roleAware)) return std::nullopt;
    const auto artifactRoot = roleAware ? parent.parent_path().parent_path().parent_path().parent_path()
                                        : parent.parent_path().parent_path().parent_path();
    const auto runtimeIdentity = RecoverPublishedRuntimeIdentity(artifactRoot, parent);
    // The only filesystem runtime evidence accepted here is a worker-package
    // local symlink into an immutable runtime package.  The retained global
    // runtime directory says nothing about which package a historical worker
    // used, even when it currently contains one entry.
    if (!runtimeIdentity) return std::nullopt;
    return Identity{*layout, *width, artifact.role, *commit, *sha, *runtimeIdentity,
                    executable.string(), manifest.string(), "validated_semanticworkers_path"};
}

inline bool ConflictsWithAuthoritativeIdentity(const Attempt& attempt,
                                               const Identity& identity)
{
    // Legacy fields are evidence, never a way to overwrite a registry or
    // manifest assertion.  Compare every populated field, including partial
    // records, so corruption fails closed as CONFLICT rather than becoming an
    // apparently harmless unresolved candidate.
    return (attempt.layout && *attempt.layout != identity.layout) ||
        (attempt.width && *attempt.width != identity.width) ||
        (attempt.role && *attempt.role != identity.role) ||
        (attempt.commit && *attempt.commit != identity.sourceCommit) ||
        (attempt.sha && *attempt.sha != identity.sha256) ||
        (attempt.runtime && *attempt.runtime != identity.runtimeIdentity) ||
        (attempt.manifest && !identity.manifest.empty() &&
         *attempt.manifest != identity.manifest);
}

inline std::vector<Attempt> CandidateAttempts(pqxx::transaction_base& tx, const Artifact& artifact,
                                              bool lock = false)
{
    const auto rows = tx.exec(
        "SELECT worker_attempt_id,canonical_executable_path,reserved_at::text,completed_at::text,"
        "reconciliation_result,semantic_layout_version,model_input_width,semantic_worker_role,source_commit,executable_sha256,runtime_identity,canonical_manifest_path,lifecycle_state,exit_code "
        "FROM experiment_scheduler_worker_attempt WHERE experiment_id=$1 AND worker_kind=$2 AND lifecycle_phase=$3 AND capacity_class=$4 "
        "AND reserved_at <= $5::timestamptz AND completed_at >= $5::timestamptz ORDER BY worker_attempt_id" +
        std::string{lock ? " FOR UPDATE;" : ";"},
        pqxx::params{artifact.experimentId, artifact.kind, artifact.phase, artifact.capacity, artifact.timestamp});
    std::vector<Attempt> result;
    for (const auto& row : rows) {
        if (!Successful(row, artifact.phase)) continue;
        Attempt a; a.id = row["worker_attempt_id"].as<long long>();
        a.executable = row["canonical_executable_path"].is_null() ? "" : row["canonical_executable_path"].as<std::string>();
        a.reservedAt = row["reserved_at"].as<std::string>(); a.completedAt = row["completed_at"].as<std::string>();
        a.reconciliation = row["reconciliation_result"].is_null() ? "" : row["reconciliation_result"].as<std::string>();
        if (!row["semantic_layout_version"].is_null()) a.layout = row["semantic_layout_version"].as<int>();
        if (!row["model_input_width"].is_null()) a.width = row["model_input_width"].as<int>();
        if (!row["semantic_worker_role"].is_null()) a.role = row["semantic_worker_role"].as<std::string>();
        if (!row["source_commit"].is_null()) a.commit = row["source_commit"].as<std::string>();
        if (!row["executable_sha256"].is_null()) a.sha = row["executable_sha256"].as<std::string>();
        if (!row["runtime_identity"].is_null()) a.runtime = row["runtime_identity"].as<std::string>();
        if (!row["canonical_manifest_path"].is_null()) a.manifest = row["canonical_manifest_path"].as<std::string>();
        result.push_back(std::move(a));
    }
    return result;
}

struct Selection {
    Outcome outcome = Outcome::missing;
    std::optional<std::pair<Attempt, Identity>> unique;
    std::size_t producerCandidateCount = 0;
    std::size_t resolvedIdentityCount = 0;
    std::size_t unresolvedIdentityCount = 0;
    std::size_t conflictingIdentityCount = 0;
};

inline bool SameIdentity(const Identity& left, const Identity& right)
{
    return left.layout == right.layout && left.width == right.width &&
        left.role == right.role && left.sourceCommit == right.sourceCommit &&
        left.sha256 == right.sha256 && left.runtimeIdentity == right.runtimeIdentity &&
        left.executable == right.executable && left.manifest == right.manifest;
}

// This is deliberately the sole producer-selection state machine.  Candidate
// count is the legitimate lifecycle/experiment/phase/capacity/timestamp
// population, before identity recovery, so an unresolved peer cannot turn two
// plausible producers into one.
inline Selection Select(pqxx::transaction_base& tx, const Artifact& artifact,
                        const std::optional<Scheduler::SemanticWorkerRegistry>& registry,
                        bool lockCandidates = false)
{
    Selection result;
    const auto candidates = CandidateAttempts(tx, artifact, lockCandidates);
    result.producerCandidateCount = candidates.size();
    std::vector<std::pair<Attempt, Identity>> resolved;
    for (const auto& attempt : candidates) {
        const auto identity = ResolvePathIdentity(attempt, artifact, registry);
        if (!identity) { ++result.unresolvedIdentityCount; continue; }
        if (identity->source != "retained_artifact_metadata" &&
            ConflictsWithAuthoritativeIdentity(attempt, *identity)) {
            ++result.conflictingIdentityCount;
            continue;
        }
        resolved.emplace_back(attempt, *identity);
    }
    result.resolvedIdentityCount = resolved.size();
    if (artifact.existing) {
        // A durable binding is immutable.  Still surface a conflict when its
        // own authoritative recovered identity contradicts persisted fields;
        // never select or replace a different producer.
        const auto legitimate = std::find_if(candidates.begin(), candidates.end(), [&](const Attempt& a) { return a.id == *artifact.existing; });
        if (legitimate == candidates.end()) { result.outcome = Outcome::conflict; return result; }
        const auto identity = ResolvePathIdentity(*legitimate, artifact, registry);
        if (!identity || (identity->source != "retained_artifact_metadata" &&
                          ConflictsWithAuthoritativeIdentity(*legitimate, *identity))) {
            // A durable binding without a valid identity is an invalid bound
            // state; unrelated temporal candidates cannot poison it.
            result.outcome = Outcome::conflict;
            return result;
        }
        result.outcome = Outcome::alreadyBound;
        return result;
    }
    if (result.conflictingIdentityCount) { result.outcome = Outcome::conflict; return result; }
    if (result.producerCandidateCount == 0) { result.outcome = Outcome::missing; return result; }
    if (result.producerCandidateCount > 1) { result.outcome = Outcome::ambiguous; return result; }
    if (resolved.size() == 1) {
        result.outcome = Outcome::proposed;
        result.unique = resolved.front();
        return result;
    }
    result.outcome = Outcome::unresolvedIdentity;
    return result;
}

inline std::optional<Artifact> ReadArtifact(pqxx::transaction_base& tx, const Artifact& requested,
                                            bool lock)
{
    const std::string suffix = lock ? " FOR UPDATE;" : ";";
    if (requested.type == "model") {
        const auto rows = tx.exec("SELECT model_id,experiment_id,producer_worker_attempt_id,created_at::text FROM model WHERE model_id=$1" + suffix, pqxx::params{requested.id});
        if (rows.empty()) return std::nullopt;
        const auto& r = rows[0];
        return Artifact{"model", r[0].as<long long>(), r[1].as<long long>(), r[2].is_null() ? std::nullopt : std::optional<long long>{r[2].as<long long>()}, r[3].as<std::string>(), "train", "train", "experiment", "train"};
    }
    const auto rows = tx.exec("SELECT r.id,m.experiment_id,r.producer_worker_attempt_id,r.completed_at::text FROM inference_eval_result r JOIN model m ON m.model_id=r.model_id WHERE r.id=$1 AND r.inference_scope='final' AND r.status='completed'" + suffix, pqxx::params{requested.id});
    if (rows.empty()) return std::nullopt;
    const auto& r = rows[0];
    return Artifact{"final_inference", r[0].as<long long>(), r[1].as<long long>(), r[2].is_null() ? std::nullopt : std::optional<long long>{r[2].as<long long>()}, r[3].as<std::string>(), "infer", "infer", "experiment", "infer"};
}

inline void PersistMissingAttemptIdentity(pqxx::transaction_base& tx, const Attempt& attempt,
                                          const Identity& identity)
{
    if (ConflictsWithAuthoritativeIdentity(attempt, identity))
        throw std::runtime_error("scientific_execution_provenance_attempt_identity_conflict");
    tx.exec("UPDATE experiment_scheduler_worker_attempt SET "
            "semantic_layout_version=COALESCE(semantic_layout_version,$1),"
            "model_input_width=COALESCE(model_input_width,$2),"
            "semantic_worker_role=COALESCE(semantic_worker_role,$3),"
            "source_commit=COALESCE(source_commit,$4),"
            "executable_sha256=COALESCE(executable_sha256,$5),"
            "runtime_identity=COALESCE(runtime_identity,$6),"
            "canonical_manifest_path=COALESCE(canonical_manifest_path,$7) "
            "WHERE worker_attempt_id=$8;",
            pqxx::params{identity.layout, identity.width, identity.role, identity.sourceCommit,
                         identity.sha256, identity.runtimeIdentity,
                         identity.manifest.empty() ? std::optional<std::string>{} : std::optional<std::string>{identity.manifest},
                         attempt.id});
}

inline std::string ConnectionString()
{
    const char* db = std::getenv("LSTM_DB_NAME"); const char* user = std::getenv("LSTM_DB_USER");
    const char* host = std::getenv("LSTM_DB_HOST"); const char* port = std::getenv("LSTM_DB_PORT");
    return "host=" + std::string{host && *host ? host : "127.0.0.1"} +
        " port=" + std::string{port && *port ? port : "5432"} +
        " user=" + std::string{user && *user ? user : "pqxx"} +
        " dbname=" + std::string{db && *db ? db : "LSTM"};
}

inline void RequireColumns(pqxx::transaction_base& tx)
{
    for (const auto& item : std::vector<std::pair<std::string, std::string>>{
             {"model", "producer_worker_attempt_id"}, {"inference_eval_result", "producer_worker_attempt_id"},
             {"experiment_scheduler_worker_attempt", "runtime_identity"}}) {
        if (tx.exec("SELECT 1 FROM information_schema.columns WHERE table_schema='public' AND table_name=$1 AND column_name=$2;", pqxx::params{item.first, item.second}).empty())
            throw std::runtime_error("scientific_execution_provenance_backfill_requires_migration_096:" + item.first + "." + item.second);
    }
}

inline std::vector<Artifact> LoadArtifacts(pqxx::transaction_base& tx, const Options& options)
{
    const std::string filter = options.experimentId ? " AND m.experiment_id=" + std::to_string(*options.experimentId) : "";
    std::vector<Artifact> result;
    for (const auto& row : tx.exec("SELECT m.model_id,m.experiment_id,m.producer_worker_attempt_id,m.created_at::text FROM model m WHERE m.experiment_id IS NOT NULL" + filter + ";"))
        result.push_back({"model", row[0].as<long long>(), row[1].as<long long>(), row[2].is_null() ? std::nullopt : std::optional<long long>{row[2].as<long long>()}, row[3].as<std::string>(), "train", "train", "experiment", "train"});
    const std::string inferFilter = options.experimentId ? " AND m.experiment_id=" + std::to_string(*options.experimentId) : "";
    for (const auto& row : tx.exec("SELECT r.id,m.experiment_id,r.producer_worker_attempt_id,r.completed_at::text FROM inference_eval_result r JOIN model m ON m.model_id=r.model_id WHERE r.inference_scope='final' AND r.status='completed' AND m.experiment_id IS NOT NULL" + inferFilter + ";"))
        result.push_back({"final_inference", row[0].as<long long>(), row[1].as<long long>(), row[2].is_null() ? std::nullopt : std::optional<long long>{row[2].as<long long>()}, row[3].as<std::string>(), "infer", "infer", "experiment", "infer"});
    return result;
}

inline int RunCli(int argc, const char* const argv[])
{
    try {
        const Options options = Parse(argc, argv);
        pqxx::connection connection{ConnectionString()};
        std::optional<Scheduler::SemanticWorkerRegistry> registry;
        try {
            Scheduler::SemanticWorkerRegistryLoadRequest request;
            request.registryPath = options.registryPath;
            registry = Scheduler::SemanticWorkerRegistry::Load(request);
        }
        catch (const std::exception& error) { std::cerr << "SCIENTIFIC_PROVENANCE_BACKFILL,registry_unavailable=" << error.what() << "\n"; }
        std::vector<Artifact> artifacts;
        { pqxx::read_transaction read{connection}; RequireColumns(read); artifacts = LoadArtifacts(read, options); }
        long long proposed = 0, writes = 0;
        for (const auto& artifact : artifacts) {
            Selection selection;
            { pqxx::read_transaction read{connection}; selection = Select(read, artifact, registry); }
            if (selection.outcome == Outcome::proposed) ++proposed;
            const auto text = [](Outcome o) { switch(o) { case Outcome::proposed:return "PROPOSED"; case Outcome::missing:return "MISSING"; case Outcome::ambiguous:return "AMBIGUOUS"; case Outcome::conflict:return "CONFLICT"; case Outcome::unresolvedIdentity:return "UNRESOLVED_IDENTITY"; case Outcome::alreadyBound:return "ALREADY_BOUND";} return "UNKNOWN"; };
            std::cout << "SCIENTIFIC_PROVENANCE_BACKFILL,mode=" << (options.apply ? "apply" : "dry-run") << ",outcome=" << text(selection.outcome)
                      << ",artifact_type=" << artifact.type << ",artifact_id=" << artifact.id << ",experiment_id=" << artifact.experimentId
                      << ",producer_candidate_count=" << selection.producerCandidateCount
                      << ",resolved_identity_count=" << selection.resolvedIdentityCount
                      << ",unresolved_identity_count=" << selection.unresolvedIdentityCount
                      << ",conflicting_identity_count=" << selection.conflictingIdentityCount
                      << ",timestamp_evidence=" << artifact.timestamp;
            if (selection.unique) { const auto& [a, id] = *selection.unique; std::cout << ",worker_attempt_id=" << a.id << ",phase=" << artifact.phase << ",role=" << id.role << ",semantic_layout=" << id.layout << ",model_input_width=" << id.width << ",source_commit=" << id.sourceCommit << ",executable_sha256=" << id.sha256 << ",runtime_identity=" << id.runtimeIdentity << ",canonical_executable=" << id.executable << ",canonical_manifest=" << id.manifest << ",identity_provenance=" << id.source << ",terminal_success_evidence=" << (a.reconciliation.empty() ? "exit_code_zero" : a.reconciliation); }
            std::cout << "\n";
            if (options.apply && selection.unique) {
                pqxx::work write{connection};
                write.exec("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE;");
                const auto current = ReadArtifact(write, artifact, true);
                // Lock the artifact and every temporally eligible attempt before
                // re-selection.  The identity completion and producer binding
                // below therefore operate on the exact revalidated rows in this
                // serializable transaction.
                const Selection revalidated = current ? Select(write, *current, registry, true) : Selection{};
                if (!current || current->existing != artifact.existing ||
                    revalidated.outcome != Outcome::proposed || !revalidated.unique ||
                    revalidated.unique->first.id != selection.unique->first.id ||
                    !SameIdentity(revalidated.unique->second, selection.unique->second)) {
                    write.abort();
                    std::cerr << "SCIENTIFIC_PROVENANCE_BACKFILL,apply_revalidation_failed,artifact_type="
                              << artifact.type << ",artifact_id=" << artifact.id << "\n";
                    continue;
                }
                // This only fills NULL identity columns after the freshly locked
                // row agreed with the authoritative recovery.  The binding is
                // deliberately the next statement in the same transaction.
                PersistMissingAttemptIdentity(write, revalidated.unique->first, revalidated.unique->second);
                const std::string table = artifact.type == "model" ? "model" : "inference_eval_result";
                const std::string idColumn = artifact.type == "model" ? "model_id" : "id";
                const auto updated = write.exec("UPDATE " + table + " SET producer_worker_attempt_id=$1 WHERE " + idColumn + "=$2 AND producer_worker_attempt_id IS NULL RETURNING " + idColumn + ";", pqxx::params{revalidated.unique->first.id, artifact.id});
                if (updated.size() == 1) { write.commit(); ++writes; }
            }
        }
        std::cout << "SCIENTIFIC_PROVENANCE_BACKFILL_SUMMARY,mode=" << (options.apply ? "apply" : "dry-run") << ",proposed=" << proposed << ",writes=" << writes << "\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << "SCIENTIFIC_PROVENANCE_BACKFILL_FAILED,error=" << error.what() << "\n"; return 1; }
}
} // namespace EA::ScientificExecutionProvenanceBackfill
