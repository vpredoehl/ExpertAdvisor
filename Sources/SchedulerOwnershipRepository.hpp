#pragma once

#include <optional>
#include <stdexcept>
#include <string>

#include <pqxx/pqxx>

namespace EA::SchedulerOwnership
{

inline constexpr int kProtocolGeneration = 52;
inline constexpr const char* kProtocolSessionSetting =
    "expertadvisor.scheduler_protocol_generation";

// Canonical scheduler lock order:
//   1. global coordination advisory lock
//   2. scheduler protocol/cutover singleton
//   3. scheduler lease singleton
//   4. scheduler invocation / administrative request rows
//   5. exact worker-attempt rows (ascending worker_attempt_id)
//   6. experiment rows (ascending experiment_id)
//   7. checkpoint-evaluation rows (ascending checkpoint_eval_id)
//   8. continuation / checkpoint-analysis result rows
//   9. campaign/recommendation rows in their documented service order
//
// A caller that does not need an earlier lock starts at the first lock it
// needs and must never acquire an earlier lock later in the transaction.
// OS process inspection is performed outside database locks except for the
// immediate verify-and-signal critical section.

struct SchedulerAuthorityContext
{
    std::string schedulerInvocationId;
    std::string invocationNonce;
    long long fencingToken = 0;
    std::string canonicalExecutablePath;
    bool held = false;

    [[nodiscard]] bool Complete() const noexcept
    {
        return held && !schedulerInvocationId.empty() &&
               fencingToken > 0 &&
               !canonicalExecutablePath.empty();
    }
};

struct ExactAttemptExpectation
{
    long long workerAttemptId = -1;
    long long experimentId = -1;
    std::optional<long long> checkpointEvalId;
    std::string workerKind;
    std::string lifecyclePhase;
    std::string capacityClass;
    std::optional<std::string> schedulerInvocationId;
    std::optional<long long> schedulerFencingToken;
    // Most callers intentionally reject identity_ambiguous attempts.  The
    // reconciliation command is the sole caller that supplies this explicit
    // source-state fence before it may inspect and restore one.
    std::optional<std::string> requiredLifecycleState;
    bool requireSignalable = false;
    bool requireCompleteProcessIdentity = false;
    bool allowTerminalLifecycle = false;
};

struct ExactAttemptSnapshot
{
    long long workerAttemptId = -1;
    std::string launchAttemptIdentity;
    std::optional<std::string> schedulerInvocationId;
    std::optional<long long> schedulerFencingToken;
    long long experimentId = -1;
    std::optional<long long> checkpointEvalId;
    std::string workerKind;
    std::string lifecyclePhase;
    std::string capacityClass;
    std::string ownershipOrigin;
    std::string lifecycleState;
    std::optional<int> workerPid;
    std::optional<int> processGroupId;
    std::optional<std::string> processStartIdentity;
    std::optional<std::string> canonicalExecutablePath;
    std::optional<std::string> commandLine;
    std::string commandIdentity;
    std::string lifecycleStatus;
    std::string lifecycleRowPhase;
    std::optional<int> lifecyclePid;
    std::optional<int> lifecycleProcessGroupId;
    std::optional<std::string> lifecycleProcessStartIdentity;
    std::optional<std::string> lifecycleExecutable;
    std::optional<std::string> lifecycleCommandLine;
};

struct ExactTerminalAttemptSnapshot
{
    long long workerAttemptId = -1;
    std::optional<std::string> schedulerInvocationId;
    std::optional<long long> schedulerFencingToken;
    long long experimentId = -1;
    std::optional<long long> checkpointEvalId;
    std::string workerKind;
    std::string lifecyclePhase;
    std::string capacityClass;
    std::string lifecycleState;
    std::optional<int> workerPid;
    std::optional<int> processGroupId;
    std::optional<std::string> processStartIdentity;
    std::optional<std::string> canonicalExecutablePath;
    std::optional<std::string> commandLine;
    std::string commandIdentity;
    std::string reconciliationResult;
    std::string diagnostic;
    std::optional<int> exitCode;
    std::optional<int> signalNumber;
};

inline bool IsActiveAttemptState(const std::string& state) noexcept
{
    return state == "reserved" || state == "spawned" ||
           state == "running" || state == "observed" ||
           state == "stopped" || state == "identity_ambiguous";
}

inline bool IsSignalableAttemptState(const std::string& state) noexcept
{
    return state == "spawned" || state == "running" ||
           state == "observed" || state == "stopped";
}

inline void SetCorrectedSchedulerProtocolSession(
    pqxx::transaction_base& transaction)
{
    transaction.exec(
        "SELECT set_config('" +
        std::string{kProtocolSessionSetting} +
        "','" + std::to_string(kProtocolGeneration) + "',true);");
}

inline bool OptionalIntegerEquals(
    const std::optional<int>& lhs,
    const std::optional<int>& rhs) noexcept
{
    return lhs == rhs;
}

inline bool OptionalTextEquals(
    const std::optional<std::string>& lhs,
    const std::optional<std::string>& rhs) noexcept
{
    return lhs == rhs;
}

inline std::optional<ExactAttemptSnapshot> LockAndVerifyExactActiveAttempt(
    pqxx::transaction_base& transaction,
    const ExactAttemptExpectation& expected,
    bool lockRows = true)
{
    if (expected.workerAttemptId <= 0 ||
        expected.experimentId <= 0 ||
        expected.workerKind.empty() ||
        expected.lifecyclePhase.empty() ||
        expected.capacityClass.empty())
    {
        return std::nullopt;
    }

    std::string attemptSql =
        "SELECT worker_attempt_id,launch_attempt_identity,"
        "scheduler_invocation_id,scheduler_fencing_token,experiment_id,"
        "checkpoint_eval_id,worker_kind,lifecycle_phase,capacity_class,"
        "ownership_origin,lifecycle_state,worker_pid,"
        "worker_process_group_id,worker_process_start_identity,"
        "canonical_executable_path,command_line,command_identity "
        "FROM experiment_scheduler_worker_attempt "
        "WHERE worker_attempt_id=$1";
    if (lockRows)
        attemptSql += " FOR UPDATE";
    pqxx::result attempts =
        transaction.exec_params(attemptSql, expected.workerAttemptId);
    if (attempts.size() != 1)
        return std::nullopt;

    const pqxx::row row = attempts[0];
    ExactAttemptSnapshot snapshot;
    snapshot.workerAttemptId = row[0].as<long long>();
    snapshot.launchAttemptIdentity = row[1].as<std::string>();
    if (!row[2].is_null())
        snapshot.schedulerInvocationId = row[2].as<std::string>();
    if (!row[3].is_null())
        snapshot.schedulerFencingToken = row[3].as<long long>();
    snapshot.experimentId = row[4].as<long long>();
    if (!row[5].is_null())
        snapshot.checkpointEvalId = row[5].as<long long>();
    snapshot.workerKind = row[6].as<std::string>();
    snapshot.lifecyclePhase = row[7].as<std::string>();
    snapshot.capacityClass = row[8].as<std::string>();
    snapshot.ownershipOrigin = row[9].as<std::string>();
    snapshot.lifecycleState = row[10].as<std::string>();
    if (!row[11].is_null())
        snapshot.workerPid = row[11].as<int>();
    if (!row[12].is_null())
        snapshot.processGroupId = row[12].as<int>();
    if (!row[13].is_null())
        snapshot.processStartIdentity = row[13].as<std::string>();
    if (!row[14].is_null())
        snapshot.canonicalExecutablePath = row[14].as<std::string>();
    if (!row[15].is_null())
        snapshot.commandLine = row[15].as<std::string>();
    snapshot.commandIdentity = row[16].as<std::string>();

    if (snapshot.experimentId != expected.experimentId ||
        snapshot.checkpointEvalId != expected.checkpointEvalId ||
        snapshot.workerKind != expected.workerKind ||
        snapshot.lifecyclePhase != expected.lifecyclePhase ||
        snapshot.capacityClass != expected.capacityClass ||
        !IsActiveAttemptState(snapshot.lifecycleState) ||
        (expected.schedulerInvocationId &&
         snapshot.schedulerInvocationId != expected.schedulerInvocationId) ||
        (expected.schedulerFencingToken &&
         snapshot.schedulerFencingToken != expected.schedulerFencingToken) ||
        (expected.requiredLifecycleState &&
         snapshot.lifecycleState != *expected.requiredLifecycleState) ||
        (expected.requireSignalable &&
         !IsSignalableAttemptState(snapshot.lifecycleState)) ||
        (!expected.requiredLifecycleState &&
         snapshot.lifecycleState == "identity_ambiguous"))
    {
        return std::nullopt;
    }

    if (expected.requireCompleteProcessIdentity &&
        (!snapshot.workerPid || !snapshot.processGroupId ||
         !snapshot.processStartIdentity ||
         snapshot.processStartIdentity->empty() ||
         !snapshot.canonicalExecutablePath ||
         snapshot.canonicalExecutablePath->empty() ||
         !snapshot.commandLine || snapshot.commandLine->empty()))
    {
        return std::nullopt;
    }

    pqxx::result lifecycle;
    if (expected.checkpointEvalId)
    {
        std::string sql =
            "SELECT status,phase,worker_pid,worker_process_group_id,"
            "worker_process_start_identity,worker_executable,"
            "worker_command_line "
            "FROM experiment_checkpoint_eval "
            "WHERE checkpoint_eval_id=$1 "
            "AND active_scheduler_worker_attempt_id=$2";
        if (lockRows)
            sql += " FOR UPDATE";
        lifecycle = transaction.exec_params(
            sql, *expected.checkpointEvalId, expected.workerAttemptId);
    }
    else
    {
        std::string sql =
            "SELECT status,phase,worker_pid,worker_process_group_id,"
            "worker_process_start_identity,worker_executable,"
            "worker_command_line "
            "FROM experiment WHERE experiment_id=$1 "
            "AND active_scheduler_worker_attempt_id=$2";
        if (lockRows)
            sql += " FOR UPDATE";
        lifecycle = transaction.exec_params(
            sql, expected.experimentId, expected.workerAttemptId);
    }
    if (lifecycle.size() != 1)
        return std::nullopt;

    snapshot.lifecycleStatus = lifecycle[0][0].as<std::string>();
    snapshot.lifecycleRowPhase = lifecycle[0][1].as<std::string>();
    if (!lifecycle[0][2].is_null())
        snapshot.lifecyclePid = lifecycle[0][2].as<int>();
    if (!lifecycle[0][3].is_null())
        snapshot.lifecycleProcessGroupId = lifecycle[0][3].as<int>();
    if (!lifecycle[0][4].is_null())
        snapshot.lifecycleProcessStartIdentity =
            lifecycle[0][4].as<std::string>();
    if (!lifecycle[0][5].is_null())
        snapshot.lifecycleExecutable = lifecycle[0][5].as<std::string>();
    if (!lifecycle[0][6].is_null())
        snapshot.lifecycleCommandLine = lifecycle[0][6].as<std::string>();

    const std::string requiredPhase =
        expected.workerKind == "checkpoint_infer"
            ? "infer"
            : expected.lifecyclePhase;
    const bool activeLifecycle =
        ((snapshot.lifecycleStatus == "running") ||
         (snapshot.lifecycleState == "stopped" &&
          (snapshot.lifecycleStatus == "paused" ||
           snapshot.lifecycleStatus == "pending"))) &&
        snapshot.lifecycleRowPhase == requiredPhase;
    const bool exactWorkerTerminalLifecycle =
        expected.allowTerminalLifecycle &&
        (snapshot.lifecycleStatus == "completed" ||
         snapshot.lifecycleStatus == "failed" ||
         snapshot.lifecycleStatus == "cancelled") &&
        (snapshot.lifecycleRowPhase == requiredPhase ||
         snapshot.lifecycleRowPhase == "done");
    if (!activeLifecycle && !exactWorkerTerminalLifecycle)
    {
        return std::nullopt;
    }
    if (expected.requireCompleteProcessIdentity &&
        activeLifecycle &&
        (!OptionalIntegerEquals(
             snapshot.workerPid, snapshot.lifecyclePid) ||
         !OptionalIntegerEquals(
             snapshot.processGroupId,
             snapshot.lifecycleProcessGroupId) ||
         !OptionalTextEquals(
             snapshot.processStartIdentity,
             snapshot.lifecycleProcessStartIdentity) ||
         !OptionalTextEquals(
             snapshot.canonicalExecutablePath,
             snapshot.lifecycleExecutable) ||
         !OptionalTextEquals(
             snapshot.commandLine,
             snapshot.lifecycleCommandLine)))
    {
        return std::nullopt;
    }
    return snapshot;
}

// A terminal attempt is intentionally verified without requiring a current
// lifecycle binding.  This is the narrow replay/reap mode used after an
// atomic phase transition has already cleared the exact old binding.  It
// still locks and verifies the immutable attempt identity, scheduler fence,
// process identity, terminal state, and reconciliation reason.
inline std::optional<ExactTerminalAttemptSnapshot>
LockAndVerifyExactTerminalAttempt(
    pqxx::transaction_base& transaction,
    const ExactAttemptExpectation& expected,
    const std::string& terminalState,
    const std::string& reconciliationResult,
    bool lockRow = true)
{
    if (expected.workerAttemptId <= 0 ||
        expected.experimentId <= 0 ||
        expected.workerKind.empty() ||
        expected.lifecyclePhase.empty() ||
        expected.capacityClass.empty() ||
        terminalState.empty() ||
        reconciliationResult.empty())
    {
        return std::nullopt;
    }

    std::string sql =
        "SELECT worker_attempt_id,scheduler_invocation_id,"
        "scheduler_fencing_token,experiment_id,checkpoint_eval_id,"
        "worker_kind,lifecycle_phase,capacity_class,lifecycle_state,"
        "worker_pid,worker_process_group_id,"
        "worker_process_start_identity,canonical_executable_path,"
        "command_line,command_identity,reconciliation_result,"
        "COALESCE(diagnostic,''),exit_code,signal_number "
        "FROM experiment_scheduler_worker_attempt "
        "WHERE worker_attempt_id=$1 AND experiment_id=$2 "
        "AND checkpoint_eval_id IS NOT DISTINCT FROM $3 "
        "AND worker_kind=$4 AND lifecycle_phase=$5 "
        "AND capacity_class=$6 AND lifecycle_state=$7 "
        "AND reconciliation_result=$8 "
        "AND ($9::text IS NULL OR scheduler_invocation_id=$9) "
        "AND ($10::bigint IS NULL OR scheduler_fencing_token=$10)";
    if (lockRow)
        sql += " FOR UPDATE";
    const pqxx::result rows = transaction.exec_params(
        sql,
        expected.workerAttemptId,
        expected.experimentId,
        expected.checkpointEvalId,
        expected.workerKind,
        expected.lifecyclePhase,
        expected.capacityClass,
        terminalState,
        reconciliationResult,
        expected.schedulerInvocationId,
        expected.schedulerFencingToken);
    if (rows.size() != 1)
        return std::nullopt;

    ExactTerminalAttemptSnapshot snapshot;
    snapshot.workerAttemptId = rows[0][0].as<long long>();
    if (!rows[0][1].is_null())
        snapshot.schedulerInvocationId = rows[0][1].as<std::string>();
    if (!rows[0][2].is_null())
        snapshot.schedulerFencingToken = rows[0][2].as<long long>();
    snapshot.experimentId = rows[0][3].as<long long>();
    if (!rows[0][4].is_null())
        snapshot.checkpointEvalId = rows[0][4].as<long long>();
    snapshot.workerKind = rows[0][5].as<std::string>();
    snapshot.lifecyclePhase = rows[0][6].as<std::string>();
    snapshot.capacityClass = rows[0][7].as<std::string>();
    snapshot.lifecycleState = rows[0][8].as<std::string>();
    if (!rows[0][9].is_null())
        snapshot.workerPid = rows[0][9].as<int>();
    if (!rows[0][10].is_null())
        snapshot.processGroupId = rows[0][10].as<int>();
    if (!rows[0][11].is_null())
        snapshot.processStartIdentity = rows[0][11].as<std::string>();
    if (!rows[0][12].is_null())
        snapshot.canonicalExecutablePath = rows[0][12].as<std::string>();
    if (!rows[0][13].is_null())
        snapshot.commandLine = rows[0][13].as<std::string>();
    snapshot.commandIdentity = rows[0][14].as<std::string>();
    snapshot.reconciliationResult = rows[0][15].as<std::string>();
    snapshot.diagnostic = rows[0][16].as<std::string>();
    if (!rows[0][17].is_null())
        snapshot.exitCode = rows[0][17].as<int>();
    if (!rows[0][18].is_null())
        snapshot.signalNumber = rows[0][18].as<int>();

    if (expected.requireCompleteProcessIdentity &&
        (!snapshot.workerPid || !snapshot.processGroupId ||
         !snapshot.processStartIdentity ||
         snapshot.processStartIdentity->empty() ||
         !snapshot.canonicalExecutablePath ||
         snapshot.canonicalExecutablePath->empty() ||
         !snapshot.commandLine || snapshot.commandLine->empty()))
    {
        return std::nullopt;
    }
    return snapshot;
}

inline void RequireAffectedExactlyOne(
    const pqxx::result& result,
    const std::string& mutation)
{
    if (result.affected_rows() != 1)
    {
        throw std::runtime_error(
            "exact_attempt_predicate_rejected:" + mutation +
            ":affected_rows=" +
            std::to_string(result.affected_rows()));
    }
}

} // namespace EA::SchedulerOwnership
