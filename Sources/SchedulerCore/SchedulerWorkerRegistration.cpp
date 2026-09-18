#include "SchedulerWorkerRegistration.hpp"

#include "GlobalExperimentControl.hpp"
#include "SchedulerOwnershipRepository.hpp"
#include "WorkerProcessController.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unistd.h>

#include <pqxx/pqxx>

namespace EA::SchedulerCore
{
namespace
{

std::string EnvironmentOrDefault(const char* name, const char* fallback)
{
    const char* value = std::getenv(name);
    return value != nullptr && *value != '\0'
        ? std::string{value}
        : std::string{fallback};
}

std::string LstmDatabaseConnectionString()
{
    return "hostaddr=" +
           EnvironmentOrDefault("LSTM_DB_HOST", "127.0.0.1") +
           " gssencmode=disable user=pqxx dbname=" +
           EnvironmentOrDefault("LSTM_DB_NAME", "LSTM");
}

} // namespace

bool RegisterSchedulerWorker(
    const SchedulerWorkerRegistration& registration)
{
    if (registration.workerAttemptId <= 0 ||
        (!registration.experimentId && !registration.checkpointEvalId) ||
        registration.workerKind.empty() ||
        registration.lifecyclePhase.empty())
    {
        return false;
    }

    try
    {
        const int pid = static_cast<int>(::getpid());
        const int processGroupId = static_cast<int>(::getpgrp());
        const std::optional<std::string> processStartIdentity =
            EA::GlobalExperimentControl::ReadProcessStartIdentity(pid);
        if (!processStartIdentity)
            throw std::runtime_error(
                "worker_process_start_identity_unavailable");
        const std::string executable =
            NativeWorkerProcessController().resolveExecutablePath();

        pqxx::connection connection{LstmDatabaseConnectionString()};
        pqxx::work transaction{connection};
        transaction.exec("SET TRANSACTION READ WRITE;");
        EA::SchedulerOwnership::SetCorrectedSchedulerProtocolSession(
            transaction);
        pqxx::result identity = transaction.exec_params(
            "SELECT experiment_id FROM "
            "experiment_scheduler_worker_attempt "
            "WHERE worker_attempt_id=$1;",
            registration.workerAttemptId);
        if (identity.size() != 1)
            throw std::runtime_error("worker_attempt_identity_missing");
        const long long experimentId = identity[0][0].as<long long>();
        if (registration.experimentId &&
            *registration.experimentId != experimentId)
        {
            throw std::runtime_error(
                "worker_attempt_experiment_identity_mismatch");
        }

        EA::SchedulerOwnership::ExactAttemptExpectation expected;
        expected.workerAttemptId = registration.workerAttemptId;
        expected.experimentId = experimentId;
        expected.checkpointEvalId = registration.checkpointEvalId;
        expected.workerKind = registration.workerKind;
        expected.lifecyclePhase = registration.lifecyclePhase;
        expected.capacityClass = registration.workerKind == "checkpoint_infer"
            ? "infer"
            : registration.lifecyclePhase;
        expected.requireCompleteProcessIdentity = true;
        const auto exact =
            EA::SchedulerOwnership::LockAndVerifyExactActiveAttempt(
                transaction, expected, true);
        if (!exact ||
            exact->workerPid != std::optional<int>{pid} ||
            exact->processGroupId != std::optional<int>{processGroupId} ||
            exact->processStartIdentity != processStartIdentity ||
            exact->canonicalExecutablePath !=
                std::optional<std::string>{executable})
        {
            throw std::runtime_error(
                "worker_attempt_exact_active_identity_mismatch");
        }

        pqxx::result registered = transaction.exec_params(
            "UPDATE experiment_scheduler_worker_attempt SET "
            "lifecycle_state='running',"
            "registered_at=COALESCE(registered_at,clock_timestamp()),"
            "last_observed_at=clock_timestamp() "
            "WHERE worker_attempt_id=$1 AND worker_pid=$2 "
            "AND worker_process_group_id=$3 "
            "AND worker_process_start_identity=$4 "
            "AND canonical_executable_path=$5 "
            "AND experiment_id=$6 "
            "AND checkpoint_eval_id IS NOT DISTINCT FROM $7 "
            "AND worker_kind=$8 AND lifecycle_phase=$9 "
            "AND lifecycle_state IN ('spawned','running') "
            "RETURNING experiment_id,checkpoint_eval_id,worker_kind,"
            "lifecycle_phase;",
            registration.workerAttemptId,
            pid,
            processGroupId,
            *processStartIdentity,
            executable,
            experimentId,
            registration.checkpointEvalId,
            registration.workerKind,
            registration.lifecyclePhase);
        if (registered.size() != 1)
        {
            transaction.abort();
            std::cerr << "SCHEDULER_WORKER_REGISTRATION_REJECTED"
                      << ",worker_attempt_id="
                      << registration.workerAttemptId
                      << ",pid=" << pid
                      << ",reason=durable_identity_mismatch"
                      << std::endl;
            return false;
        }

        transaction.commit();
        std::cout << "SCHEDULER_WORKER_REGISTERED"
                  << ",worker_attempt_id=" << registration.workerAttemptId
                  << ",experiment_id="
                  << registered[0][0].as<long long>()
                  << ",checkpoint_eval_id="
                  << (registered[0][1].is_null()
                          ? "NULL"
                          : registered[0][1].c_str())
                  << ",worker_kind="
                  << registered[0][2].as<std::string>()
                  << ",phase="
                  << registered[0][3].as<std::string>()
                  << ",pid=" << pid
                  << std::endl;
        return true;
    }
    catch (const std::exception& error)
    {
        std::cerr << "SCHEDULER_WORKER_REGISTRATION_FAILED"
                  << ",worker_attempt_id=" << registration.workerAttemptId
                  << ",pid=" << static_cast<int>(::getpid())
                  << ",error=" << error.what()
                  << std::endl;
        return false;
    }
}

} // namespace EA::SchedulerCore
