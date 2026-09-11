#include "../Sources/SchedulerCore/PostgresSchedulerRepository.hpp"

#include <cassert>
#include <optional>
#include <pqxx/pqxx>
#include <string>

int main(int argc, char* argv[])
{
    assert(argc == 2);
    using namespace EA::SchedulerCore;

    pqxx::connection connection{argv[1]};
    pqxx::work transaction{connection};
    transaction.exec(R"SQL(
        CREATE TABLE experiment_global_control(
            singleton boolean PRIMARY KEY,
            active_request_id bigint
        );
        INSERT INTO experiment_global_control VALUES(true, 77);
        CREATE TABLE experiment(
            experiment_id bigint PRIMARY KEY,
            symbol text NOT NULL,
            prediction_horizon integer NOT NULL,
            c_next_threshold double precision NOT NULL,
            core_lr_mult double precision,
            head_lr_mult double precision,
            target_epochs integer NOT NULL,
            checkpoint_interval integer NOT NULL,
            train_start timestamptz NOT NULL,
            train_end timestamptz NOT NULL,
            infer_start timestamptz,
            infer_end timestamptz,
            last_model_id bigint,
            resume_model_id bigint,
            train_log_path text,
            infer_log_path text,
            analysis_log_path text,
            donchian20_mode text NOT NULL,
            feature_warmup_scope text NOT NULL,
            donchian_lookback text NOT NULL,
            feature_ablation_mask text NOT NULL,
            resume_expand_input_width boolean NOT NULL,
            training_objective_canonical text NOT NULL,
            training_objective_hash text NOT NULL,
            scheduler_priority text NOT NULL DEFAULT 'normal',
            resume_requested boolean NOT NULL DEFAULT false,
            scheduler_resume_origin text NOT NULL DEFAULT 'none',
            active_scheduler_worker_attempt_id bigint,
            cancellation_request_id bigint,
            cancel_after_checkpoint_epoch integer,
            status text NOT NULL,
            phase text NOT NULL,
            worker_pid integer,
            worker_process_group_id integer,
            worker_process_start_identity text,
            worker_executable text,
            worker_command_line text,
            worker_started_at timestamptz,
            updated_at timestamptz NOT NULL,
            completed_at timestamptz,
            exit_code integer,
            error_message text
        );
        CREATE TABLE experiment_checkpoint_eval(
            checkpoint_eval_id bigint PRIMARY KEY,
            status text NOT NULL,
            phase text NOT NULL,
            worker_pid integer,
            worker_process_group_id integer,
            worker_process_start_identity text,
            worker_executable text,
            worker_command_line text,
            active_scheduler_worker_attempt_id bigint,
            completed_at timestamptz,
            error_message text,
            updated_at timestamptz
        );
        CREATE TABLE experiment_scheduler_worker_attempt(
            worker_attempt_id bigint PRIMARY KEY,
            scheduler_invocation_id text,
            scheduler_fencing_token bigint,
            lifecycle_state text NOT NULL,
            worker_pid integer,
            worker_process_group_id integer,
            worker_process_start_identity text,
            canonical_executable_path text,
            command_line text,
            spawned_at timestamptz,
            last_observed_at timestamptz,
            completed_at timestamptz,
            exit_code integer,
            reconciliation_result text,
            diagnostic text
        );
        CREATE TABLE experiment_scheduler_protocol(
            singleton boolean PRIMARY KEY,
            required_generation integer NOT NULL,
            cutover_state text NOT NULL,
            failure_diagnostic text,
            cutover_completed_at timestamptz,
            cutover_completed_by text,
            cutover_executable_path text,
            cutover_process_evidence text,
            updated_at timestamptz
        );
        INSERT INTO experiment_scheduler_protocol(
            singleton,required_generation,cutover_state,updated_at
        ) VALUES(true,52,'complete',clock_timestamp());
        CREATE TABLE experiment_scheduler_invocation(
            scheduler_invocation_id text PRIMARY KEY,
            process_pid integer NOT NULL,
            process_group_id integer NOT NULL,
            process_start_identity text NOT NULL,
            canonical_executable_path text NOT NULL,
            command_line text NOT NULL,
            invocation_nonce text NOT NULL,
            status text NOT NULL,
            protocol_generation integer NOT NULL,
            ownership_acquired_at timestamptz,
            ownership_released_at timestamptz,
            last_heartbeat_at timestamptz,
            ended_at timestamptz,
            terminal_reason text
        );
        CREATE TABLE experiment_scheduler_lease(
            singleton boolean PRIMARY KEY,
            owner_scheduler_invocation_id text,
            fencing_token bigint NOT NULL,
            authority_state text NOT NULL,
            acquired_at timestamptz,
            heartbeat_at timestamptz,
            expires_at timestamptz,
            released_at timestamptz,
            transition_reason text
        );
        INSERT INTO experiment_scheduler_lease(
            singleton,fencing_token,authority_state,expires_at
        ) VALUES(true,8,'vacant',clock_timestamp());
    )SQL");

    transaction.exec(R"SQL(
        INSERT INTO experiment(
            experiment_id,symbol,prediction_horizon,c_next_threshold,
            core_lr_mult,head_lr_mult,target_epochs,checkpoint_interval,
            train_start,train_end,infer_start,infer_end,last_model_id,
            resume_model_id,train_log_path,infer_log_path,analysis_log_path,
            donchian20_mode,feature_warmup_scope,donchian_lookback,
            feature_ablation_mask,resume_expand_input_width,
            training_objective_canonical,training_objective_hash,
            scheduler_priority,resume_requested,scheduler_resume_origin,
            status,phase,updated_at
        ) VALUES
        (1,'normalpreempt',4,0.1,NULL,2.5,20,10,'2020-01-01','2021-01-01',
         NULL,'2022-01-01',NULL,88,NULL,'infer.log',NULL,
         'enabled','full_history_warmup','20','none',false,'objective','hash',
         'normal',true,'preemption','pending','train','2026-01-01 00:00:00+00'),
        (2,'highnone',4,0.1,1.0,1.0,20,10,'2020-01-01','2021-01-01',
         '2021-01-01','2022-01-01',10,NULL,'train.log',NULL,'analysis.log',
         'enabled','full_history_warmup','20','none',false,'objective','hash',
         'high',false,'none','pending','train','2026-01-01 00:00:02+00'),
        (3,'normaloperatorlater',4,0.1,1.0,1.0,20,10,'2020-01-01','2021-01-01',
         NULL,NULL,NULL,NULL,NULL,NULL,NULL,'enabled','full_history_warmup','20','none',
         false,'objective','hash','normal',true,'operator','pending','train',
         '2026-01-01 00:00:02+00'),
        (4,'normaloperatorearlier',4,0.1,1.0,1.0,20,10,'2020-01-01','2021-01-01',
         NULL,NULL,NULL,NULL,NULL,NULL,NULL,'enabled','full_history_warmup','20','none',
         false,'objective','hash','normal',true,'operator','pending','train',
         '2026-01-01 00:00:01+00'),
        (5,'lowoperator',4,0.1,1.0,1.0,20,10,'2020-01-01','2021-01-01',
         NULL,NULL,NULL,NULL,NULL,NULL,NULL,'enabled','full_history_warmup','20','none',
         false,'objective','hash','low',true,'operator','pending','train',
         '2025-01-01 00:00:00+00'),
        (10,'running',4,0.1,NULL,NULL,20,10,'2020-01-01','2021-01-01',
         NULL,NULL,NULL,NULL,NULL,NULL,NULL,'enabled','full_history_warmup','20',
         'none',false,'objective','hash','normal',false,'none','running','train',
         '2026-01-02 00:00:00+00');
        UPDATE experiment SET active_scheduler_worker_attempt_id=101
        WHERE experiment_id=10;
        INSERT INTO experiment_scheduler_worker_attempt(
            worker_attempt_id,scheduler_invocation_id,
            scheduler_fencing_token,lifecycle_state
        ) VALUES
            (101,'scheduler-a',7,'reserved'),
            (102,'scheduler-a',7,'reserved'),
            (103,'scheduler-a',7,'reserved'),
            (201,'scheduler-a',7,'reserved');
        INSERT INTO experiment_checkpoint_eval(
            checkpoint_eval_id,status,phase,
            active_scheduler_worker_attempt_id,updated_at
        ) VALUES(201,'running','infer',201,clock_timestamp());
    )SQL");

    PostgresSchedulerRepository repository{transaction};
    const auto pending = repository.loadPendingExperiments("train", false);
    assert(pending.size() == 5);
    assert(pending[0].experiment.experimentId == 2);
    assert(pending[1].experiment.experimentId == 4);
    assert(pending[2].experiment.experimentId == 3);
    assert(pending[3].experiment.experimentId == 1);
    assert(pending[4].experiment.experimentId == 5);
    const auto& nullRecord = pending[3].experiment;
    assert(!nullRecord.coreLrMult);
    assert(nullRecord.headLrMult == 2.5);
    assert(!nullRecord.inferStart);
    assert(nullRecord.inferEnd.has_value());
    assert(!nullRecord.lastModelId);
    assert(nullRecord.resumeModelId == 88);
    assert(!nullRecord.trainLogPath);
    assert(nullRecord.inferLogPath == "infer.log");
    assert(!nullRecord.analysisLogPath);
    assert(!pending[3].activeWorkerAttemptId);

    const auto running = repository.loadRunningExperiments();
    assert(running.size() == 1);
    assert(running[0].experiment.experimentId == 10);
    assert(!running[0].workerPid);
    const auto queue = repository.loadQueueSnapshot();
    assert(queue.pendingTrain == 5);
    assert(queue.runningTrain == 1);
    assert(queue.pendingInfer == 0 && queue.runningInfer == 0);

    const SpawnedWorkerAttemptUpdate spawned{
        101,
        "scheduler-a",
        7,
        10,
        std::nullopt,
        "train",
        43210,
        "start-identity",
        "/tmp/LSTM_Release",
        "/tmp/LSTM_Release --train"};
    assert(repository.persistSpawnedWorkerAttempt(spawned) ==
           SpawnPersistenceResult::Updated);
    const pqxx::row mapped = transaction.exec(
        "SELECT a.lifecycle_state,a.worker_pid,a.worker_process_group_id,"
        "a.worker_process_start_identity,a.canonical_executable_path,"
        "a.command_line,e.worker_pid,e.worker_process_group_id,"
        "e.worker_process_start_identity,e.worker_executable,e.worker_command_line "
        "FROM experiment_scheduler_worker_attempt a CROSS JOIN experiment e "
        "WHERE a.worker_attempt_id=101 AND e.experiment_id=10").one_row();
    assert(mapped[0].as<std::string>() == "spawned");
    assert(mapped[1].as<int>() == 43210 && mapped[2].as<int>() == 43210);
    assert(mapped[3].as<std::string>() == "start-identity");
    assert(mapped[4].as<std::string>() == "/tmp/LSTM_Release");
    assert(mapped[5].as<std::string>() == "/tmp/LSTM_Release --train");
    assert(mapped[6].as<int>() == 43210 && mapped[7].as<int>() == 43210);
    assert(mapped[8].as<std::string>() == "start-identity");
    assert(mapped[9].as<std::string>() == "/tmp/LSTM_Release");
    assert(mapped[10].as<std::string>() == "/tmp/LSTM_Release --train");

    SpawnedWorkerAttemptUpdate wrongFence = spawned;
    wrongFence.workerAttemptId = 102;
    wrongFence.schedulerFencingToken = 8;
    assert(repository.persistSpawnedWorkerAttempt(wrongFence) ==
           SpawnPersistenceResult::AttemptPreconditionRejected);

    SpawnedWorkerAttemptUpdate checkpointSpawn = spawned;
    checkpointSpawn.workerAttemptId = 201;
    checkpointSpawn.experimentId = 11;
    checkpointSpawn.checkpointEvalId = 201;
    checkpointSpawn.phase = "infer";
    checkpointSpawn.workerPid = 43211;
    checkpointSpawn.commandLine = "/tmp/LSTM_Release --infer";
    assert(repository.persistSpawnedWorkerAttempt(checkpointSpawn) ==
           SpawnPersistenceResult::Updated);
    const pqxx::row checkpointMapped = transaction.exec(
        "SELECT worker_pid,worker_process_group_id,"
        "worker_process_start_identity,worker_executable,worker_command_line "
        "FROM experiment_checkpoint_eval WHERE checkpoint_eval_id=201")
        .one_row();
    assert(checkpointMapped[0].as<int>() == 43211);
    assert(checkpointMapped[1].as<int>() == 43211);
    assert(checkpointMapped[2].as<std::string>() == "start-identity");
    assert(checkpointMapped[3].as<std::string>() == "/tmp/LSTM_Release");
    assert(checkpointMapped[4].as<std::string>() ==
           "/tmp/LSTM_Release --infer");

    transaction.exec(
        "UPDATE experiment SET active_scheduler_worker_attempt_id=103,"
        "worker_pid=NULL,worker_process_group_id=NULL WHERE experiment_id=10;");
    assert(repository.persistWorkerAttemptLaunchFailure({
               103,
               "scheduler-a",
               7,
               10,
               std::nullopt,
               "train",
               127,
               "launch_failed_errno_2"}) ==
           LaunchFailurePersistenceResult::Updated);
    const pqxx::row failed = transaction.exec(
        "SELECT a.lifecycle_state,a.reconciliation_result,a.exit_code,"
        "e.status,e.active_scheduler_worker_attempt_id,e.error_message "
        "FROM experiment_scheduler_worker_attempt a CROSS JOIN experiment e "
        "WHERE a.worker_attempt_id=103 AND e.experiment_id=10").one_row();
    assert(failed[0].as<std::string>() == "launch_failed");
    assert(failed[1].as<std::string>() == "launch_failed");
    assert(failed[2].as<int>() == 127);
    assert(failed[3].as<std::string>() == "failed");
    assert(failed[4].is_null());
    assert(failed[5].as<std::string>() == "launch_failed_errno_2");

    repository.acquireAuthorityCoordinationLock();
    const auto protocol = repository.loadSchedulerProtocolForUpdate();
    assert(protocol && protocol->requiredGeneration == 52);
    assert(protocol->cutoverState == "complete");
    repository.registerSchedulerInvocation({
        "scheduler-repository-test",
        900,
        900,
        "scheduler-start",
        "/tmp/LSTM_Release",
        "/tmp/LSTM_Release --schedule-experiments",
        "repository-test-nonce",
        52});
    const auto lease = repository.loadSchedulerLeaseForUpdate();
    assert(lease && !lease->ownerSchedulerInvocationId);
    assert(lease->fencingToken == 8 && lease->authorityState == "vacant");
    assert(repository.acquireSchedulerLease({
        "scheduler-repository-test", 9, 90, "vacant"}));
    repository.markSchedulerInvocationOwner("scheduler-repository-test");
    const SchedulerAuthorityIdentity authority{
        "scheduler-repository-test", 9};
    assert(repository.renewSchedulerLease(authority, 90));
    repository.touchSchedulerInvocation("scheduler-repository-test");
    assert(!repository.renewSchedulerLease(
        {"scheduler-repository-test", 10}, 90));
    assert(repository.displaceSchedulerAuthorityForTest(
        authority, "scheduler-foreign", "test_failpoint:repository", 90));
    assert(!repository.releaseSchedulerLease(authority, "stale_release"));
    const SchedulerAuthorityIdentity foreign{"scheduler-foreign", 10};
    assert(repository.releaseSchedulerLease(foreign, "scheduler_exit"));

    transaction.exec(
        "UPDATE experiment_scheduler_protocol SET cutover_state='pending' "
        "WHERE singleton=true;");
    assert(repository.completeSchedulerProtocolCutover({
        52,
        "pid:900;start:scheduler-start",
        "/tmp/LSTM_Release",
        "ps_inspection_complete;active_scheduler_dispatch_processes=0"}));
    assert(transaction.exec(
               "SELECT cutover_state FROM experiment_scheduler_protocol "
               "WHERE singleton=true").one_row()[0].as<std::string>() ==
           "complete");

    transaction.abort();
    return 0;
}
