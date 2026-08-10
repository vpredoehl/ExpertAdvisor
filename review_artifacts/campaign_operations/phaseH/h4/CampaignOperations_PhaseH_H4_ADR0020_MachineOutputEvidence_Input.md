================================================================================
FILE: Sources/CampaignOperationsManagerService.cpp
RANGE: output formatting / terminal result / stop record
================================================================================
    45	    if (const auto* sqlError = dynamic_cast<const pqxx::sql_error*>(&error))
    46	    {
    47	        const auto& state = sqlError->sqlstate();
    48	        if (state == "42501" || state.rfind("08", 0U) == 0U ||
    49	            state.rfind("42", 0U) == 0U ||
    50	            state.rfind("53", 0U) == 0U || state.rfind("57", 0U) == 0U ||
    51	            state.rfind("58", 0U) == 0U)
    52	            return state == "42501"
    53	                ? ManagerGlobalStopReason::privilegeFailure
    54	                : ManagerGlobalStopReason::databaseFailure;
    55	    }
    56	    if (const auto* campaignError = dynamic_cast<const Error*>(&error);
    57	        campaignError &&
    58	        (campaignError->code() == ErrorCode::persistenceCorruption ||
    59	         campaignError->code() == ErrorCode::invalidCanonicalText ||
    60	         campaignError->code() == ErrorCode::invalidOperationalRequest))
    61	        return ManagerGlobalStopReason::databaseFailure;
    62	    return ManagerGlobalStopReason::none;
    63	}
    64
    65	void RenderRequest(std::ostream& output, const ManagerRequestResult& result)
    66	{
    67	    output << "CAMPAIGN_OPERATIONS_MANAGER_REQUEST"
    68	           << ",request_id=" << result.requestId.value()
    69	           << ",expected_request_version=" << result.expectedRequestVersion
    70	           << ",operation_key=" << result.operationKey
    71	           << ",outcome=" << ToText(result.outcome)
    72	           << ",dispatch_classification="
    73	           << (result.dispatchClassification.empty()
    74	                   ? "none" : result.dispatchClassification)
    75	           << ",replay_disposition="
    76	           << (result.replayDisposition.empty()
    77	                   ? "none" : result.replayDisposition)
    78	           << ",newly_committed="
    79	           << (result.newlyCommitted ? "true" : "false")
    80	           << ",exact_replay=" << (result.exactReplay ? "true" : "false")
    81	           << ",diagnostic_code="
    82	           << (result.diagnosticCode.empty() ? "none" : result.diagnosticCode)
    83	           << '\n';
    84	}
    85
    86	} // namespace
    87
    88	std::string ToText(ManagerRequestOutcomeClassification value)
    89	{
    90	    switch (value)
    91	    {
    92	        case ManagerRequestOutcomeClassification::dispatchResult:
    93	            return "dispatch_result";
    94	        case ManagerRequestOutcomeClassification::requestLocalSemanticFailure:
    95	            return "request_local_semantic_failure";
    96	    }
    97	    throw Error(ErrorCode::invalidEnumText,
    98	        "campaign_operations_manager_request_outcome_invalid");
    99	}
   100
   101	std::string ToText(ManagerGlobalStopReason value)
   102	{
   103	    switch (value)
   104	    {
   105	        case ManagerGlobalStopReason::none: return "none";
   106	        case ManagerGlobalStopReason::productionDisabled:
   107	            return "production_disabled";
   108	        case ManagerGlobalStopReason::schedulerProtocolIneffective:
   109	            return "scheduler_protocol_ineffective";
   110	        case ManagerGlobalStopReason::managerBuildNotReady:
   111	            return "manager_build_not_ready";
   112	        case ManagerGlobalStopReason::privilegeFailure:
   113	            return "privilege_failure";
   114	        case ManagerGlobalStopReason::databaseFailure:
   115	            return "database_failure";
   116	    }
   117	    throw Error(ErrorCode::invalidEnumText,
   118	        "campaign_operations_manager_stop_reason_invalid");
   119	}
   120
   121	namespace
   122	{
   123
   124	using ManagerCandidateHook = std::function<void(
   125	    int, int, OperationalRequestId)>;
   126
   127	ManagerRunOnceResult RunManagerOnceInternal(
   128	    const std::string& connectionString, int dispatchLimit,
   129	    const std::string& executablePath,
   130	    const std::optional<ManagerBuildContract>& fixtureBuild,
   131	    const ManagerCandidateHook& hook)
   132	{
   133	    if (dispatchLimit <= 0 ||
   134	        dispatchLimit > kCampaignOperationsManagerMaximumRunOnceLimit)
   135	        throw std::invalid_argument(
   136	            "campaign_operations_manager_run_once_limit_invalid");
   137	    const auto actualBuild = fixtureBuild
   138	        ? fixtureBuild
   139	        : CaptureActualManagerBuildContract(executablePath);
   140	    if (!actualBuild) throw std::runtime_error(
   141	        "campaign_operations_manager_build_contract_unavailable");
   142
   143	    ManagerRunOnceResult result;
   144	    result.dispatchLimit = dispatchLimit;
   145	    {
   146	        pqxx::connection connection{connectionString};
   147	        result.candidates = SelectDispatchCandidatesForManager(
   148	            connection, dispatchLimit);
   149	    }
   150	    if (hook) hook(0, -1,
   151	        result.candidates.empty()
   152	            ? OperationalRequestId(1)
   153	            : result.candidates.front().requestId);
   154
   155	    std::size_t index = 0;
   156	    for (const auto& candidate : result.candidates)
   157	    {
   158	        ManagerRequestResult requestResult{
   159	            candidate.requestId, candidate.expectedRequestVersion,
   160	            {},
   161	            ManagerRequestOutcomeClassification::dispatchResult,
   162	            {}, {}, false, false, {}};
   163	        try
   164	        {
   165	            if (hook) hook(1, static_cast<int>(index), candidate.requestId);
   166	            const auto operation = BuildManagerRequestOperationIdentity(
   167	                candidate.requestIdentityCanonical,
   168	                candidate.expectedRequestVersion);
   169	            requestResult.operationKey = operation.operationKey;
   170	            const ProductionDispatchRequest request{
   171	                candidate.requestId, candidate.expectedRequestVersion,
   172	                operation.operationKey,
   173	                ActorIdentity(kCampaignOperationsManagerActor), *actualBuild,
   174	                true};
   175	            const auto dispatched = fixtureBuild
   176	#if defined(CAMPAIGN_OPERATIONS_H3_TESTING)
   177	                ? DispatchOneRequestForProductionManagerWithFixture(
   178	                    connectionString, request,
   179	                    operation.source.canonicalText(), *fixtureBuild)
   180	#else
   181	                ? throw std::logic_error("H3 fixture adapter unavailable")
   182	#endif
   183	                : DispatchOneRequestForProductionManager(
   184	                    connectionString, request, operation.source.canonicalText(),
   185	                    executablePath);
   186	            if (dispatched.failure ==
   187	                DispatchServiceFailureClassification::commitOutcomeUnknown)
   188	            {
   189	                result.stoppedEarly = true;
   190	                result.stopReason = ManagerGlobalStopReason::databaseFailure;
   191	                result.stopDiagnostic = dispatched.diagnosticCode;
   192	                break;
   193	            }
   194	            requestResult.dispatchClassification =
   195	                ToText(dispatched.classification);
   196	            requestResult.replayDisposition =
   197	                ToText(dispatched.replayDisposition);
   198	            requestResult.newlyCommitted =
   199	                dispatched.replayDisposition == ExactReplayDisposition::newOperation;
   200	            requestResult.exactReplay =
   201	                dispatched.replayDisposition ==
   202	                    ExactReplayDisposition::authoritativeExisting &&
   203	                dispatched.classification ==
   204	                    DispatchResultClassification::existingIdentical;
   205	            if (dispatched.failure ==
   206	                DispatchServiceFailureClassification::transientDatabaseRetryExhausted)
   207	                requestResult.outcome =
   208	                    ManagerRequestOutcomeClassification::requestLocalSemanticFailure;
   209	            requestResult.diagnosticCode = dispatched.diagnosticCode;
   210	        }
   211	        catch (const std::exception& error)
   212	        {
   213	            const auto globalReason = GlobalReason(error);
   214	            if (globalReason != ManagerGlobalStopReason::none)
   215	            {
   216	                result.stoppedEarly = true;
   217	                result.stopReason = globalReason;
   218	                result.stopDiagnostic = error.what();
   219	                break;
   220	            }
   221	            requestResult.outcome =
   222	                ManagerRequestOutcomeClassification::requestLocalSemanticFailure;
   223	            requestResult.diagnosticCode = error.what();
   224	        }
   225	        if (hook) hook(2, static_cast<int>(index), candidate.requestId);
   226	        result.requests.push_back(std::move(requestResult));
   227	        ++index;
   228	    }
   229	    return result;
   230	}
   231
   232	} // namespace
   233
   234	ManagerRunOnceResult RunCampaignOperationsManagerOnce(
   235	    const std::string& connectionString, int dispatchLimit,
   236	    const std::string& executablePath)
   237	{
   238	    return RunManagerOnceInternal(connectionString, dispatchLimit,
   239	        executablePath, std::nullopt, {});
   240	}
   241
   242	#if defined(CAMPAIGN_OPERATIONS_H3_TESTING)
   243	ManagerRunOnceResult RunCampaignOperationsManagerOnceForTest(
   244	    const std::string& connectionString, int dispatchLimit,
   245	    const ManagerBuildContract& fixtureBuild, const ManagerTestHook& testHook)
   246	{
   247	    const ManagerCandidateHook hook = [&](int point, int index,
   248	        OperationalRequestId requestId)
   249	    {
   250	        if (!testHook) return;
   251	        const auto injection = point == 0
   252	            ? ManagerTestInjectionPoint::afterCandidateSnapshot
   253	            : point == 1
   254	                ? ManagerTestInjectionPoint::beforeCandidateProcessing
   255	                : ManagerTestInjectionPoint::afterCandidateProcessing;
   256	        testHook(injection, index, requestId);
   257	    };
   258	    return RunManagerOnceInternal(connectionString, dispatchLimit, {},
   259	        fixtureBuild, hook);
   260	}
   261	#endif
   262
   263	int RunCampaignOperationsManagerOnceCommand(
   264	    const std::string& connectionString, int dispatchLimit,
   265	    const std::string& executablePath, std::ostream& output,
   266	    std::ostream& errors)
   267	{
   268	    const auto result = RunCampaignOperationsManagerOnce(
   269	        connectionString, dispatchLimit, executablePath);
   270	    output << "CAMPAIGN_OPERATIONS_MANAGER_RUN_ONCE"
   271	           << ",dispatch_limit=" << result.dispatchLimit
   272	           << ",candidates_selected=" << result.candidates.size()
   273	           << ",processed=" << result.requests.size()
   274	           << ",stopped_early=" << (result.stoppedEarly ? "true" : "false")
   275	           << ",stop_reason=" << ToText(result.stopReason)
   276	           << ",candidate_request_ids=";
   277	    for (std::size_t index = 0; index < result.candidates.size(); ++index)
   278	    {
   279	        if (index != 0U) output << ':';
   280	        output << result.candidates[index].requestId.value();
   281	    }
   282	    if (result.candidates.empty()) output << "none";
   283	    output << '\n';
   284	    for (const auto& request : result.requests)
   285	        RenderRequest(output, request);
   286	    if (result.stoppedEarly)
   287	    {
   288	        errors << "CAMPAIGN_OPERATIONS_MANAGER_STOPPED"
   289	               << ",reason=" << ToText(result.stopReason)
   290	               << ",diagnostic=" << result.stopDiagnostic << '\n';
   291	        return 2;
   292	    }
   293	    return 0;
   294	}
   295
   296	} // namespace EA::CampaignOperations

================================================================================
CALLER / EXIT-CODE REFERENCES
================================================================================
Sources/ExperimentRecommendationCampaignMaterializationService.cpp:317:        return 2;
Sources/ExperimentRecommendationCampaignMaterializationService.cpp:360:        return 2;
Sources/ExperimentRecommendationCampaignApprovalService.cpp:183:        return 2;
Sources/ExperimentRecommendationCampaignApprovalService.cpp:227:        return 2;
Sources/ExperimentRecommendationCampaignApprovalService.cpp:269:        return 2;
Sources/ExperimentRecommendationRanking.cpp:135:        case RecommendationEvaluationDisposition::completedDuplicate: return 2;
Sources/ExperimentRecommendationRanking.cpp:139:            return 2;
Sources/ExperimentRecommendationRanking.cpp:151:        case RecommendationRankingBucket::nonActionable: return 2;
Sources/ExperimentRecommendationConversionWorkflowService.cpp:189:        return 2;
Sources/ExperimentRecommendationConversionWorkflowService.cpp:237:        return 2;
Sources/ExperimentRecommendationEvaluationService.cpp:353:        return 2;
Sources/ExperimentRecommendationRankingService.cpp:346:        return 2;
Sources/CampaignOperationsDispatchService.cpp:2:#include "CampaignOperationsManager.hpp"
Sources/ExperimentRecommendationCampaignPlanningService.cpp:205:        return 2;
Sources/ExperimentRecommendationCandidateGenerator.cpp:31:        case RecommendationMutationParameter::labelThreshold: return 2;
Sources/ExperimentRecommendationCampaignFollowUpProposalPreview.cpp:163:        return 2;
Sources/ExperimentRecommendationService.cpp:80:        case RecommendationMutationParameter::labelThreshold: return 2;
Sources/CampaignOperationsManager.cpp:1:#include "CampaignOperationsManager.hpp"
Sources/CampaignOperationsManager.cpp:30:    source << kCampaignOperationsManagerOperationPrefix
Sources/ExperimentRecommendationEvaluation.cpp:154:    if (status == "completed") return 2;
Sources/ExperimentRecommendationCampaignReviewService.cpp:224:        return 2;
Sources/CampaignOperationsService.cpp:56:        return 2;
Sources/CampaignOperationsCompletionService.cpp:46:        return 2;
Sources/CampaignOperationsControlService.cpp:67:    if (domain->code() == ErrorCode::persistenceConflict) return 2;
Sources/CampaignOperationsManagerService.cpp:1:#include "CampaignOperationsManagerService.hpp"
Sources/CampaignOperationsManagerService.cpp:3:#include "CampaignOperationsManager.hpp"
Sources/CampaignOperationsManagerService.cpp:134:        dispatchLimit > kCampaignOperationsManagerMaximumRunOnceLimit)
Sources/CampaignOperationsManagerService.cpp:136:            "campaign_operations_manager_run_once_limit_invalid");
Sources/CampaignOperationsManagerService.cpp:173:                ActorIdentity(kCampaignOperationsManagerActor), *actualBuild,
Sources/CampaignOperationsManagerService.cpp:234:ManagerRunOnceResult RunCampaignOperationsManagerOnce(
Sources/CampaignOperationsManagerService.cpp:243:ManagerRunOnceResult RunCampaignOperationsManagerOnceForTest(
Sources/CampaignOperationsManagerService.cpp:263:int RunCampaignOperationsManagerOnceCommand(
Sources/CampaignOperationsManagerService.cpp:268:    const auto result = RunCampaignOperationsManagerOnce(
Sources/CampaignOperationsManagerService.cpp:270:    output << "CAMPAIGN_OPERATIONS_MANAGER_RUN_ONCE"
Sources/CampaignOperationsManagerService.cpp:288:        errors << "CAMPAIGN_OPERATIONS_MANAGER_STOPPED"
Sources/CampaignOperationsManagerService.cpp:291:        return 2;
Sources/CampaignOperationsDispatchRepository.cpp:3:#include "CampaignOperationsManager.hpp"
Sources/CampaignOperationsDispatchRepository.cpp:222:    if (limit <= 0 || limit > kCampaignOperationsManagerMaximumRunOnceLimit)
Sources/CampaignOperationsDispatchRepository.cpp:223:        throw std::invalid_argument("campaign_operations_manager_run_once_limit");
Sources/CampaignOperationsDispatchRepository.cpp:285:        row[6].as<int>() != kCampaignOperationsManagerOperationContractVersion)
Sources/CampaignOperationsDispatchRepository.cpp:314:            sourceHash, kCampaignOperationsManagerOperationContractVersion});
Sources/ExperimentRecommendationCampaignFollowUpProposalReviewPresentation.cpp:136:        return 2;
Sources/ExperimentRecommendationCampaignFollowUpProposalReviewPresentation.cpp:175:        return 2;
Sources/ExperimentScheduler.cpp:63:#include "CampaignOperationsManagerService.hpp"
Sources/ExperimentScheduler.cpp:280:    std::optional<int> campaignOperationsManagerRunOnceLimit;
Sources/ExperimentScheduler.cpp:1047:            arg == "--campaign-operations-manager-run-once" ||
Sources/ExperimentScheduler.cpp:2374:        else if (arg == "--campaign-operations-manager-run-once")
Sources/ExperimentScheduler.cpp:2376:            if (options.campaignOperationsManagerRunOnceLimit)
Sources/ExperimentScheduler.cpp:2378:                    "duplicate --campaign-operations-manager-run-once");
Sources/ExperimentScheduler.cpp:2379:            options.campaignOperationsManagerRunOnceLimit = ParsePositiveInt(
Sources/ExperimentScheduler.cpp:3559:        (options.campaignOperationsManagerRunOnceLimit.has_value() ? 1 : 0) +
Sources/ExperimentScheduler.cpp:4084:    const bool campaignOperationsManagerMutation =
Sources/ExperimentScheduler.cpp:4085:        options.campaignOperationsManagerRunOnceLimit.has_value();
Sources/ExperimentScheduler.cpp:4092:        campaignOperationsManagerMutation;
Sources/ExperimentScheduler.cpp:4134:            !options.campaignOperationsManagerRunOnceLimit &&
Sources/ExperimentScheduler.cpp:4143:    if (campaignOperationsManagerMutation &&
Sources/ExperimentScheduler.cpp:4147:    if (campaignOperationsManagerMutation &&
Sources/ExperimentScheduler.cpp:5094:        return 2;
Sources/ExperimentScheduler.cpp:6052:        return 2;
Sources/ExperimentScheduler.cpp:6801:        return 2;
Sources/ExperimentScheduler.cpp:6867:        return 2;
Sources/ExperimentScheduler.cpp:7183:        return 2;
Sources/ExperimentScheduler.cpp:7192:        return 2;
Sources/ExperimentScheduler.cpp:7201:        return 2;
Sources/ExperimentScheduler.cpp:12510:        return 2;
Sources/ExperimentScheduler.cpp:13113:        return 2;
Sources/ExperimentScheduler.cpp:19760:        return 2;
Sources/ExperimentScheduler.cpp:22760:        << " --campaign-operations-manager-run-once LIMIT --yes\n"
Sources/ExperimentScheduler.cpp:22796:    if (options.campaignOperationsManagerRunOnceLimit)
Sources/ExperimentScheduler.cpp:22797:        return EA::CampaignOperations::RunCampaignOperationsManagerOnceCommand(
Sources/ExperimentScheduler.cpp:22798:            connectionString, *options.campaignOperationsManagerRunOnceLimit,
Sources/ExperimentScheduler.cpp:23736:            options.campaignOperationsManagerRunOnceLimit ||
Sources/ExperimentScheduler.cpp:23868:        return 2;
Sources/ExperimentRecommendationConversionExecutionService.cpp:79:            return 2;
Sources/ExperimentRecommendationConversionActivationService.cpp:76:            return 2;
Sources/ExperimentRecommendationCampaignHandoffService.cpp:164:        return 2;
Sources/ExperimentRecommendationCampaignHandoffService.cpp:206:        return 2;

================================================================================
LIKELY CALLER CONTEXT
================================================================================
  2360	        else if (arg == "--campaign-operations-production-disable")
  2361	        {
  2362	            if (options.campaignOperationsProductionDisable)
  2363	                throw std::invalid_argument(
  2364	                    "duplicate --campaign-operations-production-disable");
  2365	            options.campaignOperationsProductionDisable = true;
  2366	        }
  2367	        else if (arg == "--campaign-operations-dispatch-request")
  2368	        {
  2369	            if (options.campaignOperationsProductionDispatchRequest)
  2370	                throw std::invalid_argument(
  2371	                    "duplicate --campaign-operations-dispatch-request");
  2372	            options.campaignOperationsProductionDispatchRequest = true;
  2373	        }
  2374	        else if (arg == "--campaign-operations-manager-run-once")
  2375	        {
  2376	            if (options.campaignOperationsManagerRunOnceLimit)
  2377	                throw std::invalid_argument(
  2378	                    "duplicate --campaign-operations-manager-run-once");
  2379	            options.campaignOperationsManagerRunOnceLimit = ParsePositiveInt(
  2380	                arg, RequireNextArg(argc, argv, i, arg));
  2381	        }
  2382	        else if (arg == "--campaign-operations-reconcile-observe" ||
  2383	                 arg == "--campaign-operations-reconcile-recover")
  2384	        {
  2385	            if (options.campaignOperationsReconcileRunKey)
  2386	                throw std::invalid_argument(
  2387	                    "duplicate Campaign Operations reconciliation command");
  2388	            options.campaignOperationsReconcileRunKey =
  2389	                RequireNextArg(argc, argv, i, arg);
  2390	            options.campaignOperationsReconcileRecover =
  2391	                arg == "--campaign-operations-reconcile-recover";
  2392	        }
  2393	        else if (
  2394	            arg == "--campaign-operations-expected-control-version")
  2395	        {
  2396	            if (options.campaignOperationsExpectedControlVersion)
  2397	                throw std::invalid_argument(
  2398	                    "duplicate "
  2399	                    "--campaign-operations-expected-control-version");
  2400	            const long long value = ParseSignedLongLong(
  2401	                arg, RequireNextArg(argc, argv, i, arg));
  2402	            if (value < 0 || value > std::numeric_limits<int>::max())
  2403	                throw std::invalid_argument(
  2404	                    "invalid "
  2405	                    "--campaign-operations-expected-control-version");
  2406	            options.campaignOperationsExpectedControlVersion =
  2407	                static_cast<int>(value);
  2408	        }
  2409	        else if (arg == "--campaign-operations-request-id")
  2410	        {
  2411	            if (options.campaignOperationsControlRequestId)
  2412	                throw std::invalid_argument(
  2413	                    "duplicate --campaign-operations-request-id");
  2414	            options.campaignOperationsControlRequestId =
  2415	                ParsePositiveLongLong(
  2416	                    arg, RequireNextArg(argc, argv, i, arg));
  2417	        }
  2418	        else if (
  2419	            arg == "--campaign-operations-expected-request-version")
  2420	        {
  2421	            if (options.campaignOperationsExpectedRequestVersion)
  2422	                throw std::invalid_argument(
  2423	                    "duplicate "
  2424	                    "--campaign-operations-expected-request-version");
  2425	            const long long value = ParsePositiveLongLong(
 22720	        << "Campaign Operations locks. Reconciliation observes first; only "
 22721	        << "the recover form invokes a named safe owning transition. Neither "
 22722	        << "form signals workers or controls scheduler processes.\n"
 22723	        << "Usage: " << exe
 22724	        << " --campaign-operations-complete-if-settled CAMPAIGN_ID "
 22725	        << "--campaign-operations-operation-key KEY "
 22726	        << "--campaign-operations-actor ACTOR "
 22727	        << "--campaign-operations-reason REASON --yes\n"
 22728	        << "Usage: " << exe
 22729	        << " --campaign-operations-completion-status CAMPAIGN_ID\n"
 22730	        << "Campaign Operations Phase 5 records one immutable operational "
 22731	        << "completion only when every accepted obligation is proven settled "
 22732	        << "in one serializable evidence transaction. Completion never changes "
 22733	        << "experiment lifecycle and never means scientific success. There "
 22734	        << "is no force-complete, reopen, override, or delete command.\n"
 22735	        << "Usage: " << exe
 22736	        << " --campaign-operations-production-readiness | "
 22737	        << "--campaign-operations-production-status\n"
 22738	        << "Campaign Operations Phase H1 commands are read-only. They report "
 22739	        << "migration 055, exact generation-52 evidence, immutable admission "
 22740	        << "and Attempt V2 evidence, role readiness, blocked leases, and "
 22741	        << "Completion V1 nested-V2 proof.\n"
 22742	        << "Usage: " << exe
 22743	        << " --campaign-operations-production-enable "
 22744	        << "--campaign-operations-operation-key KEY "
 22745	        << "--campaign-operations-expected-production-version N "
 22746	        << "--campaign-operations-independent-verification-reference REF "
 22747	        << "--campaign-operations-actor ACTOR --campaign-operations-reason REASON --yes\n"
 22748	        << "Usage: " << exe
 22749	        << " --campaign-operations-production-disable "
 22750	        << "--campaign-operations-operation-key KEY "
 22751	        << "--campaign-operations-expected-production-version N "
 22752	        << "--campaign-operations-actor ACTOR --campaign-operations-reason REASON --yes\n"
 22753	        << "Usage: " << exe
 22754	        << " --campaign-operations-dispatch-request "
 22755	        << "--campaign-operations-request-id ID "
 22756	        << "--campaign-operations-expected-request-version N "
 22757	        << "--campaign-operations-operation-key KEY "
 22758	        << "--campaign-operations-actor ACTOR --yes\n"
 22759	        << "Usage: " << exe
 22760	        << " --campaign-operations-manager-run-once LIMIT --yes\n"
 22761	        << "Phase H3 run-once takes one optimistic read-only candidate snapshot,"
 22762	        << " processes at most LIMIT requests sequentially, and has no daemon,"
 22763	        << " polling, sleep, or continuous mode.\n"
 22764	        << "Phase H2 mutations are default-off, caller-keyed, and single-request "
 22765	        << "only. They require the dedicated deployed roles and a clean Release "
 22766	        << "build; H3 run-once is bounded and H4 continuous mode is excluded.\n"
 22767	        << "Usage: " << exe
 22768	        << " --stop-after-checkpoint=ID:EPOCH | --clear-stop-after-checkpoint=ID | "
 22769	        << "--stop-after-checkpoint-all=EPOCH | --clear-stop-after-checkpoint-all | "
 22770	        << "--enable-checkpoint-infer=ID | --disable-checkpoint-infer=ID | "
 22771	        << "--checkpoint-infer-min-epoch=ID:EPOCH | --checkpoint-infer-interval=ID:EPOCH_INTERVAL | "
 22772	        << "--enable-checkpoint-policy=ID | --disable-checkpoint-policy=ID | "
 22773	        << "--set-checkpoint-policy=ID:key=value,key=value\n"
 22774	        << "Usage: " << exe
 22775	        << " --stop-experiment=ID | --stop-all-experiments [--dry-run] [--yes] [--force]\n"
 22776	        << "Usage: " << exe
 22777	        << " --analyze-experiment=EXPERIMENT_ID | --analyze-completed-experiments | "
 22778	        << "--print-experiment-leaderboard [--leaderboard-symbol=SYMBOL] "
 22779	        << "[--leaderboard-horizon=N] [--leaderboard-limit=N]\n"
 22780	        << "Backup note: --backup-database writes a PostgreSQL custom-format dump with schema and data; "
 22781	        << "migrations remain schema history, and --backup-output=Database/backups/LSTM_latest.dump overwrites a stable file.\n"
 22782	        << "Queue exit codes: 0=created, 1=invalid_arguments, 2=database_error, 3=duplicates_only\n";
 22783	}
 22784
 22785	int RunCampaignOperationsCommand(const SchedulerOptions& options)
 22786	{
 22787	    const std::string connectionString = LstmDbConnectionString();
 22788	    if (options.campaignOperationsProductionReadiness)
 22789	        return EA::CampaignOperations::RunProductionReadinessCommand(
 22790	            connectionString, std::cout, std::cerr,
 22791	            EA::CampaignOperations::CaptureActualManagerBuildContract(
 22792	                options.selfPath));
 22793	    if (options.campaignOperationsProductionStatus)
 22794	        return EA::CampaignOperations::RunProductionStatusCommand(
 22795	            connectionString, std::cout, std::cerr);
 22796	    if (options.campaignOperationsManagerRunOnceLimit)
 22797	        return EA::CampaignOperations::RunCampaignOperationsManagerOnceCommand(
 22798	            connectionString, *options.campaignOperationsManagerRunOnceLimit,
 22799	            options.selfPath, std::cout, std::cerr);
 22800	    if (options.campaignOperationsProductionEnable)
 22801	    {
 22802	        const auto build =
 22803	            EA::CampaignOperations::CaptureActualManagerBuildContract(
 22804	                options.selfPath);
 22805	        if (!build)
 22806	            throw std::runtime_error(
 22807	                "production enable requires a clean Release build identity");
 22808	        EA::CampaignOperations::ProductionEnableRequest request{
 22809	            *options.campaignOperationsOperationKey,
 22810	            *options.campaignOperationsExpectedProductionVersion,
