Campaign Operations Phase H4 — external continuous-operation supervisor
=======================================================================

H4 is deployment-owned composition around the accepted bounded H3 command. It
does not add an ``LSTM_Release`` mode, database state, scheduler integration,
or a new Campaign Operations authority. See ADR-0020.

Install
-------

Install ``Scripts/CampaignOperationsH4Supervisor.py`` and the reviewed Release
binary at the configured absolute paths. Copy the example JSON and connection
environment file to the paths in the launchd job, set both configuration and
connection file owner-only, replace every placeholder, and validate the plist:

::

   plutil -lint Deployment/CampaignOperationsH4/com.expertadvisor.campaign-operations-h4.plist
   /usr/bin/python3 Scripts/CampaignOperationsH4Supervisor.py --config /absolute/path/campaign-operations-h4.json

The JSON configuration file itself and the connection environment file must
both be absolute regular files with owner-only permissions. A group/world
accessible configuration is rejected as ``STOP_INVALID_CONFIGURATION`` before
schedule activation. This is required because the JSON controls the reviewed
executable path/hash, deployment identity, cadence, retry policy, and
state/log destinations.

The deployment identity, deployment execution identity, target database identity, target environment,
reviewed PostgreSQL login identity,
executable SHA-256, positive H3 limit (1..100), positive normal interval,
bounded positive backoff list, finite positive retry budget, finite positive
drain timeout, state directory, and log directory are mandatory. Missing or
unsafe configuration stops before H3 launch. The connection file is owner-only
and supplies only the existing connection environment; it is never printed.
It is completely parsed and validated once during supervisor startup, then the
validated child environment is retained for later H1/H3 commands. A malformed
or unapproved setting, including ``PGSERVICE`` or ``PGUSER``, produces the
deployment-owned ``STOP_INVALID_CONFIGURATION`` health/state record and no H3
launch.
The current reviewed ``LSTM_Release`` connection builder explicitly selects the
``pqxx`` PostgreSQL login and receives only ``LSTM_DB_HOST`` and
``LSTM_DB_NAME`` from this deployment file. The config must name that exact
login; it cannot silently inherit an ambient PostgreSQL user or service.
``PGSERVICE`` and ``PGUSER`` are deliberately rejected here because they do
not override that existing explicit connection mechanism. ``UserName`` in the
plist is the separate reviewed macOS execution identity and is compared to
``deployment_execution_identity`` by the supervisor at launch.
``log_retention_policy`` is a mandatory reviewed external rotation/retention
reference; H4 itself does not delete or rotate evidence.

The actual H3 child command is exactly::

   LSTM_Release --campaign-operations-manager-run-once LIMIT --yes

Immediately before every child, the supervisor runs the existing read-only
``--campaign-operations-production-readiness`` command. H3 retains final
transactional gates.

launchd operation
-----------------

Use the reviewed single launchd label once per target database/environment and
deployment identity. ``launchd`` label ownership prevents ordinary duplicate
launches; there is no database singleton or cross-host election. Install/start
and stop/remove using the deployment owner's approved ``launchctl bootstrap`` /
``bootout`` procedure. Do not start a second copy manually. The health file is
``campaign_operations_h4_health.json`` in the state directory; durable state,
child stdout/stderr, classifier decisions, and append-only transitions are
deployment operational evidence only. Atomic state publication synchronizes
the containing directory; child capture and transition records are fsynced.

The plist uses ``RunAtLoad`` and ``KeepAlive/Crashed=true`` with a 60-second
``ThrottleInterval``. Thus an unexpected supervisor crash is recovered without
restarting an intentional zero-exit STOP state or turning a persisted MUST-stop
decision into a launchd restart loop. Retry/backoff remains supervisor-owned.

The action values are ``CONTINUE_AFTER_NORMAL_INTERVAL``,
``BACKOFF_AND_RETRY_WITH_READINESS``, ``STOP_DISABLED``,
``STOP_DEGRADED_OPERATOR_REQUIRED``, ``STOP_INVALID_CONFIGURATION``, and
``STOP_MALFORMED_RESULT``. Retry is finite and restarts preflight. A STOP state
requires operator review and an intentional service restart after correction.
On restart, only a complete durable prior action is restored; incomplete state
fails closed. A complete restored normal or retry action retains its original
bounded deadline across repeated supervisor restarts: H4 waits only its
remaining duration (or no duration once elapsed), then performs immediate H1
readiness before H3. A restored retry retains its retry count and is valid only
while another retry remains.

Resolving a persisted H4 STOP state
-----------------------------------

After investigating and correcting the underlying cause, an operator may
deliberately resolve *only* the deployment-owned H4 STOP record while the
launchd job is booted out/stopped:

::

   /usr/bin/python3 Scripts/CampaignOperationsH4Supervisor.py \
     --config /absolute/path/campaign-operations-h4.json --resolve-stop-state

The command validates the complete H4 STOP record and replaces it with one
durable ``operator_stop_resolution`` transition. It does not read, infer,
repair, or mutate H1--H3 database/workflow truth, and it does not launch H3.
It refuses missing, malformed, non-STOP, or identity-mismatched state. Start
the reviewed launchd job only after this explicit action. The normal supervisor
start validates the configuration again and performs immediate H1 readiness
before any H3 child can begin; failed readiness remains fail-closed.

Ordinary duplicate launch prevention remains owned by the reviewed launchd
label/job identity; H4 does not introduce a database singleton, lease,
heartbeat, PID ownership table, advisory lock, or process-lifetime fence.

If the deployment owner independently observes duplicate-instance drift, it
records that observation by creating:

::

   campaign_operations_h4_duplicate_drift.marker

in the configured H4 state directory. The marker is an operational observation
input only, never coordination authority. Before each readiness/H3 cycle the
supervisor checks for this marker. If present, it launches neither readiness nor
H3, records ``duplicate_drift_detected=true``, classifies
``duplicate_drift_observed``, and selects
``STOP_DEGRADED_OPERATOR_REQUIRED``. The resulting MUST-stop remains durable
until the operator investigates the duplicate deployment, removes/suppresses
the duplicate through deployment controls, removes the marker, and deliberately
resolves the H4 STOP state.

When no independent observation has been made,
``duplicate_drift_detected`` remains ``null`` rather than manufacturing a
verified ``false``. Likewise, a stopped state records
``service_alive=false``. Scheduled actions record their actual computed future
invocation time; a restored pending action has no invented schedule timestamp.

Emergency production disable is the existing immutable global disable event
first, then prevent future H4 launches and drain. H4 rollback removes/boots out
the external job only; it does not alter H1--H3. Planned maintenance prevents
future launches, drains the active bounded child, then resumes only after a
fresh configuration/readiness check. No scheduler, training, inference, or
analysis process is controlled by H4. The final H3 launch decision and child
creation run in a brief SIGTERM/SIGINT-masked critical section; an already
pending stop suppresses creation, while a stop after child creation follows the
existing bounded graceful-drain/forced-termination path.
