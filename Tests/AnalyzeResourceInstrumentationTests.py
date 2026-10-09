#!/usr/bin/env python3
"""Offline measurement math and fail-closed coverage regressions.

Small in-memory counter examples are unit-test data, never workload evidence.
"""
import unittest
import json
from pathlib import Path
import sys
from AnalyzeResourceInstrumentation import cpu_percent, summarize, verify_destination, child_pids

EVIDENCE = None


class MeasurementTests(unittest.TestCase):
    def setUp(self):
        self.attempt = ["1","42","100:500","42","/private/trial/bin/lstm-analyze-worker","command","completed"]
        p = {"pid":42,"start_identity":"100:500","start_epoch":100.,"state":2,
             "usage_available":True,"executable":self.attempt[4],"rss_bytes":4096,
             "physical_footprint_bytes":8192,"user_ns":1000000,"system_ns":500000,
             "disk_read_bytes":0,"disk_write_bytes":1024}
        host = {"pressure":1,"swapins_pages":10,"swapouts_pages":5,"pageouts_pages":2,
                "swap_used_bytes":2048,"page_size_bytes":16384}
        pg = dict(p,pid=44,start_identity="90:100",executable="postgres")
        self.samples=[]
        for i in range(12):
            s = {"monotonic":i*.005,"epoch":99.99+i*.005,"utc":str(i),"host":host.copy(),
                 "postgres":[pg.copy()],"postgres_connections":{"total":3,"worker":1,"scheduler":1,"observer":1,"backend_pids":[44]}}
            if 2 <= i <= 7:
                s["worker"] = p.copy()
            elif i > 7:
                s["worker_gone"] = True
            self.samples.append(s)
        self.lifecycle = [{"event":"fork","pid":42,"fork_before_epoch":99.999,"utc_epoch":100.,"identity":p.copy()},
            {"event":"reaped","pid":42,"utc_epoch":100.04,"status":0,"identity_before_wait":p.copy(),
             "peak_rss_bytes":10000,"user_seconds":.01,"system_seconds":.005,"input_blocks":0,"output_blocks":1}]

    def summary(self):
        return summarize(self.samples,self.lifecycle,self.attempt)

    def test_kernel_peak_and_duration_bounds(self):
        r=self.summary()
        self.assertEqual(r["qualification"],"PASS",r["errors"])
        self.assertEqual(r["worker"]["peak_rss_bytes"],10000)
        self.assertEqual(r["worker"]["sampled_max_rss_bytes"],4096)
        self.assertAlmostEqual(r["worker"]["total_cpu_seconds"],.015)
        self.assertAlmostEqual(r["worker"]["execution_duration_uncertainty_seconds"],.005)
        self.assertEqual(r["postgres_connections"]["max_total_excluding_observer"],2)

    def test_cpu_math_allows_multiple_cores(self):
        self.assertAlmostEqual(cpu_percent({"user_ns":0,"system_ns":0},{"user_ns":100000000,"system_ns":100000000},.1),200)
        with self.assertRaises(ValueError):
            cpu_percent({"user_ns":10,"system_ns":0},{"user_ns":0,"system_ns":0},.1)

    def test_pid_reuse_rejected(self):
        self.samples[4]["worker"]["start_identity"]="101:0"
        self.assertEqual(self.summary()["qualification"],"FAIL")

    def test_missing_short_worker_cannot_pass_from_exit_alone(self):
        for s in self.samples: s.pop("worker",None)
        r=self.summary()
        self.assertEqual(r["qualification"],"FAIL")
        self.assertTrue(r["missing_short_lived_worker"])

    def test_missing_kernel_exit_accounting_rejected(self):
        self.lifecycle.pop()
        self.assertEqual(self.summary()["qualification"],"FAIL")

    def test_coarse_sampling_rejected(self):
        self.samples[-1]["monotonic"] += .1
        r=self.summary()
        self.assertEqual(r["qualification"],"FAIL")
        self.assertGreater(r["missed_interval_count"],0)

    def test_peak_units_inconsistency_rejected(self):
        self.lifecycle[1]["peak_rss_bytes"]=4
        self.assertEqual(self.summary()["qualification"],"FAIL")

    def test_postgres_observation_required_during_worker(self):
        for s in self.samples: s.pop("postgres_connections")
        self.assertEqual(self.summary()["qualification"],"FAIL")

    def test_pressure_or_swapout_rejected(self):
        self.samples[4]["host"]["pressure"]=2
        self.assertEqual(self.summary()["qualification"],"FAIL")
        self.samples[4]["host"]["pressure"]=1
        self.samples[-1]["host"]["swapouts_pages"]+=1
        self.assertEqual(self.summary()["qualification"],"FAIL")

    def test_window_must_cover_birth_and_reap(self):
        self.lifecycle[-1]["utc_epoch"]=101.
        self.assertEqual(self.summary()["qualification"],"FAIL")

    def test_postgres_and_worker_memory_kept_separate(self):
        self.samples[-1]["postgres"][0]["rss_bytes"]=50000
        r=self.summary()
        self.assertEqual(r["private_postgres"]["sampled_peak_sum_rss_bytes"],50000)
        self.assertEqual(r["worker"]["peak_rss_bytes"],10000)

    def test_missed_short_backend_recorded(self):
        self.samples[4]["postgres_connections"]["backend_pids"].append(88)
        self.assertEqual(self.summary()["private_postgres"]["unsampled_observed_backend_pids"],[88])

    def test_private_destination_exact_fields(self):
        actual={"database":"private","host":"127.0.0.1","port":55489,"data_directory":"/private/trial/pgdata"}
        verify_destination(actual,"private","55489","/private/trial/pgdata")
        actual["port"]=5432
        with self.assertRaises(RuntimeError):
            verify_destination(actual,"private","55489","/private/trial/pgdata")

    def test_private_destination_wrong_directory_rejected(self):
        actual={"database":"private","host":"127.0.0.1","port":55489,"data_directory":"/Volumes/Forex Data/forexdb"}
        with self.assertRaises(RuntimeError):
            verify_destination(actual,"private","55489","/private/trial/pgdata")

    def test_conservative_duration_uses_actual_read_times(self):
        self.samples[7]["worker"]["observed_epoch_before"] = self.samples[7]["epoch"]+.002
        self.samples[8]["worker_observed_epoch_after"] = self.samples[8]["epoch"]+.003
        r=self.summary()
        self.assertAlmostEqual(r["worker"]["execution_duration_uncertainty_seconds"],.006)

    def test_zombie_identity_can_use_owning_parent_fork_receipt(self):
        self.lifecycle[-1]["identity_before_wait"]=None
        self.assertEqual(self.summary()["qualification"],"PASS")

    def test_child_census_uses_pid_count_not_bytes(self):
        self.assertEqual(child_pids([10,11,12,13,0,0,0,0],4),[10,11,12,13])
        with self.assertRaises(RuntimeError): child_pids([10,11],2)

    def test_cumulative_sampling_drift_recorded(self):
        for i,s in enumerate(self.samples): s["monotonic"] = i*.006
        r=self.summary()
        self.assertGreater(r["missed_interval_count"],0)
        self.assertAlmostEqual(r["median_observed_interval_seconds"],.006)

    def test_short_postgres_process_read_misses_recorded(self):
        self.samples[4]["postgres_missing_sample_pids"]=[88]
        r=self.summary()
        self.assertEqual(r["private_postgres"]["missed_process_read_count"],1)
        self.assertEqual(r["private_postgres"]["missed_process_read_pids"],[88])

    def test_retained_real_measurements_recompute_and_units_match(self):
        if EVIDENCE is None:
            self.skipTest("pass a retained instrumented evidence directory for the native-counter regression")
        samples=[json.loads(line) for line in (EVIDENCE/'resource-samples.jsonl').read_text().splitlines()]
        lifecycle=[json.loads(line) for line in (EVIDENCE/'resource-lifecycle.jsonl').read_text().splitlines()]
        attempt=json.loads((EVIDENCE/'single-worker-results.json').read_text())["new_attempt"]
        recorded=json.loads((EVIDENCE/'resource-summary.json').read_text())
        self.assertEqual(summarize(samples,lifecycle,attempt),recorded)
        self.assertEqual(recorded['qualification'],'PASS')
        for sample in samples:
            processes=sample['postgres']+([sample['worker']] if sample.get('worker') else [])
            for p in processes:
                for component in ('user','system'):
                    converted=p[component+'_abstime']*p['timebase_numer']//p['timebase_denom']
                    self.assertLessEqual(abs(p[component+'_ns']-converted),1)
            accounted={p['pid'] for p in sample['postgres']} | set(sample.get('postgres_missing_sample_pids',[]))
            self.assertTrue(set(sample.get('postgres_child_pids',[])) <= accounted)
        cpu=max((s['worker']['user_ns']+s['worker']['system_ns'])/1e9 for s in samples
                if s.get('worker') and s['worker']['usage_available'])
        self.assertLessEqual(cpu,recorded['worker']['total_cpu_seconds']+.001)


if __name__ == "__main__":
    if len(sys.argv)==2:
        EVIDENCE=Path(sys.argv.pop()).resolve()
    unittest.main()
