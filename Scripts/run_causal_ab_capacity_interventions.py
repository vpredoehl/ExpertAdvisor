#!/usr/bin/env python3
"""Research-only selective AB intervention batch. --test-only never accesses a database.
One canonical read-only snapshot, two deterministic native replays, no primary
publication, no outcomes, no per-candidate prefix replay. Inputs stay immutable.
"""
import argparse
import collections
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import ast
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
BASELINE_COMMIT = '900f4dde739a6782c524e4f57a5f9d5a43796c34'
REFERENCE = Path('/private/tmp/ea_phase2c13e_interventions_20261013_222953_91999')
REFERENCE_MANIFEST_SHA256 = '0ad313bfe3a796e8673e1a3cb52941b529a0d7f0203cb421af2af594b58fad99'
RESEARCH_FILES = [
    'Headers/CausalABCapacityCounterfactual.hpp', 'Sources/CausalABCapacityCounterfactual.cpp',
    'Sources/CausalABCapacityInterventionCLI.cpp', 'Tests/CausalABCapacityInterventionTests.cpp',
    'Scripts/run_causal_ab_capacity_interventions.py',
    'Scripts/run_causal_fibonacci_trendline_study.sh',
    'Tests/CausalFibonacciTrendlineHistoricalEvaluationTests.sh',
]
SOURCE_FILES = [
    'Headers/CausalABCapacityCounterfactual.hpp', 'Sources/CausalABCapacityCounterfactual.cpp',
    'Sources/CausalABCapacityInterventionCLI.cpp', 'Sources/CausalFractalHistoryShadowAudit.cpp',
    'Headers/CausalFractalHistoryShadowAudit.hpp', 'Headers/CausalFibonacciTrendlineHistoricalEvaluation.hpp',
    'Headers/CausalFibonacciTrendlineConfluenceDiagnostic.hpp', 'Headers/CausalFibonacciConfluenceIntegration.hpp',
    'Headers/CausalFractalTrendLineGeometry.hpp', 'Sources/TG4HistoricalEmpiricalEvaluation.cpp',
    'Sources/TG4HistoricalMarketDataRepository.cpp', 'Sources/RuntimeDatabaseConnection.cpp',
    'Common/HistoricalFxTimestamp.cpp', 'Sources/CausalFibonacciTrendlineHistoricalArtifact.cpp',
    'Headers/CausalFibonacciResearchCandleSnapshot.hpp', 'Scripts/tg4_analysis_config.frozen_v1.conf',
    'Tests/CausalABCapacityInterventionTests.cpp', 'Scripts/run_causal_ab_capacity_interventions.py',
]
CXX = '/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang++'
SDK = '/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk'
GIT = '/Applications/Xcode.app/Contents/Developer/usr/bin/git'
RESERVE = 40 * 1024**3

# BEGIN APPROVED RESEARCH IDENTITY
# Reviewed Phase 2C.13F identities. The script hash canonicalizes this block
# to a fixed sentinel; all executable code outside it is checked unchanged.
APPROVED_RESEARCH_SHA256 = {
    "Headers/CausalABCapacityCounterfactual.hpp": "8df4eb0a39998df7eb68c5601401a0a44b4c512c64552881107871a5262909ca",
    "Scripts/run_causal_ab_capacity_interventions.py": "cacc88e1c94ee4a7fe5231f2564d589d611d301e30597b2e143197a59cacad91",
    "Scripts/run_causal_fibonacci_trendline_study.sh": "6be39b8462ca8c6baa2cef82c692e74d8e7e1238324c0f43a8bca7cef2a9b5f4",
    "Sources/CausalABCapacityCounterfactual.cpp": "809bbe8209b0543e754adf7db4b5c6322ba4fff0fc2e5b3850ec9ea007a674c5",
    "Sources/CausalABCapacityInterventionCLI.cpp": "ede1d1a480fdeff43ca63962868ed4aa809e9b6dd72adfa82eedb4457403ff25",
    "Tests/CausalABCapacityInterventionTests.cpp": "944c145ee2e60cc0aee8aa2cca467f7dcca485bb79fbde479c81f80666b3d693",
    "Tests/CausalFibonacciTrendlineHistoricalEvaluationTests.sh": "f528a6278f48f3a0301be3d1d6fedb6c39648c2533fcfc4d2ac91d19b8bca099"
}
# END APPROVED RESEARCH IDENTITY

class IntegrityError(RuntimeError):
    pass

def require(condition, message):
    if not condition:
        raise IntegrityError(str(message))

def canonical_script_digest(path):
    data=Path(path).read_bytes()
    first=data.index(b'# BEGIN APPROVED RESEARCH IDENTITY\n')
    last=data.index(b'# END APPROVED RESEARCH IDENTITY\n',first)+len(b'# END APPROVED RESEARCH IDENTITY\n')
    return hashlib.sha256(data[:first]+b'# APPROVED IDENTITY BLOCK\n'+data[last:]).hexdigest()

def capture_source_hashes():
    paths=set(SOURCE_FILES)|set(RESEARCH_FILES)
    paths.update(str(p.relative_to(ROOT)) for base in ('Headers','Sources','Common')
                 for p in (ROOT/base).rglob('*') if p.is_file() and p.suffix in ('.hpp','.h','.cpp','.mm'))
    return {p:digest(ROOT/p) for p in sorted(paths)}

def reference_evidence():
    require(digest(REFERENCE/'intervention_checksums.sha256')==REFERENCE_MANIFEST_SHA256,
            'Frozen Phase 13E reference manifest identity mismatch')
    return verify(REFERENCE,'intervention_checksums.sha256')


def git(*args):
    return subprocess.check_output([GIT, '-C', str(ROOT), *args], text=True).strip()

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def rows(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))

def write_csv(path, data, fields=None):
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields or list(data[0]))
        writer.writeheader()
        writer.writerows(data)

def guard():
    require(git('branch','--show-current')=='lstm-feature-development','Unexpected repository branch')
    head=git('rev-parse','HEAD')
    ancestor=subprocess.run([GIT,'-C',str(ROOT),'merge-base','--is-ancestor',BASELINE_COMMIT,head],
                            stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,check=False)
    require(ancestor.returncode==0,'Frozen baseline commit is not an ancestor of implementation HEAD')
    require(not git('diff','--cached','--name-only'),'Staged repository changes are not allowed')
    changed=set(git('diff','--name-only').splitlines())
    untracked=set(git('ls-files','--others','--exclude-standard').splitlines())
    require(changed|untracked<=set(RESEARCH_FILES),'Unexpected non-research working-tree modifications')
    require(shutil.disk_usage('/private/tmp').free>=RESERVE,'Available disk is below 40 GiB reserve')
    require(set(APPROVED_RESEARCH_SHA256)==set(RESEARCH_FILES),'Incomplete approved research source identity')
    reference_evidence()
    expected=json.loads((REFERENCE/'intervention_protocol.json').read_text())['source_sha256']
    expected.update(APPROVED_RESEARCH_SHA256)
    observed=capture_source_hashes()
    require(set(observed)==set(expected),'Unexpected source inventory: missing or added compilation inputs')
    for name,approved in expected.items():
        actual=canonical_script_digest(ROOT/name) if name=='Scripts/run_causal_ab_capacity_interventions.py' else observed[name]
        require(actual==approved,'Unapproved source identity: '+name)
    return {'baseline_commit':BASELINE_COMMIT,'implementation_commit':head,
            'implementation_worktree_dirty':bool(changed|untracked),
            'approved_research_identity_sha256':hashlib.sha256(json.dumps(APPROVED_RESEARCH_SHA256,sort_keys=True).encode()).hexdigest()}

def verify(directory, manifest):
    directory=Path(directory)
    evidence={}
    text=(directory/manifest).read_text()
    require(bool(text.strip()),'Empty checksum manifest: '+str(directory/manifest))
    for line in text.splitlines():
        parts=line.split('  ',1)
        require(len(parts)==2,'Malformed checksum entry: '+line)
        expected,name=parts
        require(len(expected)==64 and all(c in '0123456789abcdef' for c in expected),
                'Invalid SHA256 in manifest: '+name)
        rel=Path(name)
        require(not rel.is_absolute() and name not in ('','.','..') and
                all(p not in ('.','..') for p in rel.parts),'Unsafe manifest path: '+name)
        path=directory/rel
        require(str(path) not in evidence,'Duplicate manifest entry: '+name)
        require(path.is_file() and not any((directory/Path(*rel.parts[:i])).is_symlink()
                for i in range(1,len(rel.parts)+1)),'Missing or symlinked evidence: '+str(path))
        require(path.resolve().is_relative_to(directory.resolve()),'Manifest path escapes publication: '+name)
        require(digest(path)==expected,'Evidence checksum mismatch: '+str(path))
        evidence[str(path)]=expected
    evidence[str(directory/manifest)]=digest(directory/manifest)
    return evidence

def run_logged(command, path, env=None):
    with Path(path).open('w') as log:
        result=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=False)
    require(result.returncode==0,'Validation/execution failed; see '+str(path))
    return {'command':command,'result':'PASS','log_sha256':digest(path)}

def unique_index(data, fields, label):
    indexed={}
    for row in data:
        key=tuple(row[f] for f in fields)
        require(key not in indexed,'Duplicate '+label+': '+repr(key))
        indexed[key]=row
    return indexed

def validate_victim_universe(states,expected_states):
    grouped=collections.defaultdict(dict)
    for r in states:
        require(r['decision_id'] in expected_states,'Unexpected victim decision: '+r['decision_id'])
        require(r['AB_identity'] not in grouped[r['decision_id']],'Duplicate victim identity: '+r['AB_identity'])
        grouped[r['decision_id']][r['AB_identity']]=r
    require(set(grouped)==set(expected_states),'Missing victim decision universe')
    for identity,expected in expected_states.items():
        require(set(grouped[identity])==set(expected),'Incomplete victim identity universe: '+identity)
        for key,e in expected.items():
            r=grouped[identity][key]
            for field in ('availability_bar','age','upward','eligible_pairs','fib_near','conjunction',
                          'eviction_bar','eviction_utc_epoch','a_confirmation_utc_epoch','b_confirmation_utc_epoch'):
                require(r[field]==e[field],'Victim provenance mismatch: '+identity+' '+key+' '+field)
            for field in ('a_price','b_price'):
                require(float(r[field])==float(e[field]),'Victim payload mismatch: '+identity+' '+key+' '+field)
            require(0<=int(r['age'])<=2048 and int(r['eviction_utc_epoch'])<=int(identity.rsplit(':',1)[1]),
                    'Victim age or eviction causality invalid: '+key)
    return {'decisions':len(grouped),'victim_occurrences':len(states),'identity_and_provenance':'PASS'}

def reconcile(decisions, subsets, attributes, expected_states):
    """Independent Python reconstruction of sufficiency/minimality, not C++ flags."""
    d_index=unique_index(decisions,['decision_id'],'decision')
    s_index=unique_index(subsets,['decision_id','mask'],'subset')
    a_index=unique_index(attributes,['decision_id','AB_identity'],'AB attribution')
    require(len(decisions)==39 and len(subsets)==196 and len(attributes)==71,
            'Independent reconciliation requires 39 decisions, 196 subsets and 71 occurrences')
    require(set(k[0] for k in d_index)==set(expected_states),'Decision completeness mismatch')
    require(set(k[0] for k in s_index)==set(expected_states),'Unexpected or missing subset decisions')
    require(set(k[0] for k in a_index)==set(expected_states),'Unexpected or missing attribution decisions')
    minimal_count=0
    for d in decisions:
        identity=d['decision_id']
        require(int(d['decision_id'].rsplit(':',1)[1])==int(d['bar_start_utc_epoch']),
                'Decision timestamp identity mismatch: '+identity)
        require(d['baseline_parity']==d['complete_parity']=='PASS','Decision baseline/full parity failed: '+identity)
        ss=[s for s in subsets if s['decision_id']==identity]
        aa=sorted((a for a in attributes if a['decision_id']==identity),key=lambda a:a['AB_identity'])
        q=len(aa)
        require(q==int(d['qualifying_ABs'])<=5,'Qualifying attribution completeness: '+identity)
        require({int(s['mask']) for s in ss}==set(range(1<<q)), 'Missing/duplicate exhaustive masks: '+identity)
        require(len(ss)==int(d['subset_evaluations'])==1<<q,'Subset evaluation count mismatch: '+identity)
        target=tuple(d['complete_'+k] for k in ('Ffib','Lline','I','cohort'))
        baseline=tuple(d['actual_'+k] for k in ('Ffib','Lline','I','cohort'))
        masks={int(s['mask']):s for s in ss}
        require(tuple(masks[0][k] for k in ('Ffib','Lline','I','cohort'))==baseline,
                'Independent baseline classification mismatch: '+identity)
        require(masks[0]['AB_identity_sha256']==d['actual_AB_identity_sha256'],
                'Independent baseline AB identity mismatch: '+identity)
        require(tuple(masks[(1<<q)-1][k] for k in ('Ffib','Lline','I','cohort'))==target,
                'Independent all-qualifying/full restoration mismatch: '+identity)
        sufficient=set()
        for mask,s in masks.items():
            selected=[a['AB_identity'] for i,a in enumerate(aa) if mask&(1<<i)]
            require(s['AB_identities']==';'.join(selected),'Subset candidate identities mismatch: '+identity)
            matches=tuple(s[k] for k in ('Ffib','Lline','I','cohort'))==target
            require(matches==(s['sufficient']=='1'),'Sufficiency flag mismatch: '+identity)
            if matches: sufficient.add(mask)
        minimal={mask for mask in sufficient if not any(other!=mask and other&mask==other for other in sufficient)}
        require(bool(minimal),'No sufficient subset: '+identity)
        for mask,s in masks.items():
            require((mask in minimal)==(s['minimal_sufficient']=='1'),'Minimal-subset mismatch: '+identity)
        minimal_count+=len(minimal)
        singles=sum(1 for mask in sufficient if mask.bit_count()==1)
        require(singles==int(d['individually_sufficient_ABs']),'Individual sufficient count mismatch: '+identity)
        require(len(minimal)==int(d['minimal_sufficient_subsets']),'Minimal sufficient count mismatch: '+identity)
        require((min(m.bit_count() for m in minimal)>1)==(d['requires_multiple_ABs']=='1'),
                'Joint-necessity flag mismatch: '+identity)
        require(int(d['redundant_sufficient_ABs'])==(singles if singles>1 else 0),'Redundancy count mismatch: '+identity)
        for i,a in enumerate(aa):
            mask=1<<i
            require(tuple(a[k] for k in ('Ffib','Lline','I','cohort'))==tuple(masks[mask][k] for k in ('Ffib','Lline','I','cohort')),
                    'Individual AB classification mismatch: '+identity)
            require((mask in sufficient)==(a['individually_sufficient']=='1'),'Individual AB sufficiency mismatch: '+identity)
            require(all(m&mask for m in sufficient)==(a['necessary_in_every_sufficient_subset']=='1'),
                    'AB necessity flag mismatch: '+identity)
            require(sum(bool(m&mask) for m in minimal)==int(a['minimal_subset_memberships']),
                    'AB minimal-subset memberships mismatch: '+identity)
            require((mask in sufficient and singles>1)==(a['redundant_sufficient']=='1'),
                    'AB redundancy flag mismatch: '+identity)
    return {'decisions':len(decisions),'qualifying_occurrences':len(attributes),'subsets':len(subsets),
            'minimal_sufficient_subsets':minimal_count,'all_flags_independently_reconciled':True}

def compare_reference(decisions,subsets,attributes):
    comparisons={}
    for name,data,fields in [('intervention_decisions.csv',decisions,['decision_id']),
                             ('intervention_subsets.csv',subsets,['decision_id','mask']),
                             ('intervention_AB_attribution.csv',attributes,['decision_id','AB_identity'])]:
        original=unique_index(rows(REFERENCE/name),fields,'reference '+name)
        new=unique_index(data,fields,name)
        require(new==original,'Unexplained result difference from Phase 13E: '+name)
        comparisons[name]={'rows':len(new),'all_fields_equal':True,'reference_sha256':digest(REFERENCE/name)}
    return comparisons

def verify_evidence_coverage(directory):
    directory=Path(directory)
    evidence=verify(directory,'intervention_checksums.sha256')
    covered={Path(path).relative_to(directory).as_posix() for path in evidence}
    actual={p.relative_to(directory).as_posix() for p in directory.rglob('*') if p.is_file()}
    require(not any(p.is_symlink() for p in directory.rglob('*')),'Symlink in published evidence')
    require(actual-{'INTERVENTION_COMPLETE'}==covered,'Checksum coverage omits or adds published evidence')
    required={'intervention_protocol.json','intervention_decisions.csv','intervention_subsets.csv',
              'intervention_AB_attribution.csv','intervention_validation.json','intervention_summary.csv',
              'focused_tests.log','python_integrity_tests.log','stage/first/state_validation.csv',
              'stage/repeat/state_validation.csv','stage/timing.csv','stage/first/resource.csv',
              'stage/repeat/resource.csv','independent_reconciliation.json','reference_comparison.json',
              'source_identity.json','victim_universe_validation.json',*(f'validation_{i}.log' for i in range(4))}
    require(required<=covered,'Required supporting evidence is absent from checksum coverage')
    return evidence

def verify_publication(directory):
    directory=Path(directory)
    require(not directory.name.endswith('.pending'),'Publication is still staged, not COMPLETE')
    marker=json.loads((directory/'INTERVENTION_COMPLETE').read_text())
    require(marker['status']=='COMPLETE' and marker['primary_validity']=='INVALID_UNCHANGED',
            'Invalid intervention completion marker')
    require(marker['manifest_sha256']==digest(directory/'intervention_checksums.sha256'),
            'Completion marker manifest identity mismatch')
    evidence=verify_evidence_coverage(directory)
    validation=json.loads((directory/'intervention_validation.json').read_text())
    require(validation['status']=='VERIFIED' and validation['baseline_parity']=='PASS' and
            validation['complete_restoration_parity']=='PASS','Incomplete intervention validation')
    return evidence

def sync_directory(path):
    fd=os.open(path,os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)

def publish_atomic(out,final):
    require(out.name.endswith('.pending') and not final.exists(),'Atomic publication needs a new destination')
    files=sorted(p for p in out.rglob('*') if p.is_file())
    require(not any(p.is_symlink() for p in out.rglob('*')),'Publication contains symlink')
    require(sum(p.stat().st_size for p in files)<=256*1024*1024,'Published evidence exceeds 256 MiB budget')
    manifest=out/'intervention_checksums.sha256'
    manifest.write_text(''.join(digest(p)+'  '+p.relative_to(out).as_posix()+'\n' for p in files))
    verify_evidence_coverage(out)
    marker={'schema':'causal-ab-selective-intervention-publication-v2','status':'COMPLETE',
            'primary_validity':'INVALID_UNCHANGED','manifest_sha256':digest(manifest)}
    (out/'INTERVENTION_COMPLETE').write_text(json.dumps(marker,sort_keys=True)+'\n')
    # COMPLETE is authoritative only outside .pending, with full manifest
    # validation. The directory rename is the single publication commit.
    for path in out.rglob('*'):
        if path.is_file():
            with path.open('rb') as f: os.fsync(f.fileno())
    for directory in sorted((p for p in out.rglob('*') if p.is_dir()),key=lambda p:len(p.parts),reverse=True):
        sync_directory(directory)
    sync_directory(out)
    require(shutil.disk_usage(out).free>=RESERVE,'Disk reserve exhausted before publication')
    out.rename(final)
    sync_directory(final.parent)
    verify_publication(final)

def build(directory):
    common = [str(ROOT/p) for p in ('Sources/TG4HistoricalEmpiricalEvaluation.cpp',
        'Sources/CausalFibonacciTrendlineHistoricalArtifact.cpp',
        'Sources/CausalFractalHistoryShadowAudit.cpp', 'Sources/CausalABCapacityCounterfactual.cpp')]
    flags = [CXX, '-std=c++20', '-O2', '-Wall', '-Wextra', '-Werror', '-pedantic',
             '-isysroot', SDK, '-I'+str(ROOT/'Headers'), '-I'+str(ROOT/'Sources')]
    test = directory/'intervention-tests'
    run_logged(flags + common + [str(ROOT/'Tests/CausalABCapacityInterventionTests.cpp'), '-lz', '-o', str(test)], directory/'focused_build.log')
    run_logged([str(test)], directory/'focused_tests.log')
    binary = directory/'intervention-batch'
    pkg_flags = subprocess.check_output(['pkg-config', '--cflags', 'libpqxx'], text=True).split()
    pkg_libs = subprocess.check_output(['pkg-config', '--libs', 'libpqxx'], text=True).split()
    run_logged(flags + ['-Wno-c++23-attribute-extensions'] + pkg_flags + common +
        [str(ROOT/p) for p in ('Sources/TG4HistoricalMarketDataRepository.cpp',
         'Sources/RuntimeDatabaseConnection.cpp','Common/HistoricalFxTimestamp.cpp',
         'Sources/CausalABCapacityInterventionCLI.cpp')] + pkg_libs + ['-lz','-o',str(binary)], directory/'standalone_build.log')
    return binary

def validate_regressions(out):
    env=os.environ.copy();env.pop('PYTHONOPTIMIZE',None);env['CXX']=CXX;env['SDKROOT']=SDK
    commands=[['bash','Tests/CausalFibonacciTrendlineHistoricalEvaluationTests.sh'],
              ['bash','Tests/TG3FibonacciConfluenceIntegrationTests.sh'],
              ['bash','Tests/TG1ACausalFractalTrendLineGeometryTests.sh'],
              ['bash','Scripts/run_causal_fibonacci_trendline_study.sh','--build-only']]
    results=[]
    for index,command in enumerate(commands):
        path=out/('validation_'+str(index)+'.log')
        with path.open('w') as log:
            subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        results.append({'command':command,'result':'PASS','log_sha256':digest(path)})
        print('VALIDATION_PASS '+' '.join(command),flush=True)
    return results

def prepare(out):
    forensics = Path('/private/tmp/ea_phase2c13c_forensics')
    preparation = Path('/private/tmp/ea_phase2c13d_intervention_preparation')
    evidence = verify(forensics, 'checksums.sha256') | verify(preparation, 'checksums.sha256') | reference_evidence()
    changed = rows(forensics/'changed_decisions.csv')
    workload = rows(preparation/'intervention_workload.csv')
    require(len(changed) == 39 and len(workload) == 71, 'Integrity check failed: len(changed) == 39 and len(workload) == 71')
    require(len({r['AB_identity'] for r in workload}) == 49, "Integrity check failed: len({r['AB_identity'] for r in workload}) == 49")
    require(len(rows(preparation/'anomalies.csv')) == 0, "Integrity check failed: len(rows(preparation/'anomalies.csv')) == 0")
    qualified = collections.defaultdict(list)
    for w in workload:
        qualified[w['decision_id']].append(w['AB_identity'])
    require(len(qualified)==39, 'Integrity check failed: len(qualified)==39')
    previous = {
        '2013-01': Path('/private/tmp/ea_phase2c12a_ab_audit_20261013_140148_18144'),
        '2013-07': Path('/private/tmp/ea_phase2c13a_20261013_161025_36950/audit_07'),
    }
    audit_maps, expected_states, checkpoints = {}, {}, {}
    for period in sorted({r['period'] for r in changed}):
        audit = previous.get(period,Path('/private/tmp/ea_phase2c13b_quarterly')/('audit_'+period))
        evidence.update(verify(audit,'audit_checksums.sha256'))
        marker=(audit/'AUDIT_COMPLETE').read_text()
        require('checksums_sha256='+digest(audit/'audit_checksums.sha256') in marker, "Integrity check failed: 'checksums_sha256='+digest(audit/'audit_checksums.sha256') in marker")
        require('primary_validity=UNCHANGED' in marker, "Integrity check failed: 'primary_validity=UNCHANGED' in marker")
        evidence[str(audit/'AUDIT_COMPLETE')]=digest(audit/'AUDIT_COMPLETE')
        provenance=json.loads((audit/'audit_provenance.json').read_text())
        protocol=json.loads((audit/'audit_protocol.json').read_text())
        summary={r['metric']:r['value'] for r in rows(audit/'summary.csv')}
        require(summary['actual_complete']==summary['counterfactual_complete']==summary['parity']=='1', "Integrity check failed: summary['actual_complete']==summary['counterfactual_complete']==summary['parity']=='1'")
        require(summary['failure']=='' and summary['unknown']=='0', "Integrity check failed: summary['failure']=='' and summary['unknown']=='0'")
        require(protocol['source_AB_capacity']==512 and protocol['AB_age_inclusive']==2048 and not protocol['restore_fractal_history'], "Integrity check failed: protocol['source_AB_capacity']==512 and protocol['AB_age_inclusive']==2048 and not protocol['restore_fractal_history']")
        require(provenance['parent_status']=='INVALID', "Integrity check failed: provenance['parent_status']=='INVALID'")
        require(provenance['TG3_source_sha256']==digest(ROOT/'Headers/CausalFibonacciConfluenceIntegration.hpp'), "Integrity check failed: provenance['TG3_source_sha256']==digest(ROOT/'Headers/CausalFibonacciConfluenceIntegration.hpp')")
        require(provenance['TG1A_source_sha256']==digest(ROOT/'Headers/CausalFractalTrendLineGeometry.hpp'), "Integrity check failed: provenance['TG1A_source_sha256']==digest(ROOT/'Headers/CausalFractalTrendLineGeometry.hpp')")
        require(protocol['ratios']==[.382,.5,.618] and protocol['market_threshold']==protocol['structural_threshold']==.25, "Integrity check failed: protocol['ratios']==[.382,.5,.618] and protocol['market_threshold']==protocol['structural_threshold']==.25")
        # Resolve by recorded content linkage, never presume a directory pairing.
        parent_candidates = list(Path('/private/tmp/ea_phase2c13b_quarterly').glob('primary_*'))
        parent_candidates += list(Path('/private/tmp/ea_phase2c13a_20261013_161025_36950').glob('primary_*'))
        parent_candidates += list(Path('/private/tmp').glob('ea_phase2c12a_primary_*'))
        matches=[p for p in parent_candidates if (p/'checksums.sha256').is_file() and digest(p/'checksums.sha256')==provenance['parent_checksums_sha256']]
        require(len(matches)==1, (period,'parent linkage',matches))
        parent=matches[0]
        evidence.update(verify(parent,'checksums.sha256'))
        require((parent/'INVALID').is_file() and not (parent/'COMPLETE').exists(), "Integrity check failed: (parent/'INVALID').is_file() and not (parent/'COMPLETE').exists()")
        require(digest(parent/'INVALID')==provenance['parent_marker_sha256'], "Integrity check failed: digest(parent/'INVALID')==provenance['parent_marker_sha256']")
        evidence[str(parent/'INVALID')]=digest(parent/'INVALID')
        primary=json.loads((parent/'provenance_manifest.json').read_text())
        require(primary['canonical_candle_content_sha256']==provenance['canonical_candle_sha256'], "Integrity check failed: primary['canonical_candle_content_sha256']==provenance['canonical_candle_sha256']")
        require(primary['configuration_source_sha256']==digest(ROOT/'Scripts/tg4_analysis_config.frozen_v1.conf'), "Integrity check failed: primary['configuration_source_sha256']==digest(ROOT/'Scripts/tg4_analysis_config.frozen_v1.conf')")
        checkpoints[protocol['score_end']] = (provenance['canonical_candle_sha256'],int(summary['bars']))
        candidates=collections.defaultdict(list)
        for r in rows(audit/'restored_candidates.csv'):
            candidates[r['decision_id']].append(r)
        evictions={r['AB_identity']:r for r in rows(audit/'source_evictions.csv')}
        require(len(evictions)==int(summary['source_evictions_full']), "Integrity check failed: len(evictions)==int(summary['source_evictions_full'])")
        audit_decisions={r['decision_id']:r for r in rows(audit/'decisions.csv')}
        primary_decisions={r['decision_id']:r for r in rows(parent/'decisions.csv')}
        audit_maps[period]=(audit_decisions,primary_decisions)
        for ch in (r for r in changed if r['period']==period):
            identity=ch['decision_id'];d=audit_decisions[identity];pr=primary_decisions[identity]
            require(d['status']=='RECONSTRUCTED' and d['primary_status']=='capacity_invalid', "Integrity check failed: d['status']=='RECONSTRUCTED' and d['primary_status']=='capacity_invalid'")
            require(pr['status']=='capacity_invalid' and pr['universe_complete']=='0' and pr['geometry_evaluated']=='0', "Integrity check failed: pr['status']=='capacity_invalid' and pr['universe_complete']=='0' and pr['geometry_evaluated']=='0'")
            require(d['actual_cohort']+'->'+d['counterfactual_cohort']==ch['transition'], "Integrity check failed: d['actual_cohort']+'->'+d['counterfactual_cohort']==ch['transition']")
            q={c['AB_identity'] for c in candidates[identity] if c['upward']=='1' and (c['fib_near']=='1' or c['conjunction']=='1')}
            require(q==set(qualified[identity]) and len(q)==len(qualified[identity]), 'Integrity check failed: q==set(qualified[identity]) and len(q)==len(qualified[identity])')
            require(len(candidates[identity])==int(ch['restored_candidate_rows'])==int(d['restored_present']), "Integrity check failed: len(candidates[identity])==int(ch['restored_candidate_rows'])==int(d['restored_present'])")
            states={}
            for c in candidates[identity]:
                e=evictions[c['AB_identity']]
                require(e['reason']=='SOURCE_AB_CAPACITY' and int(e['bar_start_utc_epoch'])<=int(d['bar_start_utc_epoch']), "Integrity check failed: e['reason']=='SOURCE_AB_CAPACITY' and int(e['bar_start_utc_epoch'])<=int(d['bar_start_utc_epoch'])")
                states[c['AB_identity']]={**c,'eviction_bar':e['bar'],'eviction_utc_epoch':e['bar_start_utc_epoch'],
                    'a_price':e['a_price'],'b_price':e['b_price'],'a_confirmation_utc_epoch':e['a_confirmation_utc_epoch'],
                    'b_confirmation_utc_epoch':e['b_confirmation_utc_epoch']}
            expected_states[identity]=states
    plan=[]
    for ch in changed:
        identity=ch['decision_id'];d,pr=(m[identity] for m in audit_maps[ch['period']])
        bars={int(r['availability_bar'])+int(r['age']) for r in expected_states[identity].values()}
        require(len(bars)==1, 'Integrity check failed: len(bars)==1')
        plan.append(['D',d['bar_start_utc_epoch'],identity,str(bars.pop()),pr['close'],pr['atr'],
            d['actual_Ffib'],d['actual_Lline'],d['actual_I'],d['actual_cohort'],d['actual_ABs'],d['lines'],
            d['counterfactual_Ffib'],d['counterfactual_Lline'],d['counterfactual_I'],d['counterfactual_cohort'],
            d['counterfactual_ABs'],d['restored_present'],ch['period'],'0','0',';'.join(sorted(qualified[identity]))])
    plan.sort(key=lambda r:int(r[1]))
    plan += [['C',str(end),h,str(count)] for end,(h,count) in sorted(checkpoints.items())]
    (out/'plan.tsv').write_text(''.join('\t'.join(row)+'\n' for row in plan))
    return evidence,expected_states,workload

def finish(out,evidence,expected_states,workload,start,source_hashes,binary,tests,identity):
    stage=out/'stage/first'
    decisions=rows(stage/'decisions.csv');subsets=rows(stage/'subsets.csv');attributes=rows(stage/'attribution.csv');states=rows(stage/'state_validation.csv')
    require(len(decisions)==39 and len(attributes)==71,'Incomplete batch: expected 39 decisions and 71 candidate occurrences')
    require(sum(p.stat().st_size for p in out.rglob('*') if p.is_file())<=256*1024*1024, 'diagnostic disk budget')
    require({r['decision_id'] for r in decisions}==set(expected_states), "Integrity check failed: {r['decision_id'] for r in decisions}==set(expected_states)")
    require({(r['decision_id'],r['AB_identity']) for r in attributes}=={(r['decision_id'],r['AB_identity']) for r in workload}, "Integrity check failed: {(r['decision_id'],r['AB_identity']) for r in attributes}=={(r['decision_id'],r['AB_identity']) for r in workload}")
    victim_validation=validate_victim_universe(states,expected_states)
    for d in decisions:
        require(d['baseline_parity']==d['complete_parity']=='PASS', "Integrity check failed: d['baseline_parity']==d['complete_parity']=='PASS'")
        ss=[s for s in subsets if s['decision_id']==d['decision_id']]
        require(len(ss)==2**int(d['qualifying_ABs'])==int(d['subset_evaluations']), "Integrity check failed: len(ss)==2**int(d['qualifying_ABs'])==int(d['subset_evaluations'])")
        require(ss[0]['AB_identity_sha256']==d['actual_AB_identity_sha256'], "Integrity check failed: ss[0]['AB_identity_sha256']==d['actual_AB_identity_sha256']")
        require(all(s['sufficient']=='1' for s in ss if s['minimal_sufficient']=='1'), "Integrity check failed: all(s['sufficient']=='1' for s in ss if s['minimal_sufficient']=='1')")
    for path,expected in evidence.items():
        require(digest(path)==expected, ('input modified',path))
    require(all(digest(ROOT/p)==h for p,h in source_hashes.items()), 'source changed during batch')
    require(guard()==identity,'Repository identity changed during batch')
    reconciliation=reconcile(decisions,subsets,attributes,expected_states)
    comparison=compare_reference(decisions,subsets,attributes)
    summary={
        'decisions_analyzed':len(decisions),'candidate_occurrences':len(attributes),
        'distinct_AB_structures':len({r['AB_identity'] for r in attributes}),
        'individually_sufficient_occurrences':sum(r['individually_sufficient']=='1' for r in attributes),
        'individually_sufficient_distinct_ABs':len({r['AB_identity'] for r in attributes if r['individually_sufficient']=='1'}),
        'decisions_requiring_multiple_ABs':sum(r['requires_multiple_ABs']=='1' for r in decisions),
        'decisions_with_redundant_ABs':sum(int(r['redundant_sufficient_ABs'])>0 for r in decisions),
        'unresolved_decisions':0,'baseline_parity_failures':0,'complete_restoration_parity_failures':0,
        'unexplained_identity_or_provenance_failures':0,'subset_evaluations':len(subsets),
        'verified_victim_occurrences':len(states),
    }
    timing={r['metric']:r['value'] for r in rows(out/'stage/timing.csv')}
    resources={r['metric']:r['value'] for r in rows(stage/'resource.csv')}
    require(summary=={**summary,
        'decisions_analyzed':39,'candidate_occurrences':71,'distinct_AB_structures':49,
        'individually_sufficient_occurrences':65,'individually_sufficient_distinct_ABs':44,
        'decisions_requiring_multiple_ABs':0,'decisions_with_redundant_ABs':15,'unresolved_decisions':0,
        'baseline_parity_failures':0,'complete_restoration_parity_failures':0,'subset_evaluations':196},
        'Review acceptance counts changed unexpectedly')
    require(int(resources['peak_live_ABs'])<=8192 and int(resources['pairs'])<=1024*1024*1024,
            'Replay resource budgets exceeded')
    require(int(timing['max_rss_bytes'])<=256*1024*1024,'Observed RSS exceeds intervention memory budget')
    require(all(int(d['pairs'])<=64*1024*1024 for d in decisions),'Per-decision pair budget exceeded')
    require(float(timing['total_seconds'])>0,'Invalid runtime measurement')
    for filename,body in [('independent_reconciliation.json',reconciliation),
                          ('reference_comparison.json',comparison),('source_identity.json',identity),
                          ('victim_universe_validation.json',victim_validation)]:
        (out/filename).write_text(json.dumps(body,indent=2,sort_keys=True)+'\n')
    validation={'schema':'causal-ab-selective-intervention-validation-v1','status':'VERIFIED',
        'baseline_parity':'PASS','complete_restoration_parity':'PASS','native_AB_identity_and_order_parity':'PASS',
        'complete_victim_universe_parity':'PASS','canonical_prefix_digest_parity':'PASS',
        'repeated_execution_byte_parity':'PASS','input_immutability':'PASS',
        'primary_validity':'INVALID_UNCHANGED','outcomes_evaluated':False,'statistical_estimation':'NOT_EXECUTED',
        'focused_tests':'PASS','existing_regression_tests':tests,'resources':resources,'timing':timing,
        'total_runtime_seconds':time.monotonic()-start,'independent_reconciliation':reconciliation,
        'reference_comparison':comparison,**summary}
    protocol={'schema':'causal-ab-selective-intervention-v2','repository_commit':identity['implementation_commit'],**identity,
        'source_sha256':source_hashes,'executable_sha256':digest(binary),
        'input_sha256':evidence,'causal_origin':1262304000,'AB_age_inclusive':2048,
        'source_AB_capacity':512,'TG3_history_per_kind':128,'TG1A_capacity':4096,'TG1A_history_per_kind':64,
        'ratios':[.382,.5,.618],'market_threshold_ATR':.25,'structural_threshold_ATR':.25,
        'intervention':'retain only selected existing source-capacity victims; identical actual lines and ATR',
        'sufficiency':'exact Ffib,Lline,I and cohort equality with complete restoration',
        'subset_order':'native identity lexical, bit masks ascending, exhaustive qualifying subsets',
        'universe_verification':'all live victims, including nonqualifying and DownAB, verified against existing ledger',
        'primary_validity':'INVALID_UNCHANGED','memory_accounting':'native mirrors; peak RSS recorded',
        'live_AB_budget':8192,'per_decision_pair_budget':64*1024*1024,'batch_pair_budget':1024*1024*1024,
        'disk_reserve_bytes':RESERVE,'decoded_diagnostic_output_budget':256*1024*1024,'max_qualifiers':5,'database_access':'READ ONLY REPEATABLE READ; closed before replay',
        'data_cutoff':'2019-11-01T00:00:00Z; no 2025','reconstruction_passes':2,
        'completion_rule':'Atomic directory rename from .pending; INTERVENTION_COMPLETE and full evidence manifest required',
        'checksum_exclusions':['intervention_checksums.sha256','INTERVENTION_COMPLETE'],
        'interpretation':'detector-geometry intervention only; no predictive validity or statistical significance'}
    (out/'intervention_protocol.json').write_text(json.dumps(protocol,indent=2,sort_keys=True)+'\n')
    (out/'intervention_validation.json').write_text(json.dumps(validation,indent=2,sort_keys=True)+'\n')
    write_csv(out/'intervention_summary.csv',[{'metric':k,'value':v} for k,v in summary.items()]+[{'metric':'total_runtime_seconds','value':validation['total_runtime_seconds']}])
    write_csv(out/'intervention_decisions.csv',decisions)
    write_csv(out/'intervention_subsets.csv',subsets)
    write_csv(out/'intervention_AB_attribution.csv',attributes)
    final=out.with_name(out.name.removesuffix('.pending'))
    publish_atomic(out,final)
    print(json.dumps(validation,indent=2))
    print('OUTPUT_DIRECTORY='+str(final))
    return final

def integrity_tests():
    """Destructive fault injection is confined to temporary fixtures/mocks.
    Run this entry point under python3 -O to prove checks cannot be disabled.
    """
    import copy
    import io
    reference_evidence()
    original_d=rows(REFERENCE/'intervention_decisions.csv')
    original_s=rows(REFERENCE/'intervention_subsets.csv')
    original_a=rows(REFERENCE/'intervention_AB_attribution.csv')
    expected_ids={d['decision_id']:{} for d in original_d}
    checks=[]
    def rejects(label,call):
        try: call()
        except IntegrityError: checks.append(label);return
        raise IntegrityError('Fault injection was not rejected: '+label)
    # No temporary Git commits: simulate a descendant while retaining actual
    # source approval checks. An ancestor failure or unexpected source still fails.
    answers={args:git(*args) for args in [('branch','--show-current'),('rev-parse','HEAD'),
             ('diff','--cached','--name-only'),('diff','--name-only'),('ls-files','--others','--exclude-standard')]}
    def descendant_git(*args):
        if args==('rev-parse','HEAD'):return 'a'*40
        return answers[args]
    with patch(__name__+'.git',side_effect=descendant_git),patch('subprocess.run',return_value=subprocess.CompletedProcess([],0)):
        require(guard()['implementation_commit']=='a'*40,'Approved descendant HEAD rejected')
    checks.append('approved_descendant_HEAD')
    with patch(__name__+'.git',side_effect=lambda *args:answers[args]),patch('subprocess.run',return_value=subprocess.CompletedProcess([],1)):
        rejects('non_ancestor_HEAD',guard)
    hashes=capture_source_hashes();hashes['Sources/CausalABCapacityCounterfactual.cpp']='0'*64
    with patch(__name__+'.capture_source_hashes',return_value=hashes):
        rejects('unexpected_source_hash',guard)
    with patch('shutil.disk_usage',return_value=shutil._ntuple_diskusage(100,99,1)):
        rejects('disk_reserve',guard)
    with tempfile.TemporaryDirectory(prefix='ea_intervention_integrity_',dir='/private/tmp') as task:
        root=Path(task);(root/'input.csv').write_text('verified input\n')
        (root/'checksums.sha256').write_text(digest(root/'input.csv')+'  input.csv\n')
        verify(root,'checksums.sha256');(root/'input.csv').write_text('corrupted\n')
        rejects('input_checksum_corruption',lambda:verify(root,'checksums.sha256'))
        (root/'checksums.sha256').write_text('0'*64+'  ../escape\n')
        rejects('manifest_path_escape',lambda:verify(root,'checksums.sha256'))
        rel='child/log.txt';(root/'child').mkdir();(root/rel).write_text('log\n')
        (root/'checksums.sha256').write_text(digest(root/rel)+'  '+rel+'\n')
        verify(root,'checksums.sha256');checks.append('nested_evidence_checksums')
    valid=reconcile(original_d,original_s,original_a,expected_ids)
    require(valid['subsets']==196 and valid['minimal_sufficient_subsets']==65,'Original independent subset validation')
    checks.append('196_independent_subsets')
    rejects('candidate_completeness',lambda:reconcile(original_d,original_s,original_a[:-1],expected_ids))
    rejects('subset_completeness',lambda:reconcile(original_d,original_s[:-1],original_a,expected_ids))
    corrupt=copy.deepcopy(original_d);corrupt[0]['actual_Ffib']=str(1-int(corrupt[0]['actual_Ffib']))
    rejects('baseline_parity',lambda:reconcile(corrupt,original_s,original_a,expected_ids))
    corrupt=copy.deepcopy(original_d);corrupt[0]['complete_I']=str(1-int(corrupt[0]['complete_I']))
    rejects('complete_restoration_parity',lambda:reconcile(corrupt,original_s,original_a,expected_ids))
    corrupt=copy.deepcopy(original_s);corrupt[0]['minimal_sufficient']='1'
    rejects('minimal_subset_flags',lambda:reconcile(original_d,corrupt,original_a,expected_ids))
    corrupt=copy.deepcopy(original_d);corrupt[0]['bar_start_utc_epoch']='0'
    rejects('decision_timestamp',lambda:reconcile(corrupt,original_s,original_a,expected_ids))
    # Full victim-universe evidence is supporting data in the old run, independently
    # checked against the new pass before a v2 publication can commit.
    victim_rows=rows(REFERENCE/'stage/first/state_validation.csv')
    expected=collections.defaultdict(dict)
    for r in victim_rows:expected[r['decision_id']][r['AB_identity']]=r
    validate_victim_universe(victim_rows,expected)
    rejects('full_victim_completeness',lambda:validate_victim_universe(victim_rows[:-1],expected))
    corrupt=copy.deepcopy(victim_rows);corrupt[0]['eviction_utc_epoch']='9999999999'
    rejects('victim_provenance',lambda:validate_victim_universe(corrupt,expected))
    with tempfile.TemporaryDirectory(prefix='ea_intervention_publish_',dir='/private/tmp') as task:
        root=Path(task)
        names=['intervention_protocol.json','intervention_decisions.csv','intervention_subsets.csv',
               'intervention_AB_attribution.csv','intervention_summary.csv','focused_tests.log','python_integrity_tests.log',
               'independent_reconciliation.json','reference_comparison.json','source_identity.json',
               'victim_universe_validation.json','stage/first/state_validation.csv','stage/repeat/state_validation.csv',
               'stage/timing.csv','stage/first/resource.csv','stage/repeat/resource.csv',
               *(f'validation_{i}.log' for i in range(4))]
        def fixture(name):
            p=root/(name+'.pending');p.mkdir()
            for rel in names:
                f=p/rel;f.parent.mkdir(parents=True,exist_ok=True);f.write_text('fixture evidence\n')
            (p/'intervention_validation.json').write_text(json.dumps({'status':'VERIFIED','baseline_parity':'PASS','complete_restoration_parity':'PASS'}))
            return p
        staged=fixture('ok');final=root/'ok';publish_atomic(staged,final);verify_publication(final)
        checks.append('atomic_complete_publication')
        (final/'validation_0.log').write_text('corrupted log\n')
        rejects('regression_log_checksum',lambda:verify_publication(final))
        staged=fixture('interrupted')
        rejects('interrupted_not_complete',lambda:verify_publication(staged))
        with patch(__name__+'.sync_directory',side_effect=IntegrityError('injected precommit interruption')):
            rejects('precommit_failure',lambda:publish_atomic(staged,root/'interrupted'))
        require(not (root/'interrupted').exists(),'Interrupted run was published')
        rejects('staged_marker_not_complete',lambda:verify_publication(staged))
        staged=fixture('omitted')
        with patch(__name__+'.verify_evidence_coverage',side_effect=IntegrityError('injected missing evidence')):
            rejects('incomplete_manifest_not_complete',lambda:publish_atomic(staged,root/'omitted'))
        require(not (root/'omitted').exists(),'Incomplete evidence published')
    # The script's AST must contain no integrity-critical Python assert statements.
    require(not any(isinstance(n,ast.Assert) for n in ast.walk(ast.parse(Path(__file__).read_text()))),
            'Python assert statements remain in integrity implementation')
    print(json.dumps({'schema':'intervention-python-integrity-tests-v1','optimization_level':sys.flags.optimize,
                      'result':'PASS','checks':checks},indent=2))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test-only',action='store_true')
    parser.add_argument('--integrity-tests',action='store_true')
    args=parser.parse_args()
    if args.integrity_tests:
        integrity_tests()
        return
    start=time.monotonic();identity=guard()
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')
    prefix='ea_ab_intervention_tests_' if args.test_only else 'ea_phase2c13f_interventions_'
    out=Path('/private/tmp')/(prefix+stamp+'_'+str(os.getpid())+'.pending')
    out.mkdir();print('STAGING_DIRECTORY='+str(out),flush=True)
    source_hashes=capture_source_hashes()
    binary=build(out)
    run_logged([sys.executable,'-O',str(Path(__file__).resolve()),'--integrity-tests'],out/'python_integrity_tests.log')
    if args.test_only:
        require(guard()==identity,'Repository changed during focused validation')
        print('Focused, Python -O integrity tests and standalone build PASS; database_access=false')
        return
    evidence,expected_states,workload=prepare(out)
    tests=validate_regressions(out)
    env=os.environ.copy();env['FOREX_DB_NAME']='forex'
    env['PGOPTIONS']='-c default_transaction_read_only=on -c statement_timeout=600000 -c lock_timeout=1000'
    env['PGCONNECT_TIMEOUT']='5'
    run_logged([str(binary),str(ROOT/'Scripts/tg4_analysis_config.frozen_v1.conf'),str(out/'plan.tsv'),str(out/'stage')],
               out/'batch_execution.log',env)
    finish(out,evidence,expected_states,workload,start,source_hashes,binary,tests,identity)

if __name__=='__main__':
    main()
