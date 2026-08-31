#!/usr/bin/env node

import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdtemp, mkdir, readFile, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";

import {
  manifestBytes,
  persistAcquisitionState,
  reconcileManifestedArtifact,
  reconcileUnmanifestedArtifact,
  retainArtifact,
} from "../EconomicCalendar/acquire_bls_releases.mjs";


const RETRIEVED_AT = "2026-08-31T12:00:00.000Z";

function descriptor() {
  return {
    artifact_kind: "release",
    bls_family: "CPI",
    link_title: "December 2018 Consumer Price Index",
    reference_period: "December 2018",
    release_date: "2019-01-11",
    source_release_identity: "cpi_01112019",
    source_url: "https://www.bls.gov/news.release/archives/cpi_01112019.htm",
  };
}

function documentBytes(extra = "") {
  const text = `
    <html><body>
    Bureau of Labor Statistics
    Transmission of material in this release is embargoed until
    8:30 a.m. (EST) January 11, 2019
    CONSUMER PRICE INDEX - DECEMBER 2018
    The Consumer Price Index for All Urban Consumers (CPI-U) declined
    0.1 percent on a seasonally adjusted basis, the U.S. Bureau of Labor
    Statistics reported today.
    ${"retained official release padding ".repeat(60)}
    ${extra}
    </body></html>`;
  return Buffer.from(text, "utf8");
}

function capture(bytes) {
  return {
    bytes,
    response: {
      headers: { "content-type": "text/html", date: "Fri, 11 Jan 2019 13:30:00 GMT" },
      mimeType: "text/html",
      protocol: "h2",
      status: 200,
    },
  };
}

function digest(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

async function main() {
  const ordered = manifestBytes([
    { source_url: "https://www.bls.gov/z", nested: { z: 1, a: 2 } },
    { source_url: "https://www.bls.gov/a", nested: { b: 2, a: 1 } },
  ]);
  assert.equal(
    ordered.toString("utf8"),
    '{"nested":{"a":1,"b":2},"source_url":"https://www.bls.gov/a"}\n' +
      '{"nested":{"a":2,"z":1},"source_url":"https://www.bls.gov/z"}\n',
  );
  assert.deepEqual(ordered, manifestBytes([
    { nested: { a: 1, b: 2 }, source_url: "https://www.bls.gov/a" },
    { nested: { a: 2, z: 1 }, source_url: "https://www.bls.gov/z" },
  ]));

  const repositoryRoot = await mkdtemp(path.join(os.tmpdir(), "ea-bls-acquire-"));
  const rawRoot = path.join(repositoryRoot, "EconomicCalendar/raw/bls");
  const manifest = new Map();
  const conflicts = [];
  const failures = [];
  const firstBytes = documentBytes();
  const first = await retainArtifact(
    descriptor(),
    capture(firstBytes),
    RETRIEVED_AT,
    manifest,
    conflicts,
    repositoryRoot,
  );
  assert.equal(first.outcome, "admitted");
  await persistAcquisitionState(manifest, conflicts, failures, rawRoot);

  // A failure after admission cannot orphan the artifact: both bytes and the
  // fsynced canonical manifest are already present and mutually hash-bound.
  const persistedRows = (await readFile(path.join(rawRoot, "manifest.jsonl"), "utf8"))
    .trim()
    .split("\n")
    .map(JSON.parse);
  assert.equal(persistedRows.length, 1);
  assert.equal(persistedRows[0].sha256, digest(firstBytes));
  const retainedPath = path.join(repositoryRoot, persistedRows[0].immutable_local_path);
  assert.equal(digest(await readFile(retainedPath)), persistedRows[0].sha256);

  const duplicate = await retainArtifact(
    descriptor(),
    capture(firstBytes),
    RETRIEVED_AT,
    manifest,
    conflicts,
    repositoryRoot,
  );
  assert.equal(duplicate.outcome, "duplicate_identical");
  assert.equal(conflicts.length, 0);

  const changedBytes = documentBytes("official page changed after first admission");
  const changed = await retainArtifact(
    descriptor(),
    capture(changedBytes),
    "2026-08-31T12:05:00.000Z",
    manifest,
    conflicts,
    repositoryRoot,
  );
  assert.equal(changed.outcome, "conflict_preserved_original");
  assert.equal(conflicts.length, 1);
  assert.equal(conflicts[0].first_admitted_sha256, digest(firstBytes));
  assert.equal(conflicts[0].fetched_sha256, digest(changedBytes));
  assert.equal(digest(await readFile(retainedPath)), digest(firstBytes));

  const resumed = await reconcileManifestedArtifact(
    descriptor(),
    persistedRows[0],
    repositoryRoot,
  );
  assert.equal(resumed.outcome, "duplicate_identical");

  const orphanRoot = await mkdtemp(path.join(os.tmpdir(), "ea-bls-orphan-"));
  const orphanPath = path.join(
    orphanRoot,
    "EconomicCalendar/raw/bls/releases/cpi/cpi_01112019.htm",
  );
  await mkdir(path.dirname(orphanPath), { recursive: true });
  await writeFile(orphanPath, firstBytes);
  const orphanManifest = new Map();
  const reconciled = await reconcileUnmanifestedArtifact(
    descriptor(),
    RETRIEVED_AT,
    orphanManifest,
    orphanRoot,
  );
  assert.equal(reconciled.outcome, "orphan_reconciled");
  assert.equal(orphanManifest.size, 1);
  assert.equal(reconciled.row.sha256, digest(firstBytes));
  assert.equal(
    reconciled.row.http_source_metadata.capture_basis,
    "validated_interrupted_cdp_raw_artifact",
  );

  const invalidRoot = await mkdtemp(path.join(os.tmpdir(), "ea-bls-invalid-"));
  const invalidPath = path.join(
    invalidRoot,
    "EconomicCalendar/raw/bls/releases/cpi/cpi_01112019.htm",
  );
  await mkdir(path.dirname(invalidPath), { recursive: true });
  await writeFile(invalidPath, Buffer.from("filename alone is not evidence"));
  const invalidManifest = new Map();
  await assert.rejects(
    reconcileUnmanifestedArtifact(
      descriptor(), RETRIEVED_AT, invalidManifest, invalidRoot,
    ),
    /invalid_or_empty_bls_document/,
  );
  assert.equal(invalidManifest.size, 0);

  console.log("BLS_ACQUISITION_TESTS=PASS");
}

await main();
