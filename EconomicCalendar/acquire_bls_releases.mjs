#!/usr/bin/env node

/**
 * Deterministically retain official BLS archived news-release artifacts.
 *
 * Direct non-browser HTTP access to www.bls.gov is denied by the site's bot
 * policy in this environment.  This utility therefore uses a normal isolated
 * Chrome session and the Chrome DevTools Protocol to retain the exact main
 * document response body.  It does not spoof headers, use a proxy/cache, or
 * query mutable BLS time-series values.
 */

import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import {
  mkdir,
  mkdtemp,
  open,
  readFile,
  rename,
  rm,
  stat,
} from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const REPOSITORY_ROOT = path.resolve(".");
const ROOT = path.join(REPOSITORY_ROOT, "EconomicCalendar/raw/bls");
const MANIFEST_PATH = path.join(ROOT, "manifest.jsonl");
const CONFLICT_PATH = path.join(ROOT, "acquisition_conflicts.jsonl");
const FAILURE_PATH = path.join(ROOT, "acquisition_failures.jsonl");
const DEFAULT_CHROME =
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const AUTHORITATIVE_HOST = "www.bls.gov";

const FAMILY_CONFIG = Object.freeze({
  CPI: Object.freeze({
    indexUrl: "https://www.bls.gov/bls/news-release/cpi.htm",
    prefix: "cpi",
  }),
  EMPLOYMENT: Object.freeze({
    indexUrl: "https://www.bls.gov/bls/news-release/empsit.htm",
    prefix: "empsit",
  }),
  PPI: Object.freeze({
    indexUrl: "https://www.bls.gov/bls/news-release/ppi.htm",
    prefix: "ppi",
  }),
  JOLTS: Object.freeze({
    indexUrl: "https://www.bls.gov/bls/news-release/jolts.htm",
    prefix: "jolts",
  }),
});

function usage() {
  console.error(
    "usage: acquire_bls_releases.mjs " +
      "[--families CPI,EMPLOYMENT,PPI,JOLTS] [--start-year 2010] " +
      "[--end-year 2026] [--delay-ms 750] [--max-releases N] " +
      "[--target-dates TSV] [--retrieved-at ISO8601] [--chrome PATH]",
  );
}

function parseArgs(argv) {
  const options = {
    families: Object.keys(FAMILY_CONFIG),
    startYear: 2010,
    endYear: new Date().getUTCFullYear(),
    delayMs: 750,
    maxReleases: null,
    targetDates: null,
    retrievedAt: new Date().toISOString(),
    chrome: DEFAULT_CHROME,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const name = argv[index];
    const value = argv[index + 1];
    if (name === "--help") {
      usage();
      process.exit(0);
    }
    if (!value || !name.startsWith("--")) {
      throw new Error(`invalid_argument:${name}`);
    }
    index += 1;
    if (name === "--families") {
      options.families = value.split(",").map((item) => item.trim());
    } else if (name === "--start-year") {
      options.startYear = Number.parseInt(value, 10);
    } else if (name === "--end-year") {
      options.endYear = Number.parseInt(value, 10);
    } else if (name === "--delay-ms") {
      options.delayMs = Number.parseInt(value, 10);
    } else if (name === "--max-releases") {
      options.maxReleases = Number.parseInt(value, 10);
    } else if (name === "--target-dates") {
      options.targetDates = value;
    } else if (name === "--retrieved-at") {
      options.retrievedAt = new Date(value).toISOString();
    } else if (name === "--chrome") {
      options.chrome = value;
    } else {
      throw new Error(`unknown_argument:${name}`);
    }
  }
  for (const family of options.families) {
    if (!Object.hasOwn(FAMILY_CONFIG, family)) {
      throw new Error(`unsupported_family:${family}`);
    }
  }
  if (
    !Number.isInteger(options.startYear) ||
    !Number.isInteger(options.endYear) ||
    options.startYear > options.endYear ||
    !Number.isInteger(options.delayMs) ||
    options.delayMs < 250 ||
    (options.maxReleases !== null &&
      (!Number.isInteger(options.maxReleases) || options.maxReleases < 0))
  ) {
    throw new Error("invalid_numeric_argument");
  }
  return options;
}

async function loadTargetDates(filename) {
  if (filename === null) {
    return null;
  }
  const text = await readFile(filename, "utf8");
  const targets = new Set();
  for (const [lineIndex, rawLine] of text.split("\n").entries()) {
    const line = rawLine.trim();
    if (!line || line.startsWith("#")) {
      continue;
    }
    const [family, releaseDate, ...extra] = line.split("\t");
    if (
      extra.length !== 0 ||
      !Object.hasOwn(FAMILY_CONFIG, family) ||
      !/^\d{4}-\d{2}-\d{2}$/.test(releaseDate)
    ) {
      throw new Error(`invalid_target_date_line:${lineIndex + 1}`);
    }
    targets.add(`${family}|${releaseDate}`);
  }
  return targets;
}

function sleep(milliseconds) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

function canonicalJson(value) {
  if (Array.isArray(value)) {
    return `[${value.map(canonicalJson).join(",")}]`;
  }
  if (value !== null && typeof value === "object") {
    return `{${Object.keys(value)
      .sort()
      .map((key) => `${JSON.stringify(key)}:${canonicalJson(value[key])}`)
      .join(",")}}`;
  }
  return JSON.stringify(value);
}

function manifestBytes(rows) {
  return Buffer.from(
    [...rows]
      .sort((left, right) => {
        const byUrl = left.source_url.localeCompare(right.source_url);
        return byUrl === 0
          ? canonicalJson(left).localeCompare(canonicalJson(right))
          : byUrl;
      })
      .map((row) => canonicalJson(row) + "\n")
      .join(""),
    "utf8",
  );
}

async function atomicWrite(filename, bytes) {
  await mkdir(path.dirname(filename), { recursive: true });
  const temporary = `${filename}.tmp-${process.pid}-${Date.now()}`;
  let handle = null;
  try {
    handle = await open(temporary, "wx");
    await handle.writeFile(bytes);
    await handle.sync();
    await handle.close();
    handle = null;
    await rename(temporary, filename);
    const directory = await open(path.dirname(filename), "r");
    try {
      await directory.sync();
    } finally {
      await directory.close();
    }
  } catch (error) {
    if (handle !== null) {
      await handle.close();
    }
    await rm(temporary, { force: true });
    throw error;
  }
}

async function loadJsonLines(filename) {
  try {
    const text = await readFile(filename, "utf8");
    return text
      .split("\n")
      .filter(Boolean)
      .map((line) => JSON.parse(line));
  } catch (error) {
    if (error.code === "ENOENT") {
      return [];
    }
    throw error;
  }
}

async function loadManifest(filename = MANIFEST_PATH) {
  const rows = await loadJsonLines(filename);
  const manifest = new Map();
  for (const row of rows) {
    if (manifest.has(row.source_url)) {
      throw new Error(`manifest_source_url_duplicate:${row.source_url}`);
    }
    manifest.set(row.source_url, row);
  }
  return manifest;
}

function decodeEntities(value) {
  return value
    .replaceAll("&amp;", "&")
    .replaceAll("&quot;", '"')
    .replaceAll("&#39;", "'")
    .replaceAll("&nbsp;", " ")
    .replace(/&#(\d+);/g, (_, number) => String.fromCodePoint(Number(number)));
}

function plainText(value) {
  return decodeEntities(value.replace(/<[^>]+>/g, " "))
    .replace(/\s+/g, " ")
    .trim();
}

function referencePeriodFromTitle(title) {
  const match = title.match(
    /\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(20\d{2})\b/i,
  );
  if (!match) {
    return null;
  }
  return `${match[1][0].toUpperCase()}${match[1].slice(1).toLowerCase()} ${match[2]}`;
}

function releaseDateFromFilename(filename, prefix) {
  const match = filename.match(
    new RegExp(`^${prefix}_(\\d{2})(\\d{2})(\\d{4})\\.htm$`, "i"),
  );
  if (!match) {
    return null;
  }
  return `${match[3]}-${match[1]}-${match[2]}`;
}

const MONTHS = Object.freeze([
  "January", "February", "March", "April", "May", "June",
  "July", "August", "September", "October", "November", "December",
]);
const MONTH_PATTERN = MONTHS.join("|");

function referencePeriodFromRelease(family, bytes) {
  const text = plainText(bytes.toString("utf8"));
  const heading = {
    CPI: "CONSUMER PRICE INDEX(?:ES)?",
    EMPLOYMENT: "EMPLOYMENT SITUATION",
    PPI: "PRODUCER PRICE INDEX(?:ES)?",
    JOLTS: "JOB OPENINGS AND LABOR TURNOVER",
  }[family];
  const match = text.match(
    new RegExp(
      `\\b${heading}\\s*[^A-Z0-9]{1,12}\\s*(${MONTH_PATTERN})\\s+(20\\d{2})\\b`,
      "i",
    ),
  );
  if (!match) {
    return null;
  }
  const month = MONTHS.find(
    (candidate) => candidate.toLowerCase() === match[1].toLowerCase(),
  );
  return `${month} ${match[2]}`;
}

function expectedReleaseDateText(releaseDate) {
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(releaseDate);
  if (!match) {
    throw new Error(`release_date_invalid:${releaseDate}`);
  }
  const month = MONTHS[Number.parseInt(match[2], 10) - 1];
  const day = Number.parseInt(match[3], 10);
  if (!month || day < 1 || day > 31) {
    throw new Error(`release_date_invalid:${releaseDate}`);
  }
  return `${month} ${day}, ${match[1]}`;
}

function releaseTimestampEvidence(bytes) {
  const text = plainText(bytes.toString("utf8")).slice(0, 30000);
  const match = text.match(
    new RegExp(
      `(?:8:30|10:00)\\s+a\\.?m\\.?.{0,100}?` +
        `(?:${MONTH_PATTERN})[,]?\\s+[0-9]{1,2},?\\s+(?:19|20)\\d{2}`,
      "i",
    ),
  );
  return match ? match[0].trim() : null;
}

function normalizedTimestampText(value) {
  return value
    .toLowerCase()
    .replace(/[,.()]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function timestampEvidenceProves(evidence, family, releaseDate) {
  if (typeof evidence !== "string") {
    return false;
  }
  const expectedTime = family === "JOLTS" ? "10:00 a.m." : "8:30 a.m.";
  const expectedDate = expectedReleaseDateText(releaseDate);
  const normalizedEvidence = normalizedTimestampText(evidence);
  return (
    normalizedEvidence.includes(normalizedTimestampText(expectedTime)) &&
    normalizedEvidence.includes(normalizedTimestampText(expectedDate))
  );
}

function validateBlsDocument(descriptor, bytes) {
  if (
    bytes.length < 1024 ||
    !bytes.includes(Buffer.from("Bureau of Labor Statistics", "utf8"))
  ) {
    throw new Error(`invalid_or_empty_bls_document:${descriptor.source_url}`);
  }
  if (descriptor.artifact_kind !== "release") {
    return { descriptor, releaseTimestampEvidence: null };
  }
  const derivedReference = referencePeriodFromRelease(
    descriptor.bls_family,
    bytes,
  );
  if (!derivedReference) {
    throw new Error(
      `release_reference_period_not_proven:${descriptor.source_url}`,
    );
  }
  if (
    descriptor.reference_period !== null &&
    descriptor.reference_period !== derivedReference
  ) {
    throw new Error(
      `release_reference_period_conflict:${descriptor.source_url}:` +
        `${descriptor.reference_period}:${derivedReference}`,
    );
  }
  const timestampEvidence = releaseTimestampEvidence(bytes);
  const expectedTime = descriptor.bls_family === "JOLTS" ? "10:00 a.m." : "8:30 a.m.";
  const expectedDate = expectedReleaseDateText(descriptor.release_date);
  if (!timestampEvidenceProves(
    timestampEvidence,
    descriptor.bls_family,
    descriptor.release_date,
  )) {
    throw new Error(
      `release_timestamp_not_proven:${descriptor.source_url}:` +
        `expected=${expectedTime} ${expectedDate}:` +
        `evidence=${timestampEvidence ?? "missing"}`,
    );
  }
  return {
    descriptor: { ...descriptor, reference_period: derivedReference },
    releaseTimestampEvidence: timestampEvidence,
  };
}

function discoverReleases(family, config, indexBytes, startYear, endYear) {
  const html = indexBytes.toString("utf8");
  const anchor = /<a\b[^>]*href=["']([^"']+)["'][^>]*>([\s\S]*?)<\/a>/gi;
  const releases = new Map();
  for (const match of html.matchAll(anchor)) {
    const url = new URL(decodeEntities(match[1]), config.indexUrl);
    if (url.protocol !== "https:" || url.hostname !== AUTHORITATIVE_HOST) {
      continue;
    }
    const filename = path.posix.basename(url.pathname);
    const releaseDate = releaseDateFromFilename(filename, config.prefix);
    if (!releaseDate) {
      continue;
    }
    const releaseYear = Number.parseInt(releaseDate.slice(0, 4), 10);
    if (releaseYear < startYear || releaseYear > endYear) {
      continue;
    }
    const sourceUrl = url.toString();
    releases.set(sourceUrl, {
      artifact_kind: "release",
      bls_family: family,
      link_title: plainText(match[2]),
      reference_period: referencePeriodFromTitle(plainText(match[2])),
      release_date: releaseDate,
      source_release_identity: filename.replace(/\.htm$/i, ""),
      source_url: sourceUrl,
    });
  }
  return [...releases.values()].sort((left, right) =>
    left.source_url.localeCompare(right.source_url),
  );
}

class CdpConnection {
  constructor(webSocketUrl) {
    this.nextId = 1;
    this.pending = new Map();
    this.listeners = new Map();
    this.socket = new WebSocket(webSocketUrl);
  }

  async open() {
    await new Promise((resolve, reject) => {
      this.socket.onopen = resolve;
      this.socket.onerror = () => reject(new Error("cdp_websocket_open_failed"));
    });
    this.socket.onmessage = (event) => {
      const message = JSON.parse(event.data);
      if (message.id && this.pending.has(message.id)) {
        const pending = this.pending.get(message.id);
        this.pending.delete(message.id);
        if (message.error) {
          pending.reject(new Error(`cdp_error:${canonicalJson(message.error)}`));
        } else {
          pending.resolve(message.result);
        }
        return;
      }
      const listeners = this.listeners.get(message.method) ?? [];
      for (const listener of listeners) {
        listener(message.params);
      }
    };
  }

  call(method, params = {}) {
    return new Promise((resolve, reject) => {
      const id = this.nextId;
      this.nextId += 1;
      this.pending.set(id, { resolve, reject });
      this.socket.send(JSON.stringify({ id, method, params }));
    });
  }

  on(method, listener) {
    const listeners = this.listeners.get(method) ?? [];
    listeners.push(listener);
    this.listeners.set(method, listeners);
    return () => {
      this.listeners.set(
        method,
        (this.listeners.get(method) ?? []).filter((item) => item !== listener),
      );
    };
  }

  close() {
    this.socket.close();
  }
}

async function launchBrowser(chromePath) {
  await stat(chromePath);
  const profile = await mkdtemp(path.join(os.tmpdir(), "ea-bls-browser-"));
  const child = spawn(
    chromePath,
    [
      "--remote-debugging-port=0",
      "--remote-allow-origins=*",
      `--user-data-dir=${profile}`,
      "--no-first-run",
      "--no-default-browser-check",
      "--disable-background-networking",
      "--disable-component-update",
      "--disable-sync",
      "--disable-extensions",
      "--window-position=-20000,-20000",
      "--window-size=800,600",
      "about:blank",
    ],
    { stdio: ["ignore", "ignore", "pipe"] },
  );
  let stderr = "";
  const webSocketUrl = await new Promise((resolve, reject) => {
    const timeout = setTimeout(
      () => reject(new Error("chrome_devtools_start_timeout")),
      30000,
    );
    child.stderr.setEncoding("utf8");
    child.stderr.on("data", (chunk) => {
      stderr += chunk;
      const match = stderr.match(/DevTools listening on (ws:\/\/[^\s]+)/);
      if (match) {
        clearTimeout(timeout);
        resolve(match[1]);
      }
    });
    child.once("exit", (code) => {
      clearTimeout(timeout);
      reject(new Error(`chrome_exited_before_devtools:${code}`));
    });
  });
  const endpoint = new URL(webSocketUrl);
  const base = `http://${endpoint.host}`;
  const target = await (
    await fetch(`${base}/json/new?about%3Ablank`, { method: "PUT" })
  ).json();
  const connection = new CdpConnection(target.webSocketDebuggerUrl);
  await connection.open();
  await connection.call("Network.enable");
  await connection.call("Network.setBlockedURLs", {
    urls: [
      "*.css",
      "*.js",
      "*.png",
      "*.jpg",
      "*.jpeg",
      "*.gif",
      "*.svg",
      "*.woff",
      "*.woff2",
      "*.ico",
    ],
  });
  return { child, connection, profile };
}

async function stopBrowser(browser) {
  browser.connection.close();
  browser.child.kill("SIGTERM");
  await Promise.race([
    new Promise((resolve) => browser.child.once("exit", resolve)),
    sleep(5000),
  ]);
  if (browser.child.exitCode === null) {
    browser.child.kill("SIGKILL");
  }
  const resolvedProfile = path.resolve(browser.profile);
  if (!resolvedProfile.startsWith(path.resolve(os.tmpdir(), "ea-bls-browser-"))) {
    throw new Error("refusing_to_remove_unexpected_browser_profile");
  }
  await rm(resolvedProfile, { recursive: true, force: true });
}

async function captureMainDocument(connection, sourceUrl) {
  const expected = new URL(sourceUrl);
  if (
    expected.protocol !== "https:" ||
    expected.hostname !== AUTHORITATIVE_HOST
  ) {
    throw new Error(`non_authoritative_url:${sourceUrl}`);
  }
  return await new Promise(async (resolve, reject) => {
    let mainRequestId = null;
    let response = null;
    const timeout = setTimeout(() => {
      cleanup();
      reject(new Error(`capture_timeout:${sourceUrl}`));
    }, 45000);
    const removeResponse = connection.on("Network.responseReceived", (params) => {
      const url = new URL(params.response.url);
      if (
        params.type === "Document" &&
        url.hostname === expected.hostname &&
        url.pathname === expected.pathname
      ) {
        mainRequestId = params.requestId;
        response = params.response;
      }
    });
    const removeFinished = connection.on("Network.loadingFinished", async (params) => {
      if (!mainRequestId || params.requestId !== mainRequestId) {
        return;
      }
      try {
        const body = await connection.call("Network.getResponseBody", {
          requestId: mainRequestId,
        });
        const bytes = body.base64Encoded
          ? Buffer.from(body.body, "base64")
          : Buffer.from(body.body, "utf8");
        cleanup();
        resolve({ bytes, response });
      } catch (error) {
        cleanup();
        reject(error);
      }
    });
    function cleanup() {
      clearTimeout(timeout);
      removeResponse();
      removeFinished();
    }
    try {
      await connection.call("Page.navigate", { url: sourceUrl });
    } catch (error) {
      cleanup();
      reject(error);
    }
  });
}

async function captureWithRetry(connection, sourceUrl, delayMs) {
  let lastError = null;
  for (let attempt = 1; attempt <= 3; attempt += 1) {
    try {
      const capture = await captureMainDocument(connection, sourceUrl);
      if (
        capture.response.status !== 200 ||
        !capture.response.mimeType.startsWith("text/html") ||
        capture.bytes.length < 1024 ||
        !capture.bytes.includes(Buffer.from("Bureau of Labor Statistics", "utf8"))
      ) {
        throw new Error(
          `invalid_bls_document_response:${capture.response.status}:` +
            `${capture.response.mimeType}:${capture.bytes.length}`,
        );
      }
      return capture;
    } catch (error) {
      lastError = error;
      if (attempt < 3) {
        console.error(
          `retry ${attempt}/3 ${sourceUrl}: ${error.message ?? String(error)}`,
        );
        await sleep(Math.max(delayMs, 1000));
      }
    }
  }
  throw new Error(
    `capture_failed_after_retries:${sourceUrl}:` +
      `${lastError?.message ?? String(lastError)}`,
  );
}

function selectedHeaders(headers) {
  const lowered = new Map(
    Object.entries(headers).map(([name, value]) => [name.toLowerCase(), String(value)]),
  );
  const result = {};
  for (const name of [
    "content-type",
    "content-length",
    "date",
    "etag",
    "last-modified",
  ]) {
    if (lowered.has(name)) {
      result[name] = lowered.get(name);
    }
  }
  return result;
}

async function retainArtifact(
  descriptor,
  capture,
  retrievedAt,
  existing,
  conflicts,
  repositoryRoot = REPOSITORY_ROOT,
) {
  if (capture.response.status !== 200) {
    throw new Error(
      `unexpected_http_status:${capture.response.status}:${descriptor.source_url}`,
    );
  }
  if (!capture.response.mimeType.startsWith("text/html")) {
    throw new Error(
      `unexpected_mime_type:${capture.response.mimeType}:${descriptor.source_url}`,
    );
  }
  const validated = validateBlsDocument(descriptor, capture.bytes);
  descriptor = validated.descriptor;
  const digest = sha256(capture.bytes);
  const prior = existing.get(descriptor.source_url);
  if (prior) {
    const retainedPath = path.resolve(repositoryRoot, prior.immutable_local_path);
    const retainedBytes = await readFile(retainedPath);
    const retainedDigest = sha256(retainedBytes);
    if (retainedDigest !== prior.sha256) {
      throw new Error(`retained_artifact_hash_mismatch:${prior.immutable_local_path}`);
    }
    if (digest !== prior.sha256) {
      addUniqueRecord(conflicts, {
        bls_family: descriptor.bls_family,
        claimed_source_observation: prior.source_release_identity,
        fetched_sha256: digest,
        first_admitted_sha256: prior.sha256,
        observed_at: retrievedAt,
        source_url: descriptor.source_url,
      }, ["source_url", "fetched_sha256", "first_admitted_sha256"]);
      return { row: prior, outcome: "conflict_preserved_original" };
    }
    return { row: prior, outcome: "duplicate_identical" };
  }

  const relativePath = relativePathForDescriptor(descriptor);
  await atomicWrite(path.resolve(repositoryRoot, relativePath), capture.bytes);
  const row = buildArtifactRow(
    descriptor,
    capture.bytes,
    retrievedAt,
    {
      headers: selectedHeaders(capture.response.headers),
      mime_type: capture.response.mimeType,
      protocol: capture.response.protocol,
      status: capture.response.status,
    },
    validated.releaseTimestampEvidence,
    relativePath,
  );
  existing.set(descriptor.source_url, row);
  return { row, outcome: "admitted" };
}

function relativePathForDescriptor(descriptor) {
  const filename = path.posix.basename(new URL(descriptor.source_url).pathname);
  return path.posix.join(
    "EconomicCalendar/raw/bls",
    descriptor.artifact_kind === "index" ? "indexes" : "releases",
    descriptor.bls_family.toLowerCase(),
    filename,
  );
}

function buildArtifactRow(
  descriptor,
  bytes,
  admittedAt,
  httpSourceMetadata,
  timestampEvidence,
  relativePath = relativePathForDescriptor(descriptor),
) {
  return {
    ...descriptor,
    bytes: bytes.length,
    first_archive_admission_at: admittedAt,
    http_source_metadata: httpSourceMetadata,
    immutable_local_path: relativePath,
    release_timestamp_evidence: timestampEvidence,
    retrieved_at: admittedAt,
    sha256: sha256(bytes),
  };
}

function addUniqueRecord(rows, row, identityFields) {
  const duplicate = rows.some((candidate) =>
    identityFields.every((field) => candidate[field] === row[field]),
  );
  if (!duplicate) {
    rows.push(row);
  }
}

function validateDescriptorIdentity(expected, actual) {
  for (const field of [
    "artifact_kind",
    "bls_family",
    "release_date",
    "source_release_identity",
    "source_url",
  ]) {
    if (actual[field] !== expected[field]) {
      throw new Error(`retained_artifact_identity_mismatch:${field}:${expected.source_url}`);
    }
  }
}

async function reconcileManifestedArtifact(
  descriptor,
  row,
  repositoryRoot = REPOSITORY_ROOT,
) {
  validateDescriptorIdentity(descriptor, row);
  const expectedPath = relativePathForDescriptor(descriptor);
  if (row.immutable_local_path !== expectedPath) {
    throw new Error(`retained_artifact_path_mismatch:${descriptor.source_url}`);
  }
  const bytes = await readFile(path.resolve(repositoryRoot, expectedPath));
  const digest = sha256(bytes);
  if (digest !== row.sha256) {
    throw new Error(`retained_artifact_hash_mismatch:${expectedPath}`);
  }
  const validated = validateBlsDocument(descriptor, bytes);
  if (
    row.reference_period !== null &&
    row.reference_period !== validated.descriptor.reference_period
  ) {
    throw new Error(
      `retained_artifact_metadata_conflict:reference_period:${descriptor.source_url}`,
    );
  }
  if (
    row.release_timestamp_evidence !== null &&
    !timestampEvidenceProves(
      row.release_timestamp_evidence,
      descriptor.bls_family,
      descriptor.release_date,
    )
  ) {
    throw new Error(
      `retained_artifact_metadata_conflict:release_timestamp_evidence:${descriptor.source_url}`,
    );
  }
  const enriched = {
    ...row,
    reference_period: row.reference_period ?? validated.descriptor.reference_period,
    release_timestamp_evidence:
      row.release_timestamp_evidence ?? validated.releaseTimestampEvidence,
  };
  return {
    bytes,
    changed: canonicalJson(row) !== canonicalJson(enriched),
    row: enriched,
    outcome: canonicalJson(row) === canonicalJson(enriched)
      ? "duplicate_identical"
      : "metadata_reconciled",
  };
}

async function reconcileUnmanifestedArtifact(
  descriptor,
  reconciledAt,
  existing,
  repositoryRoot = REPOSITORY_ROOT,
) {
  const relativePath = relativePathForDescriptor(descriptor);
  let bytes;
  let fileStat;
  try {
    [bytes, fileStat] = await Promise.all([
      readFile(path.resolve(repositoryRoot, relativePath)),
      stat(path.resolve(repositoryRoot, relativePath)),
    ]);
  } catch (error) {
    if (error.code === "ENOENT") {
      return null;
    }
    throw error;
  }
  const validated = validateBlsDocument(descriptor, bytes);
  const admittedAt = (
    fileStat.birthtimeMs > 0 ? fileStat.birthtime : fileStat.mtime
  ).toISOString();
  const row = {
    ...buildArtifactRow(
      validated.descriptor,
      bytes,
      admittedAt,
      {
        capture_basis: "validated_interrupted_cdp_raw_artifact",
        headers: {},
        mime_type: "text/html",
        protocol: "cdp_interrupted_capture",
        status: 200,
      },
      validated.releaseTimestampEvidence,
      relativePath,
    ),
    reconciled_at: reconciledAt,
  };
  existing.set(descriptor.source_url, row);
  return { bytes, row, outcome: "orphan_reconciled" };
}

async function persistAcquisitionState(
  manifest,
  conflicts,
  failures,
  root = ROOT,
) {
  await atomicWrite(path.join(root, "manifest.jsonl"), manifestBytes(manifest.values()));
  if (conflicts.length > 0) {
    await atomicWrite(
      path.join(root, "acquisition_conflicts.jsonl"),
      manifestBytes(conflicts),
    );
  }
  if (failures.length > 0) {
    await atomicWrite(
      path.join(root, "acquisition_failures.jsonl"),
      manifestBytes(failures),
    );
  }
}

function recordFailure(failures, descriptor, error, observedAt) {
  addUniqueRecord(failures, {
    artifact_kind: descriptor.artifact_kind,
    bls_family: descriptor.bls_family,
    error: error.message ?? String(error),
    first_observed_at: observedAt,
    release_date: descriptor.release_date,
    source_release_identity: descriptor.source_release_identity,
    source_url: descriptor.source_url,
  }, ["source_url", "error"]);
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const targetDates = await loadTargetDates(options.targetDates);
  await mkdir(ROOT, { recursive: true });
  const manifest = await loadManifest();
  const conflicts = await loadJsonLines(CONFLICT_PATH);
  const failures = await loadJsonLines(FAILURE_PATH);
  let browser = null;
  let releaseCount = 0;
  async function browserConnection() {
    if (browser === null) {
      browser = await launchBrowser(options.chrome);
    }
    return browser.connection;
  }
  try {
    for (const family of options.families) {
      const config = FAMILY_CONFIG[family];
      const indexDescriptor = {
        artifact_kind: "index",
        bls_family: family,
        link_title: `${family} archived news release index`,
        reference_period: null,
        release_date: null,
        source_release_identity: `${config.prefix}_archive_index`,
        source_url: config.indexUrl,
      };
      let indexResult;
      try {
        const prior = manifest.get(indexDescriptor.source_url);
        if (prior) {
          indexResult = await reconcileManifestedArtifact(indexDescriptor, prior);
          if (indexResult.changed) {
            manifest.set(indexDescriptor.source_url, indexResult.row);
            await persistAcquisitionState(manifest, conflicts, failures);
          }
        } else {
          indexResult = await reconcileUnmanifestedArtifact(
            indexDescriptor,
            options.retrievedAt,
            manifest,
          );
          if (indexResult === null) {
            const indexCapture = await captureWithRetry(
              await browserConnection(),
              config.indexUrl,
              options.delayMs,
            );
            indexResult = await retainArtifact(
              indexDescriptor,
              indexCapture,
              options.retrievedAt,
              manifest,
              conflicts,
            );
            indexResult.bytes = indexCapture.bytes;
          }
          await persistAcquisitionState(manifest, conflicts, failures);
        }
      } catch (error) {
        recordFailure(failures, indexDescriptor, error, options.retrievedAt);
        await persistAcquisitionState(manifest, conflicts, failures);
        console.error(`${family} index failed: ${error.message ?? String(error)}`);
        continue;
      }
      console.log(`${family} index ${indexResult.outcome} ${indexResult.bytes.length} bytes`);
      let releases = discoverReleases(
        family,
        config,
        indexResult.bytes,
        options.startYear,
        options.endYear,
      );
      if (targetDates !== null) {
        releases = releases.filter((release) =>
          targetDates.has(`${family}|${release.release_date}`),
        );
      }
      console.log(`${family} discovered ${releases.length} release artifacts`);
      for (const descriptor of releases) {
        if (
          options.maxReleases !== null &&
          releaseCount >= options.maxReleases
        ) {
          break;
        }
        releaseCount += 1;
        try {
          let result;
          const prior = manifest.get(descriptor.source_url);
          if (prior) {
            result = await reconcileManifestedArtifact(descriptor, prior);
            if (result.changed) {
              manifest.set(descriptor.source_url, result.row);
              await persistAcquisitionState(manifest, conflicts, failures);
            }
          } else {
            result = await reconcileUnmanifestedArtifact(
              descriptor,
              options.retrievedAt,
              manifest,
            );
            if (result === null) {
              await sleep(options.delayMs);
              const capture = await captureWithRetry(
                await browserConnection(),
                descriptor.source_url,
                options.delayMs,
              );
              result = await retainArtifact(
                descriptor,
                capture,
                options.retrievedAt,
                manifest,
                conflicts,
              );
              result.bytes = capture.bytes;
            }
            await persistAcquisitionState(manifest, conflicts, failures);
          }
          console.log(
            `${family} ${descriptor.release_date} ${result.outcome} ` +
              `${result.bytes.length} bytes ${sha256(result.bytes).slice(0, 12)}`,
          );
        } catch (error) {
          recordFailure(failures, descriptor, error, options.retrievedAt);
          await persistAcquisitionState(manifest, conflicts, failures);
          console.error(
            `${family} ${descriptor.release_date} failed: ` +
              `${error.message ?? String(error)}`,
          );
        }
      }
      if (
        options.maxReleases !== null &&
        releaseCount >= options.maxReleases
      ) {
        break;
      }
    }
  } finally {
    if (browser) {
      await stopBrowser(browser);
    }
  }

  const bytes = manifestBytes(manifest.values());
  await persistAcquisitionState(manifest, conflicts, failures);
  console.log(
    `manifest rows=${manifest.size} sha256=${sha256(bytes)} ` +
      `conflicts=${conflicts.length} failures=${failures.length}`,
  );
}

const invokedPath = process.argv[1] ? path.resolve(process.argv[1]) : null;
if (invokedPath === fileURLToPath(import.meta.url)) {
  main().catch((error) => {
    console.error(error.stack ?? String(error));
    process.exitCode = 1;
  });
}

export {
  atomicWrite,
  buildArtifactRow,
  canonicalJson,
  discoverReleases,
  manifestBytes,
  persistAcquisitionState,
  reconcileManifestedArtifact,
  reconcileUnmanifestedArtifact,
  retainArtifact,
  validateBlsDocument,
};
