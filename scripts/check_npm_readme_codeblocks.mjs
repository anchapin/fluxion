#!/usr/bin/env node
//
// check_npm_readme_codeblocks.mjs — drift gate for npm/README.md fenced code blocks.
//
// Issue #2511: fragmented/broken code fences in npm/README.md (Quick Start,
// TypeScript Usage, Error Handling) had silently merged. This script extracts
// every fenced code block from npm/README.md and validates it so future drift
// fails CI instead of slipping through review.
//
// Two tiers of validation run on every JS/TS block:
//
//   1. STRUCTURAL (always on, zero-dependency, gates on ALL Node versions)
//      A state-machine scanner tracks (), [], {} balance while respecting
//      single/double/backtick strings, // and /* */ comments, and ${...}
//      template interpolation. It also verifies each fence is actually closed.
//      This reliably catches the exact #2511 failure mode — truncated bodies,
//      unclosed functions, dangling delimiters, unclosed fences — without any
//      parser dependency and with zero false positives.
//
//   2. PARSE (best-effort via the built-in V8 parser, `node --check`)
//      - JS blocks (javascript/js): the block is written to a temp .cjs
//        (CommonJS) or .mjs (ESM, when import/export appears) file and
//        `node --check` validates it. This is a real V8 syntax check and is
//        GATING. It catches non-structural errors (typos, bad syntax) that the
//        structural tier cannot.
//      - TS blocks (typescript/ts): `node --check` type-stripping for .ts is
//        inconsistent in --check mode (it silently accepts some broken code),
//        so the TS parse is reported ADVISORY only and never gates. TS blocks
//        are gated by the structural tier (the issue's accepted minimum:
//        "brace/paren balance check and a basic syntax sanity scan"). Full TS
//        type-checking is the job of `tsc`, not a docs gate.
//
// A block fails the gate iff its structural check fails, or (for JS blocks) its
// parse check runs and fails. The exit code is non-zero if any block fails.
//
// Usage:
//   node scripts/check_npm_readme_codeblocks.mjs           # check npm/README.md
//   node scripts/check_npm_readme_codeblocks.mjs <path>     # check another file
//
// Wired into CI via .github/workflows/docs-hygiene.yml and runnable locally
// with `npm run test:readme` from the npm/ directory. No npm dependencies,
// no built wheel required.

import { readFileSync, writeFileSync, unlinkSync, mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, dirname, resolve } from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = join(__dirname, '..');
const DEFAULT_README = join(REPO_ROOT, 'npm', 'README.md');

// Languages we validate. `bash`, `text`, etc. are skipped.
const JS_LANGS = new Set(['js', 'javascript', 'node']);
const TS_LANGS = new Set(['ts', 'typescript']);
const CHECKED_LANGS = new Set([...JS_LANGS, ...TS_LANGS]);

// Node version — type stripping for .ts is auto-enabled at 22.18 / 23+.
function parseNodeVersion(v) {
  const m = /v(\d+)\.(\d+)\.(\d+)/.exec(v || process.version);
  return m ? { major: +m[1], minor: +m[2], patch: +m[3] } : { major: 0, minor: 0, patch: 0 };
}
function supportsTsStrip(info) {
  return info.major > 22 || (info.major === 22 && info.minor >= 18);
}

// --- Fence extraction (CommonMark-ish) --------------------------------------
//
// A fence opens with a line whose content is 3+ backticks optionally followed
// by an info string (the language). It closes with a line of 3+ backticks and
// nothing else. We scan line by line so the language tag and body are exact.

function extractBlocks(md) {
  const lines = md.split('\n');
  const blocks = [];
  let i = 0;
  while (i < lines.length) {
    const open = /^(\s*)(`{3,})(.*)$/.exec(lines[i]);
    if (!open) { i++; continue; }
    const indent = open[1];
    const fenceLen = open[2].length;
    const lang = open[3].trim().split(/\s+/)[0] || '';
    const startLine = i + 1; // 1-indexed
    const body = [];
    let endLine = null;
    let j = i + 1;
    while (j < lines.length) {
      const close = /^(\s*)(`{3,})\s*$/.exec(lines[j]);
      if (close && close[1] === indent && close[2].length >= fenceLen) {
        endLine = j + 1;
        break;
      }
      body.push(lines[j]);
      j++;
    }
    blocks.push({
      index: blocks.length + 1,
      lang,
      code: body.join('\n'),
      startLine,
      endLine,
      closed: endLine !== null,
    });
    i = endLine !== null ? endLine : j;
  }
  return blocks;
}

// --- Tier 1: structural balance ---------------------------------------------
//
// Track (), [], {} depth while skipping over string/comment/template regions
// so a delimiter inside a literal never counts. Template literals are scanned
// with ${...} interpolation awareness (interpolation braces are pushed as
// synthetic frames and auto-popped, so they neither falsely balance an outer
// brace nor mask a real imbalance). Returns { ok, error }.

function structuralCheck(code) {
  const pairs = { ')': '(', ']': '[', '}': '{' };
  const opening = { '(': true, '[': true, '{': true };
  const stack = []; // { ch, line, interp? }
  let state = 'code'; // code | sq | dq | bt | lineComment | blockComment
  let strCh = null;
  for (let i = 0; i < code.length; i++) {
    const c = code[i];
    const n = code[i + 1];
    const line = lineOf(code, i);
    switch (state) {
      case 'sq':
      case 'dq':
      case 'bt':
        if (c === '\\') { i++; break; }                   // escaped next char
        if (state === 'bt' && c === '$' && n === '{') {   // ${ ... } interpolation
          stack.push({ ch: '{', line, interp: true });
          i++;                                            // consume the '{'
          break;
        }
        if (c === strCh) state = 'code';
        break;
      case 'lineComment':
        if (c === '\n') state = 'code';
        break;
      case 'blockComment':
        if (c === '*' && n === '/') { state = 'code'; i++; }
        break;
      default: // code
        if (c === '/' && n === '/') { state = 'lineComment'; i++; break; }
        if (c === '/' && n === '*') { state = 'blockComment'; i++; break; }
        if (c === '"' || c === "'" || c === '`') {
          state = c === '`' ? 'bt' : (c === '"' ? 'dq' : 'sq');
          strCh = c;
          break;
        }
        if (opening[c]) { stack.push({ ch: c, line }); break; }
        if (pairs[c]) {
          while (stack.length && stack[stack.length - 1].interp) stack.pop();
          if (!stack.length) return { ok: false, error: `unmatched closing "${c}" at line ${line}` };
          const top = stack.pop();
          if (top.ch !== pairs[c]) {
            return { ok: false, error: `"${top.ch}" (line ${top.line}) closed by "${c}" (line ${line})` };
          }
          break;
        }
    }
  }
  if (state === 'sq' || state === 'dq' || state === 'bt') {
    return { ok: false, error: `unterminated ${strCh === '`' ? 'template literal' : 'string literal'} (\`${strCh}\`)` };
  }
  if (state === 'blockComment') return { ok: false, error: 'unterminated block comment /*' };
  const real = stack.filter((s) => !s.interp);
  if (real.length) {
    const top = real[real.length - 1];
    return { ok: false, error: `unmatched opening "${top.ch}" at line ${top.line}` };
  }
  return { ok: true };
}

function lineOf(code, upto) {
  let n = 1;
  for (let i = 0; i < upto; i++) if (code[i] === '\n') n++;
  return n;
}

// --- Tier 2: parse via `node --check` ---------------------------------------

function isModuleSyntax(code) {
  return /(^|\n)\s*(import|export)\b/.test(code);
}

function newTempFile(ext) {
  const dir = mkdtempSync(join(tmpdir(), 'fluxion-readme-'));
  return { path: join(dir, `block${ext}`), dir };
}

function runNodeCheck(code, ext) {
  const { path, dir } = newTempFile(ext);
  try {
    writeFileSync(path, code);
    const res = spawnSync(process.execPath, ['--check', path], { encoding: 'utf8' });
    return { ok: res.status === 0, stderr: (res.stderr || '').trim() };
  } finally {
    try { unlinkSync(path); } catch {}
    try { unlinkSync(dir); } catch {}
  }
}

// JS blocks: GATING real V8 parse. ESM (import/export) -> .mjs, else .cjs.
function parseJs(block) {
  const ext = isModuleSyntax(block.code) ? '.mjs' : '.cjs';
  return { gating: true, ...runNodeCheck(block.code, ext) };
}

// TS blocks: ADVISORY-only. node --check type-strips .ts/.mts inconsistently in
// --check mode (it silently accepts some broken code), so a TS parse result can
// never be trusted to gate. Report it for information; the structural tier gates.
function parseTs(block, nodeInfo) {
  if (!supportsTsStrip(nodeInfo)) {
    return { gating: false, ran: false, skipped: true, reason: `Node ${nodeInfo.major}.${nodeInfo.minor} < 22.18` };
  }
  // Module-mode .mts enables type stripping on Node >= 22.18. If the snippet
  // lacks import/export, force module mode with `export {};`.
  const body = isModuleSyntax(block.code) ? block.code : block.code + '\nexport {};\n';
  return { gating: false, ...runNodeCheck(body, '.mts') };
}

// --- Driver -----------------------------------------------------------------

function fmtLoc(b) {
  return b.closed ? `lines ${b.startLine}-${b.endLine}` : `line ${b.startLine} (UNCLOSED fence)`;
}

function indent(s, n) {
  const pad = ' '.repeat(n);
  return s.split('\n').map((l) => pad + l).join('\n');
}

function main() {
  const target = process.argv[2] ? resolve(process.cwd(), process.argv[2]) : DEFAULT_README;
  const nodeInfo = parseNodeVersion();
  const md = readFileSync(target, 'utf8');
  const blocks = extractBlocks(md);

  const checks = blocks.filter((b) => CHECKED_LANGS.has(b.lang));
  const skippedLangs = blocks.filter((b) => !CHECKED_LANGS.has(b.lang));
  const failures = [];
  let passCount = 0;

  console.log(`\n npm/README.md code-block drift gate (#2511)`);
  console.log(` file: ${target}`);
  console.log(` node ${nodeInfo.major}.${nodeInfo.minor}.${nodeInfo.patch}`);
  console.log(` ${blocks.length} fenced block(s): ${checks.length} JS/TS checked, ${skippedLangs.length} non-code skipped.\n`);

  for (const b of checks) {
    const isTs = TS_LANGS.has(b.lang);
    const struct = b.closed ? structuralCheck(b.code) : { ok: false, error: 'fence never closed' };
    const parse = struct.ok
      ? (isTs ? parseTs(b, nodeInfo) : parseJs(b))
      : { gating: !isTs, ran: false, skipped: true, reason: 'structural failed' };

    // Gate = structural (always) AND parse (JS only). TS parse is advisory.
    const structOk = struct.ok;
    const parseGatingFailed = parse.gating && parse.ran !== false && !parse.ok;
    const ok = structOk && !parseGatingFailed;

    const parseLabel = parse.skipped
      ? `parse: skipped (${parse.reason})`
      : `parse: ${parse.ok ? 'ok' : 'FAILED'}${isTs ? ' [advisory]' : ''}`;
    console.log(`  [${ok ? 'PASS' : 'FAIL'}] #${b.index} ${(b.lang || '(none)').padEnd(11)} ${fmtLoc(b).padEnd(28)} structural: ${struct.ok ? 'ok' : 'BAD'}  ${parseLabel}`);
    if (!struct.ok) console.log(`          structural: ${struct.error}`);
    if (!parse.skipped && !parse.ok) console.log(`          parse:\n${indent(parse.stderr, 10)}`);

    if (ok) passCount++;
    else failures.push(b);
  }

  if (skippedLangs.length) {
    const langs = [...new Set(skippedLangs.map((b) => b.lang || 'plain'))].join(', ');
    console.log(`\n  (skipped ${skippedLangs.length} non-JS/TS block(s): ${langs})`);
  }
  console.log(`\n  TS parse results are advisory (node --check type-stripping is inconsistent);`);
  console.log(`  the structural tier gates all blocks and catches the #2511 truncation mode.`);

  console.log(`\n ${passCount}/${checks.length} JS/TS block(s) passed the gate.`);
  if (failures.length) {
    console.log(` \x1b[31mFAIL\x1b[0m — ${failures.length} block(s) broken. Fix the code fence(s) above.\n`);
    process.exit(1);
  }
  console.log(` \x1b[32mOK\x1b[0m — all code blocks well-formed.\n`);
}

try {
  main();
} catch (e) {
  console.error(`check_npm_readme_codeblocks: ${e.stack || e.message}`);
  process.exit(2);
}
