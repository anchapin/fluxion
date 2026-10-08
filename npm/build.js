#!/usr/bin/env node

/**
 * Build script for Fluxion Node.js native bindings
 *
 * This script handles building the native module across different platforms
 * and architectures using the napi-rs CLI.
 */

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

const platform = process.platform;
const arch = process.arch;

// Resolve the napi CLI from this package's own node_modules (Issue #4337).
// Bare `napi` is only on PATH when the script is started through `npm run`;
// a direct `node build.js` must use the installed binary path. Windows uses
// a .cmd shim in node_modules/.bin.
const napiBin = path.join(
  __dirname,
  'node_modules',
  '.bin',
  platform === 'win32' ? 'napi.cmd' : 'napi',
);

console.log(`Building Fluxion native bindings for ${platform}-${arch}...`);

try {
  // Ensure napi-rs CLI is installed (Issue #4337). Prefer the exact-pinned
  // CLI that `npm ci`/`npm install` placed in this package's
  // node_modules/.bin; fall back to a PATH `napi` only if that exists; only
  // then install, with --save-exact so the "3.10.5" pin in package.json /
  // package-lock.json stays intact (Issue #4202).
  console.log('Checking for @napi-rs/cli...');
  let napiCmd;
  if (fs.existsSync(napiBin)) {
    console.log(`Using local napi CLI at ${napiBin}...`);
    napiCmd = `"${napiBin}"`;
  } else {
    try {
      execSync('napi --version', { stdio: 'inherit' });
      console.log('Using napi CLI from PATH...');
      napiCmd = 'napi';
    } catch (error) {
      // Deterministic fallback: pin to the exact version in package.json's
      // devDependencies (Issue #4202). `npm ci` in CI always installs the
      // CLI from the lockfile, so this path only fires for local builds
      // with a missing node_modules -- and even then it must not float.
      // --save-exact keeps the "3.10.5" pin intact instead of rewriting it
      // to "^3.10.5" (Issue #4337).
      console.log('Installing @napi-rs/cli...');
      execSync('npm install --save-exact @napi-rs/cli@3.10.5', { stdio: 'inherit' });
      if (!fs.existsSync(napiBin)) {
        throw new Error(`@napi-rs/cli installed but ${napiBin} is missing`);
      }
      napiCmd = `"${napiBin}"`;
    }
  }

  // Build the native module
  console.log('Building native module with napi-rs...');
  const buildArgs = [
    'build',
    '--manifest-path', '../Cargo.toml',
    '--package-json-path', 'package.json',
    '--output-dir', '.',
    '--features', 'napi-bindings',
    '--dts', 'index.d.ts',
  ];

  if (process.argv.includes('--release') || process.env.NODE_ENV === 'production') {
    buildArgs.push('--release');
  }

  execSync(`${napiCmd} ${buildArgs.join(' ')}`, {
    stdio: 'inherit',
    env: {
      ...process.env,
      RUST_MIN_STACK: process.env.RUST_MIN_STACK || '16777216',
    },
  });

  // Verify the build output
  const nativeModulePath = path.join(__dirname, 'fluxion.node');
  if (!fs.existsSync(nativeModulePath)) {
    throw new Error(`Native module not found at ${nativeModulePath}`);
  }

  console.log('✓ Build completed successfully!');
  console.log(`✓ Native module: ${nativeModulePath}`);

} catch (error) {
  console.error('✗ Build failed:', error.message);
  process.exit(1);
}
