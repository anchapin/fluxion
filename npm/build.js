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

console.log(`Building Fluxion native bindings for ${platform}-${arch}...`);

try {
  // Ensure napi-rs CLI is installed
  console.log('Checking for @napi-rs/cli...');
  try {
    execSync('napi --version', { stdio: 'inherit' });
  } catch (error) {
    console.log('Installing @napi-rs/cli...');
    execSync('npm install @napi-rs/cli', { stdio: 'inherit' });
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

  execSync(`napi ${buildArgs.join(' ')}`, {
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
