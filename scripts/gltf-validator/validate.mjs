// Validate glTF files with the Khronos glTF-Validator (the version pinned in
// package.json; `npm ci` here first). Buffers and images are loaded next to
// each file, so they are validated too.
//
//   node scripts/gltf-validator/validate.mjs <file.gltf | dir> ...
//
// Directories are searched for *.gltf/*.glb. Prints one JSON report on
// stdout: the validator's version, per file its issue counts and every error
// and warning, and totals with a count per issue code. Exit code 1 when any
// file has an error.

import { readdir, readFile, stat } from "node:fs/promises";
import path from "node:path";
import validator from "gltf-validator";

const SEVERITY = ["error", "warning"]; // infos and hints are only counted

async function gltfFiles(target) {
  if (!(await stat(target)).isDirectory()) return [target];
  const entries = await readdir(target, { recursive: true, withFileTypes: true });
  return entries
    .filter((e) => e.isFile() && /\.(gltf|glb)$/i.test(e.name))
    .map((e) => path.join(e.parentPath, e.name))
    .sort();
}

async function validate(file) {
  const dir = path.dirname(file);
  const report = await validator.validateBytes(new Uint8Array(await readFile(file)), {
    uri: path.basename(file),
    maxIssues: 0, // unlimited
    writeTimestamp: false,
    externalResourceFunction: async (uri) => new Uint8Array(await readFile(path.resolve(dir, decodeURIComponent(uri)))),
  });
  const { numErrors, numWarnings, numInfos, numHints, messages } = report.issues;
  return {
    file,
    errors: numErrors,
    warnings: numWarnings,
    infos: numInfos,
    hints: numHints,
    messages: messages
      .filter((m) => m.severity < SEVERITY.length)
      .map(({ code, severity, pointer, message }) => ({ code, severity: SEVERITY[severity], pointer, message })),
  };
}

const targets = process.argv.slice(2);
if (targets.length === 0) {
  console.error("usage: node validate.mjs <file.gltf | dir> ...");
  process.exit(2);
}

const files = [];
for (const target of targets) {
  for (const file of await gltfFiles(target)) files.push(await validate(file));
}

const totals = { files: files.length, errors: 0, warnings: 0, infos: 0, hints: 0 };
const codes = {};
for (const f of files) {
  for (const key of ["errors", "warnings", "infos", "hints"]) totals[key] += f[key];
  for (const m of f.messages) codes[m.code] = (codes[m.code] ?? 0) + 1;
}

console.log(JSON.stringify({ validator: validator.version(), totals, codes, files }, null, 1));
process.exitCode = totals.errors > 0 ? 1 : 0;
