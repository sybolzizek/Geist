#!/usr/bin/env node
"use strict";

const { spawnSync } = require("node:child_process");
const fs = require("node:fs");
const path = require("node:path");

const root = path.resolve(__dirname, "..", "..");
const sourceDir = path.join(root, "src");
const sourceCli = path.join(sourceDir, "geist", "cli.py");
const userArgs = process.argv.slice(2);

function candidateCommands() {
  if (process.env.GEIST_PYTHON) {
    return [{ command: process.env.GEIST_PYTHON, prefix: [] }];
  }
  const commands = [
    { command: "python", prefix: [] },
    { command: "python3", prefix: [] },
  ];
  if (process.platform === "win32") {
    commands.unshift({ command: "py", prefix: ["-3"] });
  }
  return commands;
}

function findPython() {
  for (const item of candidateCommands()) {
    const probe = spawnSync(
      item.command,
      [
        ...item.prefix,
        "-c",
        "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)",
      ],
      { stdio: "ignore" },
    );
    if (probe.status === 0) {
      return item;
    }
  }
  return null;
}

const python = findPython();
if (!python) {
  console.error("geist requires Python 3.10+ on PATH. Set GEIST_PYTHON to a Python executable if needed.");
  process.exit(127);
}

const env = { ...process.env };
if (fs.existsSync(sourceCli)) {
  env.PYTHONPATH = env.PYTHONPATH
    ? `${sourceDir}${path.delimiter}${env.PYTHONPATH}`
    : sourceDir;
}

const result = spawnSync(
  python.command,
  [...python.prefix, "-m", "geist.cli", ...userArgs],
  { stdio: "inherit", env },
);

if (result.error) {
  console.error(`failed to start geist: ${result.error.message}`);
  process.exit(1);
}
process.exit(result.status === null ? 1 : result.status);
