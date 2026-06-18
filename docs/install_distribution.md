# Geist Install And Distribution

This document tracks the practical installation paths for the standalone Geist
agent package. It is intentionally separate from the README while the project is
still settling.

## Python Package

The Python distribution name is:

```text
geist-agent
```

The import package and CLI command remain:

```text
geist
```

This avoids tying the public installer name to a possibly contested bare package
name while preserving the short command users should type.

## Local Development Install

From the repository root:

```powershell
python -m pip install -e ".[dev]"
geist --version
geist --help
```

The module entry point is also available:

```powershell
python -m geist --help
```

## pipx / uv Tool Install

Once published to PyPI:

```powershell
pipx install geist-agent
geist --help
```

With uv:

```powershell
uv tool install geist-agent
geist --help
```

For one-shot execution without a persistent tool install:

```powershell
uvx --from geist-agent geist --help
```

## npm / pnpm Wrapper

The repository also contains a thin Node launcher:

```text
npm/bin/geist.js
```

The npm package name is currently aligned with the Python distribution:

```text
geist-agent
```

The wrapper does not reimplement the agent. It locates Python 3.10+, injects the
bundled `src` directory into `PYTHONPATH`, and runs:

```text
python -m geist.cli
```

That makes these forms possible after npm publication:

```powershell
npx geist-agent --help
pnpm dlx geist-agent --help
```

Both command names are exposed:

```powershell
geist --help
geist-agent --help
```

The npm package still requires a local Python 3.10+ runtime. If Python is not on
`PATH`, users can point the wrapper at it:

```powershell
$env:GEIST_PYTHON="C:\Path\To\python.exe"
npx geist-agent --help
```

Provider setup can be done interactively after install:

```powershell
geist login
```

Or non-interactively:

```powershell
geist login --api-key <key> --base-url <url> --model <model>
```

Health check:

```powershell
geist doctor
geist doctor --json
```

## Release Checklist

Before the first public release:

- choose and reserve the final PyPI and npm package names
- update package URLs after the public GitHub repository is fixed
- run `python -m pytest -q`
- run `python -m pip install -e ".[dev]"` and verify `geist --help`
- run `node npm/bin/geist.js --help`
- run a local tarball install and verify both `geist` and `geist-agent`
- build a wheel/sdist and inspect included files
- run `npm pack --dry-run` and inspect included files

## Current Constraint

The package currently has no third-party runtime dependency. If future provider,
TUI, MCP, or browser features add dependencies, the npm wrapper should either:

- keep bundling pure-Python source only for the minimal local CLI, or
- switch to a launcher that delegates to `uvx --from geist-agent geist`.
