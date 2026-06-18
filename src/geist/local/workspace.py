"""Workspace-local coding substrate for Geist.

The object here is intentionally small: one root directory, explicit file
operations, bounded command execution, and read-only git instruments. It does
not know about prompts, memory, skills, or fractal flow.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class WorkspaceError(ValueError):
    """Raised when a requested local operation escapes the workspace."""


@dataclass(frozen=True)
class LocalWorkspace:
    """Root-scoped local coding workbench."""

    root: Path | str

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root).resolve())

    def resolve(self, raw_path: str | Path = ".") -> Path:
        text = str(raw_path or ".").strip() or "."
        path = Path(text)
        if not path.is_absolute():
            path = Path(self.root) / path
        resolved = path.resolve()
        try:
            resolved.relative_to(Path(self.root))
        except ValueError as exc:
            raise WorkspaceError(f"path must stay inside workspace: {raw_path}") from exc
        if _is_git_path(resolved):
            raise WorkspaceError("path must not target the .git directory")
        return resolved

    def entry(self, path: str | Path) -> dict[str, Any]:
        return _file_entry(self.resolve(path), root=Path(self.root))

    def read(
        self,
        path: str | Path,
        *,
        max_chars: int = 20000,
        offset: int = 0,
        line_start: int | None = None,
        line_count: int | None = None,
        encoding: str = "utf-8",
    ) -> dict[str, Any]:
        try:
            resolved = self.resolve(path)
            if not resolved.exists():
                return {"ok": False, "error": f"file not found: {path}"}
            if not resolved.is_file():
                return {"ok": False, "error": f"path is not a file: {path}"}
            text = resolved.read_text(encoding=encoding, errors="replace")
            content, range_info = _slice_text(
                text,
                max_chars=max_chars,
                offset=offset,
                line_start=line_start,
                line_count=line_count,
            )
            return {
                "ok": True,
                "action": "read",
                "entry": _file_entry(resolved, root=Path(self.root)),
                "content": content,
                "range": range_info,
                "truncated": bool(range_info.get("truncated")),
                "next_read": _next_read(path=path, range_info=range_info, max_chars=max_chars),
            }
        except Exception as exc:
            return {"ok": False, "error": f"read failed: {exc}"}

    def write(
        self,
        path: str | Path,
        content: str,
        *,
        append: bool = False,
        overwrite: bool = True,
        expected_sha256: str = "",
        encoding: str = "utf-8",
    ) -> dict[str, Any]:
        try:
            resolved = self.resolve(path)
            if resolved.exists() and not resolved.is_file():
                return {"ok": False, "error": f"path exists and is not a file: {path}"}
            if resolved.exists() and not overwrite and not append:
                return {"ok": False, "error": f"file already exists: {path}", "current": _file_entry(resolved, root=Path(self.root))}
            before = _file_entry(resolved, root=Path(self.root)) if resolved.exists() else None
            if expected_sha256 and before and before.get("sha256") != expected_sha256:
                return {
                    "ok": False,
                    "error": "expected_sha256 mismatch",
                    "expected_sha256": expected_sha256,
                    "current": before,
                }
            resolved.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if append else "w"
            with resolved.open(mode, encoding=encoding, errors="replace") as handle:
                handle.write(str(content or ""))
            after = _file_entry(resolved, root=Path(self.root))
            return {"ok": True, "action": "write", "path": str(resolved), "before": before, "after": after}
        except Exception as exc:
            return {"ok": False, "error": f"write failed: {exc}"}

    def edit(
        self,
        path: str | Path,
        *,
        old_text: str,
        new_text: str,
        replace_all: bool = False,
        expected_sha256: str = "",
        dry_run: bool = False,
        encoding: str = "utf-8",
    ) -> dict[str, Any]:
        try:
            resolved = self.resolve(path)
            if not resolved.exists() or not resolved.is_file():
                return {"ok": False, "error": f"file not found: {path}"}
            before = _file_entry(resolved, root=Path(self.root))
            if expected_sha256 and before.get("sha256") != expected_sha256:
                return {
                    "ok": False,
                    "error": "expected_sha256 mismatch",
                    "expected_sha256": expected_sha256,
                    "current": before,
                }
            text = resolved.read_text(encoding=encoding, errors="replace")
            old = str(old_text)
            if old not in text:
                return {"ok": False, "error": "old_text not found", "current": before}
            count = text.count(old)
            updated = text.replace(old, str(new_text), -1 if replace_all else 1)
            planned = {
                "ok": True,
                "action": "edit",
                "path": str(resolved),
                "before": before,
                "replacement_count": count if replace_all else 1,
                "dry_run": dry_run,
            }
            if dry_run:
                return planned
            resolved.write_text(updated, encoding=encoding, errors="replace")
            return {**planned, "after": _file_entry(resolved, root=Path(self.root))}
        except Exception as exc:
            return {"ok": False, "error": f"edit failed: {exc}"}

    def list(
        self,
        path: str | Path = ".",
        *,
        pattern: str = "*",
        recursive: bool = False,
        max_entries: int = 200,
    ) -> dict[str, Any]:
        try:
            resolved = self.resolve(path)
            if not resolved.exists():
                return {"ok": False, "error": f"path not found: {path}"}
            if resolved.is_file():
                return {"ok": True, "action": "list", "entries": [_file_entry(resolved, root=Path(self.root))], "count": 1}
            iterator = resolved.rglob(pattern) if recursive else resolved.glob(pattern)
            candidates = [
                item
                for item in iterator
                if not _is_git_path(item) and not _is_skipped_path(item)
            ]
            entries = [_file_entry(item, root=Path(self.root)) for item in sorted(candidates, key=lambda p: str(p))[:max_entries]]
            return {
                "ok": True,
                "action": "list",
                "path": str(resolved),
                "entries": entries,
                "count": len(entries),
                "truncated": len(candidates) > max_entries,
            }
        except Exception as exc:
            return {"ok": False, "error": f"list failed: {exc}"}

    def run(
        self,
        command: str | list[str],
        *,
        cwd: str | Path = ".",
        timeout_ms: int = 30000,
        max_chars: int = 20000,
        env: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        try:
            parts = _command_parts(command)
            _validate_command(parts)
            working_dir = self.resolve(cwd)
            if not working_dir.exists() or not working_dir.is_dir():
                return {"ok": False, "error": f"cwd is not a directory: {cwd}"}
            run_env = dict(os.environ)
            run_env.setdefault("PYTHONIOENCODING", "utf-8")
            run_env.setdefault("PYTHONUTF8", "1")
            if env:
                run_env.update({str(k): str(v) for k, v in env.items()})
            started = datetime.now(timezone.utc).isoformat()
            proc = subprocess.run(
                parts,
                cwd=str(working_dir),
                env=run_env,
                text=True,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=max(int(timeout_ms or 30000), 1) / 1000,
                shell=False,
            )
            stdout, stdout_truncated = _truncate(str(proc.stdout or ""), max_chars)
            stderr, stderr_truncated = _truncate(str(proc.stderr or ""), max_chars)
            return {
                "ok": proc.returncode == 0,
                "action": "run",
                "command": parts,
                "cwd": str(working_dir),
                "started_at": started,
                "returncode": proc.returncode,
                "stdout": stdout,
                "stderr": stderr,
                "stdout_truncated": stdout_truncated,
                "stderr_truncated": stderr_truncated,
            }
        except subprocess.TimeoutExpired as exc:
            stdout, stdout_truncated = _truncate(str(exc.stdout or ""), max_chars)
            stderr, stderr_truncated = _truncate(str(exc.stderr or ""), max_chars)
            return {
                "ok": False,
                "action": "run",
                "error": "command timed out",
                "timeout_ms": timeout_ms,
                "stdout": stdout,
                "stderr": stderr,
                "stdout_truncated": stdout_truncated,
                "stderr_truncated": stderr_truncated,
            }
        except Exception as exc:
            return {"ok": False, "error": f"run failed: {exc}"}

    def git_status(self, *, path: str | Path = ".", max_files: int = 80, include_untracked: bool = True) -> dict[str, Any]:
        repo = self._git_repo(path)
        if repo.get("ok") is not True:
            return repo
        args = ["status", "--porcelain=v1", "-z"]
        if not include_untracked:
            args.append("--untracked-files=no")
        result = _run_git(Path(repo["repo_root"]), args)
        if result.get("ok") is not True:
            return result
        files = _parse_git_status(str(result.get("stdout") or ""))[:max_files]
        head = _run_git_text(Path(repo["repo_root"]), ["rev-parse", "--short", "HEAD"])
        branch = _run_git_text(Path(repo["repo_root"]), ["branch", "--show-current"])
        return {
            "ok": True,
            "action": "git.status",
            "repo_root": repo["repo_root"],
            "branch": branch or None,
            "head": head or None,
            "dirty": bool(files),
            "files": files,
            "count": len(files),
        }

    def git_diff_summary(
        self,
        *,
        path: str | Path = ".",
        paths: str | list[str] | None = None,
        staged: bool = False,
        max_files: int = 80,
    ) -> dict[str, Any]:
        repo = self._git_repo(path)
        if repo.get("ok") is not True:
            return repo
        args = ["diff", "--numstat"]
        if staged:
            args.append("--staged")
        args.extend(_git_pathspecs(paths))
        result = _run_git(Path(repo["repo_root"]), args)
        if result.get("ok") is not True:
            return result
        files = [_git_numstat_file(line) for line in str(result.get("stdout") or "").splitlines() if line.strip()]
        files = [item for item in files if item is not None][:max_files]
        return {
            "ok": True,
            "action": "git.diff_summary",
            "repo_root": repo["repo_root"],
            "staged": staged,
            "files": files,
            "count": len(files),
        }

    def git_diff_read(
        self,
        *,
        path: str | Path = ".",
        paths: str | list[str] | None = None,
        staged: bool = False,
        max_chars: int = 20000,
    ) -> dict[str, Any]:
        repo = self._git_repo(path)
        if repo.get("ok") is not True:
            return repo
        args = ["diff", "--patch"]
        if staged:
            args.append("--staged")
        args.extend(_git_pathspecs(paths))
        result = _run_git(Path(repo["repo_root"]), args, max_chars=max_chars)
        result["action"] = "git.diff_read"
        result["repo_root"] = repo["repo_root"]
        result["staged"] = staged
        return result

    def git_snapshot(self, *, path: str | Path = ".", max_files: int = 200) -> dict[str, Any]:
        status = self.git_status(path=path, max_files=max_files, include_untracked=True)
        if status.get("ok") is not True:
            return status
        snapshot = {
            "repo_root": status.get("repo_root"),
            "branch": status.get("branch"),
            "head": status.get("head"),
            "files": status.get("files") or [],
        }
        digest = hashlib.sha256(json.dumps(snapshot, sort_keys=True, default=str).encode("utf-8")).hexdigest()
        return {"ok": True, "action": "git.snapshot", "snapshot": snapshot, "sha256": digest}

    def git_delta(self, snapshot: dict[str, Any] | str, *, path: str | Path = ".", max_files: int = 200) -> dict[str, Any]:
        previous = _coerce_snapshot(snapshot)
        if previous is None:
            return {"ok": False, "error": "snapshot must be an object or JSON object string"}
        current = self.git_snapshot(path=path, max_files=max_files)
        if current.get("ok") is not True:
            return current
        before = _git_file_signature_map(previous.get("files"))
        after = _git_file_signature_map((current.get("snapshot") or {}).get("files"))
        added = sorted(set(after) - set(before))
        removed = sorted(set(before) - set(after))
        changed = sorted(key for key in set(before) & set(after) if before[key] != after[key])
        return {
            "ok": True,
            "action": "git.delta",
            "moved": bool(added or removed or changed),
            "added": added,
            "removed": removed,
            "changed": changed,
            "current": current.get("snapshot"),
        }

    def _git_repo(self, path: str | Path = ".") -> dict[str, Any]:
        try:
            anchor = self.resolve(path)
            cwd = anchor if anchor.is_dir() else anchor.parent
            result = _run_git(cwd, ["rev-parse", "--show-toplevel"])
            if result.get("ok") is not True:
                return {"ok": False, "error": "not a git repository", "path": str(anchor), "detail": result}
            repo_root = Path(str(result.get("stdout") or "").strip()).resolve()
            try:
                repo_root.relative_to(Path(self.root))
            except ValueError:
                return {"ok": False, "error": "git repo root is outside workspace", "repo_root": str(repo_root)}
            return {"ok": True, "repo_root": str(repo_root)}
        except Exception as exc:
            return {"ok": False, "error": f"git repo detection failed: {exc}"}


def _file_entry(path: Path, *, root: Path) -> dict[str, Any]:
    stat = path.stat()
    try:
        relative = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        relative = str(path)
    entry: dict[str, Any] = {
        "path": str(path),
        "relative_path": relative,
        "type": "dir" if path.is_dir() else "file" if path.is_file() else "other",
        "size": stat.st_size if path.is_file() else None,
        "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
    }
    if path.is_file():
        entry["sha256"] = _sha256_file(path)
    return entry


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _slice_text(
    text: str,
    *,
    max_chars: int,
    offset: int = 0,
    line_start: int | None = None,
    line_count: int | None = None,
) -> tuple[str, dict[str, Any]]:
    value = str(text or "")
    cap = max(int(max_chars or 0), 0)
    if line_start is not None:
        lines = value.splitlines(keepends=True)
        start = max(int(line_start), 1) - 1
        count = max(int(line_count or 0), 0) or None
        end = len(lines) if count is None else min(start + count, len(lines))
        selected = "".join(lines[start:end])
        selected, char_truncated = _truncate(selected, cap)
        return selected, {
            "mode": "lines",
            "line_start": start + 1,
            "line_end": start + len(selected.splitlines()),
            "total_lines": len(lines),
            "total_chars": len(value),
            "max_chars": cap,
            "truncated": end < len(lines) or char_truncated,
        }
    start = min(max(int(offset or 0), 0), len(value))
    end = len(value) if cap <= 0 else min(start + cap, len(value))
    return value[start:end], {
        "mode": "chars",
        "offset": start,
        "end": end,
        "total_chars": len(value),
        "total_lines": value.count("\n") + 1 if value else 0,
        "max_chars": cap,
        "truncated": end < len(value),
    }


def _next_read(*, path: str | Path, range_info: dict[str, Any], max_chars: int) -> dict[str, Any] | None:
    if not range_info.get("truncated"):
        return None
    if range_info.get("mode") == "lines":
        return {
            "tool": "read",
            "arguments": {"path": str(path), "line_start": int(range_info.get("line_end") or 0) + 1, "max_chars": max_chars},
        }
    return {
        "tool": "read",
        "arguments": {"path": str(path), "offset": range_info.get("end"), "max_chars": max_chars},
    }


def _truncate(text: str, max_chars: int) -> tuple[str, bool]:
    cap = max(int(max_chars or 0), 0)
    if cap <= 0 or len(text) <= cap:
        return text, False
    return text[:cap], True


def _command_parts(command: str | list[str]) -> list[str]:
    if isinstance(command, str):
        parts = shlex.split(command, posix=os.name != "nt")
    elif isinstance(command, list):
        parts = [str(item) for item in command]
    else:
        raise ValueError("command must be a string or argv list")
    if not parts or not str(parts[0]).strip():
        raise ValueError("command is required")
    return parts


def _validate_command(parts: list[str]) -> None:
    first = Path(parts[0]).name.lower()
    blocked = {"rm", "del", "erase", "format", "shutdown", "reboot"}
    if first in blocked:
        raise WorkspaceError(f"blocked destructive command: {parts[0]}")
    if first == "git":
        _validate_git_command(parts)
    for token in parts:
        if any(mark in token for mark in ("&&", "||", ";", "|", ">", "<")):
            raise WorkspaceError("shell chaining and redirection are not accepted; pass argv directly")
        if token == ".git" or token.startswith(".git/") or token.startswith(".git\\"):
            raise WorkspaceError("command must not target .git")


def _validate_git_command(parts: list[str]) -> None:
    if len(parts) < 2:
        return
    readonly = {
        "status",
        "diff",
        "show",
        "log",
        "rev-parse",
        "branch",
        "ls-files",
        "describe",
        "remote",
    }
    if parts[1] not in readonly:
        raise WorkspaceError(f"write-oriented git command is not allowed here: git {parts[1]}")


def _run_git(cwd: Path, args: list[str], *, max_chars: int = 200000) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=15,
            shell=False,
        )
        stdout, stdout_truncated = _truncate(str(proc.stdout or ""), max_chars)
        stderr, stderr_truncated = _truncate(str(proc.stderr or ""), max_chars)
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "stdout_truncated": stdout_truncated,
            "stderr_truncated": stderr_truncated,
        }
    except Exception as exc:
        return {"ok": False, "error": f"git failed: {exc}"}


def _run_git_text(cwd: Path, args: list[str]) -> str:
    result = _run_git(cwd, args, max_chars=20000)
    if result.get("ok") is not True:
        return ""
    return str(result.get("stdout") or "").strip()


def _parse_git_status(stdout: str) -> list[dict[str, Any]]:
    parts = [item for item in stdout.split("\0") if item]
    rows: list[dict[str, Any]] = []
    index = 0
    while index < len(parts):
        item = parts[index]
        xy = item[:2]
        path_text = item[3:] if len(item) > 3 else ""
        original_path = None
        if xy.startswith("R") or xy.startswith("C"):
            index += 1
            original_path = parts[index] if index < len(parts) else None
        rows.append({
            "xy": xy,
            "index_status": xy[:1],
            "worktree_status": xy[1:2],
            "path": path_text,
            "original_path": original_path,
            "untracked": xy == "??",
        })
        index += 1
    return rows


def _git_numstat_file(line: str) -> dict[str, Any] | None:
    parts = line.split("\t")
    if len(parts) < 3:
        return None
    added, deleted, path = parts[0], parts[1], parts[2]
    return {
        "path": path,
        "added": None if added == "-" else _safe_int(added),
        "deleted": None if deleted == "-" else _safe_int(deleted),
        "binary": added == "-" or deleted == "-",
    }


def _git_pathspecs(paths: str | list[str] | None) -> list[str]:
    if paths is None or paths == "":
        return []
    raw = paths if isinstance(paths, list) else [paths]
    result = []
    for item in raw:
        text = str(item or "").strip().replace("\\", "/")
        if not text or text.startswith("../") or "/.git/" in text or text.startswith(".git/"):
            continue
        result.append(text)
    return ["--", *result] if result else []


def _coerce_snapshot(value: dict[str, Any] | str) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except Exception:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _git_file_signature_map(files: Any) -> dict[str, str]:
    if not isinstance(files, list):
        return {}
    result: dict[str, str] = {}
    for item in files:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path") or "")
        if not path:
            continue
        result[path] = json.dumps(item, sort_keys=True, default=str)
    return result


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


_SKIP_DIRS = {".git", ".hg", ".svn", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox", ".venv", "venv", "__pycache__", "node_modules"}


def _is_git_path(path: Path) -> bool:
    return any(part == ".git" for part in path.parts)


def _is_skipped_path(path: Path) -> bool:
    return any(part in _SKIP_DIRS for part in path.parts)
