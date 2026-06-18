from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from geist.local import LocalWorkspace


def test_local_workspace_file_operations(tmp_path: Path) -> None:
    workspace = LocalWorkspace(tmp_path)

    written = workspace.write("notes/a.txt", "one\ntwo\nthree\n")
    read = workspace.read("notes/a.txt", line_start=2, line_count=1)
    edited = workspace.edit("notes/a.txt", old_text="two", new_text="TWO")
    listed = workspace.list(".", recursive=True)

    assert written["ok"] is True
    assert read["content"] == "two\n"
    assert edited["ok"] is True
    assert workspace.read("notes/a.txt")["content"] == "one\nTWO\nthree\n"
    assert any(item["relative_path"] == "notes/a.txt" for item in listed["entries"])


def test_local_workspace_rejects_path_escape(tmp_path: Path) -> None:
    workspace = LocalWorkspace(tmp_path)

    result = workspace.write("../outside.txt", "nope")

    assert result["ok"] is False
    assert "inside workspace" in result["error"]


def test_local_workspace_runs_bounded_command(tmp_path: Path) -> None:
    workspace = LocalWorkspace(tmp_path)

    result = workspace.run([sys.executable, "-c", "print('hello geist')"])

    assert result["ok"] is True
    assert result["returncode"] == 0
    assert result["stdout"].strip() == "hello geist"


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not available")
def test_local_workspace_git_status_diff_and_delta(tmp_path: Path) -> None:
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    subprocess.run(["git", "config", "user.email", "geist@example.local"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Geist Test"], cwd=tmp_path, check=True)
    (tmp_path / "app.py").write_text("print('one')\n", encoding="utf-8")
    subprocess.run(["git", "add", "app.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=tmp_path, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    workspace = LocalWorkspace(tmp_path)
    before = workspace.git_snapshot()
    workspace.write("app.py", "print('two')\n")
    status = workspace.git_status()
    summary = workspace.git_diff_summary(paths="app.py")
    patch = workspace.git_diff_read(paths="app.py")
    delta = workspace.git_delta(before["snapshot"])

    assert status["ok"] is True
    assert status["dirty"] is True
    assert summary["ok"] is True
    assert summary["files"][0]["path"] == "app.py"
    assert patch["ok"] is True
    assert "print('two')" in patch["stdout"]
    assert delta["ok"] is True
    assert delta["moved"] is True
    assert delta["added"] == ["app.py"]
