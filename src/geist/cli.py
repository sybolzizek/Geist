"""Command line interface for Geist."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from geist.agent import GeistAgent, result_to_json_text
from geist.provider import ProviderError, save_auth_config
from geist.trust import TrustStore


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if raw and raw[0] in {"trust", "sessions", "login"}:
        parser = _build_command_parser()
        args = parser.parse_args(raw)
    else:
        parser = _build_run_parser()
        args = parser.parse_args(raw)
    workspace = Path(args.cwd or ".").resolve()
    if args.command == "trust":
        info = TrustStore().trust(workspace)
        print(f"trusted {info['path']}")
        return 0
    if args.command == "login":
        return _login(args)
    if args.command == "sessions":
        return _sessions(args, workspace)
    if args.prompt:
        return asyncio.run(_run_print(args, workspace))
    return asyncio.run(_run_repl(args, workspace))


def _build_run_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="geist",
        description="Geist local fractal coding agent",
        epilog="Commands: geist trust, geist sessions, geist login",
    )
    parser.add_argument("prompt", nargs="*", help="Prompt to run. Omit for interactive mode.")
    parser.add_argument("-C", "--cwd", help="Workspace directory. Default current directory.")
    parser.add_argument("-p", "--print", dest="print_mode", action="store_true", help="Run one prompt and print the response.")
    parser.add_argument("--json", action="store_true", help="Emit JSON for one-shot runs.")
    parser.add_argument("-c", "--continue", dest="continue_latest", action="store_true", help="Continue the latest session for this workspace.")
    parser.add_argument("--session", help="Session id to use.")
    parser.add_argument("--no-session", action="store_true", help="Do not persist session events.")
    parser.add_argument("--trusted", action="store_true", help="Trust this project for this run.")
    parser.add_argument("--no-trust", action="store_true", help="Do not load trusted project-local .geist context for this run.")
    parser.add_argument("--debug", action="store_true", help="Show compact runtime trace events during the run.")
    parser.set_defaults(command=None)
    return parser


def _build_command_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="geist", description="Geist management commands")
    sub = parser.add_subparsers(dest="command", required=True)
    trust = sub.add_parser("trust", help="Trust a workspace for project-local .geist context.")
    trust.add_argument("-C", "--cwd", help="Workspace directory. Default current directory.")
    sessions = sub.add_parser("sessions", help="Show session storage location for a workspace.")
    sessions.add_argument("-C", "--cwd", help="Workspace directory. Default current directory.")
    sessions.add_argument("--json", action="store_true", help="Emit JSON.")
    login = sub.add_parser("login", help="Save OpenAI-compatible provider configuration.")
    login.add_argument("-C", "--cwd", help=argparse.SUPPRESS)
    login.add_argument("--json", action="store_true", help=argparse.SUPPRESS)
    login.add_argument("--api-key", required=True)
    login.add_argument("--base-url", default="https://api.openai.com/v1")
    login.add_argument("--model", required=True)
    return parser


async def _run_print(args: argparse.Namespace, workspace: Path) -> int:
    prompt = " ".join(args.prompt).strip()
    try:
        agent = _agent(args, workspace)
        result = await agent.run_turn(prompt)
    except ProviderError as exc:
        return _error(str(exc), json_mode=args.json)
    if args.json:
        print(result_to_json_text(result))
    else:
        if result.error and not result.ok:
            print(f"[runtime error: {result.error}]")
        print(result.response)
    return 0 if result.ok else 1


async def _run_repl(args: argparse.Namespace, workspace: Path) -> int:
    try:
        agent = _agent(args, workspace)
    except ProviderError as exc:
        print(f"provider error: {exc}", file=sys.stderr)
        return 2
    print(f"geist | {workspace}")
    if agent.session:
        print(f"session {agent.session.session_id}")
    print("type /exit to quit, /tools to list tools, /trust to trust this project")
    while True:
        try:
            prompt = input("geist > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not prompt:
            continue
        if prompt in {"/exit", "/quit"}:
            return 0
        if prompt == "/tools":
            print(", ".join(sorted(agent.dispatcher.get_tools())))
            continue
        if prompt == "/session":
            if agent.session:
                print(f"{agent.session.session_id} | {agent.session.path}")
            else:
                print("session disabled")
            continue
        if prompt == "/model":
            config = getattr(agent.provider, "config", None)
            print(getattr(config, "model", "unknown"))
            continue
        if prompt == "/compact":
            if agent.session:
                agent.session_store.append(agent.session, {"event": "compact", "note": "manual compaction marker"})
            print("compaction marker written")
            continue
        if prompt == "/trust":
            TrustStore().trust(workspace)
            agent.trusted = True
            print(f"trusted {workspace}")
            continue
        result = await agent.run_turn(prompt)
        if result.error and not result.ok:
            print(f"[runtime error: {result.error}]")
        print(result.response)


def _agent(args: argparse.Namespace, workspace: Path) -> GeistAgent:
    trusted: bool | None = None
    if args.trusted:
        trusted = True
    if args.no_trust:
        trusted = False
    trace_sink = _debug_trace if args.debug else None
    return GeistAgent(
        workspace,
        trusted=trusted,
        session_id=args.session,
        continue_latest=args.continue_latest,
        use_session=not args.no_session,
        trace_sink=trace_sink,
    )


def _debug_trace(event: dict[str, Any]) -> None:
    label = str(event.get("event") or "")
    seq = event.get("sequence")
    bits = [f"[{seq}] {label}"]
    if event.get("tools"):
        bits.append("tools=" + ",".join(str(item) for item in event.get("tools") or []))
    if event.get("reason"):
        bits.append("reason=" + str(event.get("reason")))
    print(" ".join(bits), file=sys.stderr)


def _sessions(args: argparse.Namespace, workspace: Path) -> int:
    from geist.session import SessionStore

    store = SessionStore()
    latest = store.latest_session_id(workspace)
    payload = {
        "workspace": str(workspace),
        "store": str(store.root / store.workspace_key(workspace)),
        "latest": latest,
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"workspace: {payload['workspace']}")
        print(f"store: {payload['store']}")
        print(f"latest: {payload['latest'] or '-'}")
    return 0


def _login(args: argparse.Namespace) -> int:
    try:
        path = save_auth_config(api_key=args.api_key, base_url=args.base_url, model=args.model)
    except ProviderError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"saved provider config: {path}")
    return 0


def _error(message: str, *, json_mode: bool) -> int:
    if json_mode:
        print(json.dumps({"ok": False, "error": message}, ensure_ascii=False, indent=2))
    else:
        print(f"error: {message}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
