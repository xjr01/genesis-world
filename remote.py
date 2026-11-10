"""Synchronize this workspace and run commands in its Linux Conda environment."""

import argparse
import json
import os
import re
import shlex
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

ROOT = Path(__file__).resolve().parent
LOCAL = ROOT / ".remote-sync"
REMOTE = "/data1/xjr/genesis-world-demos"
HOST = "genesis-remote"


class SyncSession(NamedTuple):
    name: str
    path: str
    ignores: tuple[str, ...]


SESSIONS = (
    SyncSession("genesis-workspace", ".", ("/out", "/outputs")),
    SyncSession("genesis-runs", "out/remote-runs", ()),
    SyncSession("genesis-outputs", "outputs", ("pytest_*/", "*_tmp/")),
    SyncSession("genesis-results", "out", ("/remote-runs", "pytest_*/", "*_tmp/")),
)


def mutagen(*arguments: str, is_capture: bool = False) -> subprocess.CompletedProcess:
    """Run the project's isolated synchronization client."""
    return subprocess.run(
        [str(LOCAL / "mutagen" / "mutagen.exe"), *arguments],
        cwd=ROOT,
        check=True,
        capture_output=is_capture,
        text=True,
        encoding="utf-8",
    )


def ssh(command: str, script: str | None = None) -> subprocess.CompletedProcess:
    """Execute a shell command through the project's password-only SSH transport."""
    return subprocess.run(
        [str(LOCAL / "ssh.exe"), HOST, command], input=None if script is None else script.encode("utf-8"), check=True
    )


def start() -> None:
    """Create missing sessions and resume existing sessions."""
    mutagen("daemon", "start")
    states = json.loads(mutagen("sync", "list", "--template", "{{json .}}", is_capture=True).stdout)
    for session in SESSIONS:
        state = next((state for state in states if state["name"] == session.name), None)
        if state is not None:
            if state["paused"]:
                mutagen("sync", "resume", session.name)
            continue
        (ROOT / session.path).mkdir(parents=True, exist_ok=True)
        destination = REMOTE + "/workspace"
        if session.path != ".":
            destination += "/" + session.path
        arguments = [
            "sync",
            "create",
            "--name",
            session.name,
            "--label",
            "project=genesis-world-demos",
            "--no-global-configuration",
            "--configuration-file",
            str(LOCAL / "sync-defaults.yml"),
        ]
        for pattern in session.ignores:
            arguments.extend(("--ignore", pattern))
        arguments.extend((str(ROOT / session.path), HOST + ":" + destination))
        mutagen(*arguments)


def flush_workspace() -> None:
    """Finish code synchronization before running a command or taking a snapshot."""
    mutagen("sync", "flush", "genesis-workspace")
    states = json.loads(
        mutagen("sync", "list", "genesis-workspace", "--template", "{{json .}}", is_capture=True).stdout
    )
    for state in states:
        if state.get("conflicts") or state.get("lastError"):
            raise RuntimeError("Workspace synchronization needs attention; run 'python remote.py status'.")
        for endpoint in (state["alpha"], state["beta"]):
            if endpoint.get("scanProblems") or endpoint.get("transitionProblems"):
                raise RuntimeError("Workspace files need attention; run 'python remote.py status'.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    commands.add_parser("start", help="Start automatic synchronization.")
    status = commands.add_parser("status", help="Show progress and conflicts.")
    status.add_argument("--json", dest="is_json", action="store_true")
    commands.add_parser("flush", help="Wait for all files, including historical results.")
    commands.add_parser("pause", help="Pause synchronization.")
    commands.add_parser("resume", help="Resume synchronization.")
    commands.add_parser("backup", help="Take a remote code and resource snapshot.")
    run = commands.add_parser("run", help="Flush code and launch a detached remote command.")
    run.add_argument("--gpu", default="4", help="CUDA_VISIBLE_DEVICES value (default: 4).")
    run.add_argument("--display", choices=("virtual", "none"), default="virtual")
    run.add_argument("command", nargs=argparse.REMAINDER)
    logs = commands.add_parser("logs", help="Read the latest output of a remote job.")
    logs.add_argument("job")
    args = parser.parse_args()
    os.environ["MUTAGEN_DATA_DIRECTORY"] = str(LOCAL / "state")
    os.environ["MUTAGEN_SSH_PATH"] = str(LOCAL)

    if args.action == "start":
        start()
    elif args.action == "status":
        options = ("--template", "{{json .}}") if args.is_json else ("--long",)
        mutagen("sync", "list", *options)
    elif args.action in ("flush", "pause", "resume"):
        mutagen("sync", args.action, *(session.name for session in SESSIONS))
    elif args.action == "backup":
        flush_workspace()
        ssh("bash " + shlex.quote(REMOTE + "/tools/backup.sh"))
    elif args.action == "logs":
        if re.fullmatch(r"[0-9]{8}T[0-9]{6}Z-[0-9a-f]{6}", args.job) is None:
            parser.error("Use the job identifier printed by the run command.")
        directory = REMOTE + "/workspace/out/remote-runs/" + args.job
        ssh(
            "tail -n 100 " + shlex.quote(directory + "/run.log") + "; "
            "if test -f " + shlex.quote(directory + "/exit-code") + "; then "
            "printf '\\nExit code: '; cat " + shlex.quote(directory + "/exit-code") + "; fi"
        )
    elif args.action == "run":
        arguments = args.command[1:] if args.command[:1] == ["--"] else args.command
        if not arguments:
            parser.error("Supply a command after '--', for example '-- python examples/vbd/vbd_shell.py -g'.")
        flush_workspace()
        job = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:6]
        directory = REMOTE + "/workspace/out/remote-runs/" + job
        task = REMOTE + "/tools/jobs/" + job + ".sh"
        if args.display == "virtual":
            arguments = ["xvfb-run", "--auto-servernum", "--server-args=-screen 0 1280x720x24", *arguments]
        exit_trap = 'status=$?; printf "%s\\n" "$status" > ' + shlex.quote(directory + "/exit-code")
        script = (
            "set -eu\n"
            "trap " + shlex.quote(exit_trap) + " EXIT\n"
            "source " + shlex.quote(REMOTE + "/tools/remote-env.sh") + "\n"
            "cd " + shlex.quote(REMOTE + "/workspace") + "\n"
            "export CUDA_VISIBLE_DEVICES=" + shlex.quote(args.gpu) + "\n"
            "export GENESIS_RUN_DIRECTORY=" + shlex.quote(directory) + "\n" + shlex.join(arguments) + "\n"
        )
        ssh("mkdir -p " + shlex.quote(directory) + " " + shlex.quote(REMOTE + "/tools/jobs"))
        ssh("cat > " + shlex.quote(task), script=script)
        ssh(
            "nohup bash "
            + shlex.quote(task)
            + " > "
            + shlex.quote(directory + "/run.log")
            + " 2>&1 < /dev/null & echo $!"
        )
        print("Job:", job)
        print("Local log:", ROOT / "out" / "remote-runs" / job / "run.log")
        print("Inspect:", "python remote.py logs", job)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode) from error
