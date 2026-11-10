Genesis remote development

Server: user@111.62.35.107:22
Remote root: /data1/xjr/genesis-world-demos
Workspace: /data1/xjr/genesis-world-demos/workspace
Python: /data1/xjr/genesis-world-demos/envs/genesis-world/bin/python
Authentication: password from .agents/memory.md, with public-key authentication disabled.

From the local project directory:
  python remote.py start
  python remote.py status
  python remote.py pause
  python remote.py resume
  python remote.py flush
  python remote.py backup
  python remote.py run --gpu 4 -- python examples/vbd/vbd_shell.py -g -v --scenario hang -r -o out/remote-runs/vbd-hang
  python remote.py logs JOB_ID

The Windows task GenesisWorldDemosSync starts synchronization at login.
Four independent two-way-safe sessions synchronize the workspace, outputs/,
out/, and out/remote-runs/. Code flushes before a run. Large historical results
transfer independently. Conflicting edits require resolution before those files
can converge. Deletions propagate in both directions.

The run command launches a detached job. Each job writes run.log and exit-code
under out/remote-runs/JOB_ID. An exit-code of 0 indicates success. The job exports
GENESIS_RUN_DIRECTORY for custom output scripts. Pass the example's output-dir
option to out/remote-runs/NAME for prompt result delivery during the first upload.
The default virtual X display supports viewer-enabled commands on the server.
Its software rendering is slower; simulation physics still uses the selected GPU.
Use --display none before -- for programs that need no display.

Independent files: Git metadata, credentials, .agents, .codex, .remote-sync,
virtual environments, build products, platform binaries, caches, and temporary
pytest directories. See sync-defaults.yml and SESSIONS in remote.py for exact rules.
The project's .gitignore does not determine synchronization.

Code/resource snapshots: remote backups/code/, daily at 03:30 UTC, retained for
14 days. Snapshots use hard links between immutable snapshots. out/, outputs/,
and root-level MP4 results are kept as synchronized copies instead of snapshots.
The server's crontab entry is marked 'genesis-world-demos snapshot'.

The Conda environment pins the local genesis-world Python/package versions.
msvc_runtime is Windows-specific and excluded. Linux-specific Genesis
dependencies are resolved from pyproject.toml. Package manifests and the
installation scripts are in this directory and remote tools/.

Mutagen 0.18.1 is installed locally in mutagen/ and verified against the official
SHA256SUMS. transport.cs implements the password prompt and redirects the remote
agent into the project directory. The native OpenSSH server key is pinned in
known_hosts. Runtime state and downloaded tools stay in .remote-sync/.
