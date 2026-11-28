#!/usr/bin/env python3
"""Optional skeleton launcher; existing reviewed submission workflows are valid.

Not tested against your cluster. Fill in the CONFIGURE section, then rename to
scripts/launch.py.

  python scripts/launch.py submit configs/R-031.json [--requested-by nazar]
  python scripts/launch.py audit

submit: check-run -> tunnel check -> remote checkout of the pushed commit ->
        run dir with read ACL for the other person -> sbatch -> record-run
        -> optional authorized dir-visibility request to the other person's agent.
audit:  lists your running/pending jobs that have no run record.
"""
import argparse
import json
import os
import re
import shlex
import subprocess
import sys

# ---- CONFIGURE ------------------------------------------------------------
SSH_HOST = os.environ.get("COLLAB_SSH_HOST", "cluster")       # your ssh alias
REMOTE_REPO = "<path on cluster to your clone of this repo>"
RUN_ROOT = "<scratch or project path>/collab-runs"
SBATCH_SCRIPT = "scripts/train.sbatch"                       # takes the config path as $1
OTHER_CLUSTER_USER = "<other person's cluster username>"      # for the read ACL
OTHER_MEMBER = "<other person's collab name>"                 # as in collab/TEAM.md
# ---------------------------------------------------------------------------

SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", SSH_HOST]
COLLAB = [sys.executable, "collab/bin/collab.py"]


def sh(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        sys.exit("FAILED: %s\n%s" % (" ".join(cmd), (r.stderr or r.stdout).strip()))
    return r.stdout.strip()


def remote(script):
    return sh(SSH + [script])


def submit(args):
    cfg = json.load(open(args.config))
    run_id = cfg["run_id"]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        sys.exit("Invalid run ID")
    if any(value.startswith("<") for value in (REMOTE_REPO, RUN_ROOT, OTHER_CLUSTER_USER, OTHER_MEMBER)):
        sys.exit("Optional launcher is not configured; use the existing approved workflow.")
    # 1. Advisory decision review; this remote checkout needs a pushed clean commit.
    print(sh(COLLAB + ["check-run", args.config, "--require-pushed"]))
    commit = sh(["git", "rev-parse", "HEAD"])
    # 2. Tunnel alive (fails fast instead of hanging on a password or 2FA prompt).
    if subprocess.run(["ssh", "-O", "check", SSH_HOST], capture_output=True).returncode != 0:
        sys.exit("ssh control connection to %s is down; reopen the tunnel first." % SSH_HOST)
    # 3. Remote checkout of exactly this commit, run dir, read-only ACL for the other person.
    run_dir = "%s/%s" % (RUN_ROOT, run_id)
    q = shlex.quote
    # Refuse stale checkouts rather than silently ignoring worktree/config errors.
    remote(
        "set -e; cd {repo}; git fetch -q origin; mkdir -p {rd}; "
        "git worktree add -q --detach {code} {commit}; "
        "test \"$(git -C {code} rev-parse HEAD)\" = {commit}; "
        "mkdir -p {out}; cp {cfg} {dest}; "
        "setfacl -m {acl} {rd} {out}; setfacl -d -m {acl} {rd} {out}".format(
            repo=q(REMOTE_REPO), rd=q(run_dir), code=q(run_dir + "/code"),
            commit=q(commit), out=q(run_dir + "/out"),
            cfg=q(run_dir + "/code/" + args.config), dest=q(run_dir + "/config.json"),
            acl=q("u:%s:rX" % OTHER_CLUSTER_USER))
    )
    # 4. Submit.
    job_id = remote("cd %s && sbatch --parsable --job-name %s --chdir %s %s %s"
                    % (q(run_dir + "/code"), q("collab-" + run_id), q(run_dir + "/out"),
                       q(SBATCH_SCRIPT), q(run_dir + "/config.json"))).split(";")[0]
    # 5. Record and ask the other side to confirm read access.
    rec = COLLAB + ["record-run", "--run-id", run_id, "--job-id", job_id, "--config", args.config,
                    "--run-dir", run_dir, "--commit", commit]
    if args.requested_by:
        rec += ["--requested-by", args.requested_by]
    print(sh(rec))
    if args.request_visibility:
        print(sh(COLLAB + ["send", "--to", OTHER_MEMBER, "--type", "request", "--topic", "dir-visibility",
                           "--refs", run_id, "--subject", "Confirm read access to %s" % run_id,
                           "--body", "Please check `test -r %s/config.json` from your account and ack." % run_dir]))
    print("Submitted %s as job %s" % (run_id, job_id))


def audit(_args):
    """List your running/pending Slurm jobs that have no run record on the shared branch."""
    jobs = remote("squeue -u $USER -h -o '%i %j'").splitlines()
    branch = os.environ.get("COLLAB_BRANCH") or sh(["git", "symbolic-ref", "--short", "refs/remotes/origin/HEAD"]).split("/", 1)[1]
    sh(["git", "fetch", "--quiet", "origin", branch])
    names = sh(["git", "ls-tree", "-r", "--name-only", "origin/" + branch, "--", "collab/runs"]).splitlines()
    recorded = set()
    for n in names:
        for line in sh(["git", "show", "origin/%s:%s" % (branch, n)]).splitlines():
            if line.startswith("job_id:"):
                recorded.add(line.split(":", 1)[1].strip().strip('"'))
    stray = [j for j in jobs if j.split()[0] not in recorded]
    print("Jobs without a run record (%d):" % len(stray))
    for j in stray:
        print("  " + j)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("submit")
    s.add_argument("config")
    s.add_argument("--requested-by")
    s.add_argument("--request-visibility", action="store_true", help="send a visibility message only with user authorization")
    s.set_defaults(fn=submit)
    sub.add_parser("audit").set_defaults(fn=audit)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
