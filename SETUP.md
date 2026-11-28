# Setup

All tools and personal setup below are optional; existing project workflows and
AGENTS.md take precedence. Shared files go in the repository. Personal files go in each person's home
directory and are never committed.

## 1. Repository (once, either person)

Copy into the repo root and commit:

```
AGENTS.md                        shared instructions (Claude Code, Cursor, Codex all read it)
collab/PROTOCOL.md               formats and procedures
collab/CHARTER.md                fill in goal, values, constraints, priority order
collab/GLOSSARY.md               the terms you already use differently
collab/TEAM.md                   names, git emails (laptop AND cluster), cluster usernames
collab/templates/decision.md
collab/prompts/handle_message.md prompt for watcher-started agent runs
collab/decisions/                seed with your spec's open decisions
collab/bin/collab.py
scripts/launch_example.py        optional: configure and rename if desired
.githooks/pre-commit             optional: lint decision files
.gitignore                       must contain .collab-local/
```

Do not add `CLAUDE.md` or `CLAUDE.local.md` to the repo. Claude Code reads those
instead of `AGENTS.md`, and Cursor reads both, so you would get two copies.

## 2. Optional personal setup, once per machine

```bash
mkdir -p ~/.collab
echo harry > ~/.collab/me                                   # or nazar
cp personal/delegation.example.md ~/.collab/delegation.md   # then edit it
cp personal/watch.example.md ~/.collab/watch.md             # then edit it
git config core.hooksPath .githooks                         # optional
git config user.email                                       # must be listed in collab/TEAM.md
```

Personal instruction file:

| Tool | Where | Snippet |
|---|---|---|
| Claude Code (Harry) | `~/.claude/rules/collab.md` | `personal/claude-code-rule.md` |
| Codex (Nazar) | append to `~/.codex/AGENTS.md` | `personal/codex-global-AGENTS.md` |
| Cursor (Nazar) | User Rules (Customize > Rules) | `personal/cursor-user-rule.md` |

## 3. Optional watcher

Run it in the repo on the machine where your agent tool is installed, in a
terminal or tmux window you leave open:

```bash
python collab/bin/collab.py watch
```

Start with `notify` only, with `handler` left empty, for a few days. Add a
handler once you trust how your agent handles messages interactively.

Handler choices (set `handler:` in `~/.collab/watch.md`):

| Tool | Command | What the flags allow |
|---|---|---|
| Claude Code | `claude -p "Follow collab/prompts/handle_message.md for message $COLLAB_MSG_ID." --allowedTools "Read" "Bash(python collab/bin/collab.py *)" "Bash(python3 collab/bin/collab.py *)" "Bash(ssh cluster test -r *)"` | Only the listed tools run without a prompt; anything else is refused, because nobody is there to approve it. |
| Codex | `codex exec --sandbox workspace-write -c sandbox_workspace_write.network_access=true "Follow collab/prompts/handle_message.md for message $COLLAB_MSG_ID."` | Writes limited to the repo; network on, because `collab.py` must fetch and push. |
| Cursor | `agent -p --force "Follow collab/prompts/handle_message.md for message $COLLAB_MSG_ID."` | Print mode has full write and shell access; the prompt is the only limit. |

Replace `cluster` with your ssh alias. Test a handler by hand first:
`COLLAB_MSG_ID=<an open message id> <handler command>`.

## 4. Smoke test (each tool, once)

1. In a new agent session in the repo, ask: "Summarize the collaboration rules
   you were given, and name your user." It should mention `collab.py status`,
   checking `inbox` during the session, the three message cases, and your name.
2. Ask: "Check for collaboration messages." It should run `collab.py status`.
3. From one side: "Send Nazar a question asking whether he can read `<a path>`."
   It should show the draft before sending. On the other side, the watcher
   should notify within a minute, and a new session's `status` should list it.
4. Run `check-run` on a config with an unratified decision. It should print an
   advisory and succeed by default; only an explicit strict audit should fail.
5. Edit a ratified decision and push. The other person's watcher should report
   that their ratification is void.

If step 1 fails in a tool, that tool is not loading `AGENTS.md`. Fix that before
relying on anything else.
