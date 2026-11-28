---
# Copy to ~/.collab/watch.md and edit. Never commit it.
# Start the watcher in the repo:  python collab/bin/collab.py watch
interval_seconds: 60
# Shown for every new message and every decision the other person changes.
# macOS:   osascript -e "display notification \"$COLLAB_TEXT\" with title \"collab\""
# Linux:   notify-send collab "$COLLAB_TEXT"
notify: osascript -e "display notification \"$COLLAB_TEXT\" with title \"collab\""
# Optional: start a headless agent for these message types. Leave empty to only notify.
# Pick ONE line for your tool (see SETUP.md for what each flag allows):
# handler: claude -p "Follow collab/prompts/handle_message.md for message $COLLAB_MSG_ID." --allowedTools "Read" "Bash(python collab/bin/collab.py *)" "Bash(python3 collab/bin/collab.py *)" "Bash(ssh cluster test -r *)"
# handler: codex exec --sandbox workspace-write -c sandbox_workspace_write.network_access=true "Follow collab/prompts/handle_message.md for message $COLLAB_MSG_ID."
# handler: agent -p --force "Follow collab/prompts/handle_message.md for message $COLLAB_MSG_ID."
handler:
handle_types: [request, question]
max_handler_runs_per_hour: 6
handler_timeout_seconds: 900
---

Notifications go to you for every new message. A handler run deals only with the
cases your delegation file covers and leaves a note (shown by `collab.py status`)
for everything else.
