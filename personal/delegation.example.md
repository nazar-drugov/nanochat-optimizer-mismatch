---
# Copy to ~/.collab/delegation.md and edit. Never commit it.
# collab.py reads these two keys when your agent claims the other person's runs.
claim_from: [harry]
claim_max_gpu_hours_per_week: 20
---

# What my agent may do for the other person's agent without asking me

Written by me. My agent must not edit this file. Anything not listed here
needs my approval in the current session.

## Answer without asking
- Questions about run status, run directories, checkpoint paths, and what a
  decision record says, answered from the records in the repository.
- Whether a path is readable from my account (`test -r`), for dir-visibility requests.

## Do without asking, then tell me
- Grant read-only ACLs to the other person on my run directories under my run root.
- Claim shared runs from the people in `claim_from`, within the weekly cap above,
  when I have a free job slot (ordinary interactive sessions, not watcher runs).
  Use an approved reviewed submission workflow.

## Always ask me first
- Material scientific changes outside my already-authorized plans.
- Any write access for the other person, anywhere.
- Deleting, moving or overwriting any run output.
- Cancelling or changing any job.
- Spending compute beyond the cap, or claiming a run tagged for one person only.
