---
name: sync-fork-commits
description: Synchronize a fork branch with an upstream repository branch while replaying the fork's local commits one by one. Use when a user has a fork such as origin/dev that has local commits, the original upstream main has advanced, and they want the branch reset onto upstream while preserving non-duplicate local commits, dropping local commits already present upstream, and pushing back to the fork safely.
---

# Sync Fork Commits

## Overview

Use this workflow to rebuild a fork branch on top of the original upstream branch while preserving the user's local commits as individual commits. Prefer `cherry-pick` or `rebase --reapply-cherry-picks` patterns that let Git skip or expose duplicate commits; do not squash unless the user explicitly asks.

## Inputs

- Local branch to update, usually the current branch.
- Fork remote and target branch, usually `origin` and the same branch name.
- Upstream repository or remote, usually `upstream/main` or an official repo URL fetched into `refs/remotes/upstream/main`.
- Local commits to preserve, normally the commits on the fork branch that are not in upstream.

## Workflow

1. Inspect the repository before changing history.

   ```bash
   git branch --show-current
   git remote -v
   git log --oneline --decorate -12
   git status --short --branch
   ```

   If Git LFS filters make `status` fail in a sandbox, retry with LFS disabled only for inspection:

   ```bash
   git -c filter.lfs.process= -c filter.lfs.clean=cat -c filter.lfs.smudge=cat -c filter.lfs.required=false status --short --branch
   ```

2. Fetch the official upstream branch into a stable local ref.

   ```bash
   git fetch <upstream-url-or-remote> main:refs/remotes/upstream/main
   ```

   Use the user's stated upstream remote or official URL. If no upstream remote exists, fetching by URL is acceptable.

3. Determine the local commits to replay.

   ```bash
   git log --oneline --decorate --graph --left-right upstream/main...HEAD
   git rev-list --left-right --count upstream/main...HEAD
   git cherry -v upstream/main HEAD
   ```

   Treat lines from `git cherry -v` beginning with `-` as patch-equivalent to upstream and usually drop them. Treat lines beginning with `+` as local commits to replay, preserving chronological order from oldest to newest.

4. Create a temporary branch from upstream and replay non-duplicate local commits.

   ```bash
   git switch -c codex/<branch>-sync upstream/main
   git cherry-pick <oldest-local-commit>
   git cherry-pick <next-local-commit>
   ```

   Prefer explicit `cherry-pick` when the local commit count is small or the user referred to specific commits. For many commits, use an equivalent `rebase --onto upstream/main <merge-base> <branch>` only after confirming duplicates will be handled correctly.

5. Resolve conflicts narrowly.

   - Inspect conflict files and prefer upstream when the local change is already represented upstream.
   - Preserve the user's local behavior when it is not duplicated upstream.
   - Use `git cherry-pick --skip` for a commit that becomes empty or patch-equivalent after conflict resolution.
   - Use `git cherry-pick --continue` after resolving and staging conflict files.

6. Validate the rebuilt branch.

   ```bash
   git log --oneline --decorate --graph --left-right upstream/main...HEAD
   git rev-list --left-right --count upstream/main...HEAD
   git diff --stat upstream/main..HEAD
   git status --short --branch
   ```

   The expected count is usually `0 N`, where `N` is the number of preserved local commits.

7. Move the real local branch and push to the fork.

   ```bash
   git branch -f <branch> HEAD
   git switch <branch>
   git push --force-with-lease <fork-remote> <branch>
   ```

   Use `--force-with-lease`, not plain `--force`, because this workflow rewrites the fork branch history.

8. Clean temporary branches only after successful push.

   ```bash
   git branch -D codex/<branch>-sync
   ```

## Planning Script

Use `scripts/plan_sync.py` when a quick read-only summary helps. It prints the current branch, commit counts, and `git cherry -v` classification:

```bash
python3 .agents/skills/sync-fork-commits/scripts/plan_sync.py --upstream upstream/main --branch HEAD
```

The script does not change the repository. Still inspect its output before rewriting history.

## Safety Rules

- Never discard uncommitted work. Stop and ask the user unless the changes are clearly unrelated and the requested operation can continue without touching them.
- Never squash commits unless the user explicitly asks.
- Drop duplicate local commits when Git marks them patch-equivalent to upstream or a cherry-pick becomes empty.
- Keep commit order stable: oldest local commit first, newest last.
- Use a temporary branch for the rebuild, then update the target branch after validation.
- Push rewritten fork branches with `--force-with-lease`.
- Report the final upstream base, preserved commit hashes, dropped duplicate hashes if any, and push result.
