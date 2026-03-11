# CE Repository Stabilization Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure the `CE` repository so that `wpa/` is the only active WPA directory, historical WPA variants and `compare/` are archived, and stale branches are removed safely.

**Architecture:** This plan performs a staged filesystem migration inside the existing monorepo, then repairs all path references and repository documentation before any branch pruning. The flow is intentionally conservative: inventory first, move archived material next, promote the active WPA directory last, and only then clean up branches.

**Tech Stack:** Git, zsh shell tooling, ripgrep, existing Python test suite, Markdown documentation

---

## Chunk 1: Inventory And Safety Boundary

### Task 1: Capture the current repo state

**Files:**
- Modify: `docs/superpowers/plans/2026-03-11-ce-repo-stabilization.md`

- [ ] **Step 1: Record the working tree state**

Run: `git status --short`
Expected: a list of existing unstaged or untracked files is shown.

- [ ] **Step 2: Record current branches**

Run: `git branch --format='%(refname:short)'`
Expected: includes `main`, `claude_method`, and stale historical branches.

- [ ] **Step 3: Record candidate path references**

Run: `rg -n "claude_version_cct_rgb_gain_method|wpa_python_port/claude_version|wpa_simple_version|compare/" .`
Expected: a list of docs, scripts, or tests that reference pre-migration paths.

- [ ] **Step 4: Save the inventory notes**

Add a short notes section to this plan or a sibling execution log listing the files that will need updates.

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/plans/2026-03-11-ce-repo-stabilization.md docs/superpowers/specs/2026-03-11-ce-repo-stabilization-design.md
git commit -m "docs: add CE repository stabilization design and plan"
```

## Chunk 2: Archive Historical Directories

### Task 2: Create the archive structure

**Files:**
- Create: `archive/`
- Create: `archive/wpa/`

- [ ] **Step 1: Create the archive directories**

Run: `mkdir -p archive/wpa`
Expected: command exits with code 0 and creates the target directories.

- [ ] **Step 2: Verify the new structure**

Run: `find archive -maxdepth 2 -type d | sort`
Expected: includes `archive` and `archive/wpa`.

- [ ] **Step 3: Commit**

```bash
git add archive
git commit -m "chore: add archive layout for historical algorithms"
```

### Task 3: Move obsolete directories into the archive

**Files:**
- Move: `compare/` -> `archive/compare/`
- Move: `wpa/` -> `archive/wpa/matlab-original/`
- Move: `wpa_python_port/claude_version/` -> `archive/wpa/python-port-legacy/`
- Move: `wpa_simple_version/` -> `archive/wpa/python-simple-legacy/`

- [ ] **Step 1: Move `compare/` into the archive**

Run: `git mv compare archive/compare`
Expected: `git status --short` shows a rename for `compare`.

- [ ] **Step 2: Move the MATLAB original directory**

Run: `git mv wpa archive/wpa/matlab-original`
Expected: `git status --short` shows the MATLAB source moved under `archive/wpa/`.

- [ ] **Step 3: Move the legacy Python port**

Run: `git mv wpa_python_port/claude_version archive/wpa/python-port-legacy`
Expected: the old path disappears and the archive path appears in status.

- [ ] **Step 4: Move the simplified WPA version**

Run: `git mv wpa_simple_version archive/wpa/python-simple-legacy`
Expected: the old path disappears and the archive path appears in status.

- [ ] **Step 5: Verify the archive content**

Run: `find archive -maxdepth 3 -type d | sort`
Expected: includes `archive/compare`, `archive/wpa/matlab-original`, `archive/wpa/python-port-legacy`, and `archive/wpa/python-simple-legacy`.

- [ ] **Step 6: Commit**

```bash
git add archive
git commit -m "chore: archive legacy WPA and comparison directories"
```

## Chunk 3: Promote The Active WPA Directory

### Task 4: Move the active WPA implementation to the repo root

**Files:**
- Move: `wpa_python_port/claude_version_cct_rgb_gain_method/` -> `wpa/`

- [ ] **Step 1: Ensure the old top-level `wpa/` path is free**

Run: `test ! -e wpa`
Expected: exit code 0 after the MATLAB directory has been archived.

- [ ] **Step 2: Promote the active WPA directory**

Run: `git mv wpa_python_port/claude_version_cct_rgb_gain_method wpa`
Expected: `git status --short` shows the active directory relocated to the root.

- [ ] **Step 3: Verify the root layout**

Run: `find . -maxdepth 1 -type d | sort`
Expected: root contains `./fca`, `./wpa`, and `./archive`.

- [ ] **Step 4: Commit**

```bash
git add wpa
git commit -m "refactor: promote active WPA directory to repo root"
```

## Chunk 4: Repair Paths And Documentation

### Task 5: Update path references

**Files:**
- Modify: `README.md`
- Modify: `wpa/README.md`
- Modify: all docs, scripts, and tests identified by the inventory task

- [ ] **Step 1: Re-scan for stale paths**

Run: `rg -n "claude_version_cct_rgb_gain_method|wpa_python_port/claude_version|wpa_simple_version|compare/" .`
Expected: only files that still need updating are listed.

- [ ] **Step 2: Update root documentation**

Edit `README.md` to define:
- `wpa/` as the only active WPA directory
- `archive/` as the historical area
- `archive/compare/` and `archive/wpa/*` as reference-only

- [ ] **Step 3: Update WPA-local documentation**

Edit `wpa/README.md` so the normal workflow points only at the new root path.

- [ ] **Step 4: Update scripts and tests with old paths**

Replace stale relative paths or hard-coded snippets so they point at `wpa/` or the new archive locations as appropriate.

- [ ] **Step 5: Verify no stale active-path references remain**

Run: `rg -n "claude_version_cct_rgb_gain_method|wpa_python_port/claude_version|wpa_simple_version" .`
Expected: no results, or only deliberate notes inside archive/history docs.

- [ ] **Step 6: Commit**

```bash
git add README.md wpa/README.md
git add .
git commit -m "docs: update CE paths after WPA migration"
```

## Chunk 5: Verification

### Task 6: Run repository checks after migration

**Files:**
- Test: existing WPA tests under `wpa/tests/`

- [ ] **Step 1: Run targeted WPA tests**

Run: `pytest wpa/tests -q`
Expected: passing test summary, or a known baseline failure list captured for follow-up.

- [ ] **Step 2: Run a focused path smoke check**

Run: `test -d wpa && test -d archive/wpa && test -d archive/compare`
Expected: exit code 0.

- [ ] **Step 3: Confirm the active-directory rule is visible**

Run: `rg -n "only active WPA|reference-only|archive/" README.md wpa/README.md`
Expected: documentation contains the new workflow rules.

- [ ] **Step 4: Commit**

```bash
git add .
git commit -m "test: verify CE repository stabilization"
```

## Chunk 6: Branch Cleanup

### Task 7: Prune stale branches

**Files:**
- Modify: Git refs only

- [ ] **Step 1: Build the keep-list**

Run: `printf "main\nclaude_method\n" && git branch --format='%(refname:short)' | rg '^feature/'`
Expected: a human-reviewed keep-list of active branches.

- [ ] **Step 2: Delete stale local branches**

Run: `git branch --format='%(refname:short)' | rg -v '^(main|claude_method|feature/...)$'`
Expected: shows only branches approved for deletion.

- [ ] **Step 3: Delete one stale branch at a time**

Run: `git branch -D <stale-branch>`
Expected: Git reports the branch was deleted locally.

- [ ] **Step 4: Delete stale remote branches if desired**

Run: `git push origin --delete <stale-branch>`
Expected: remote deletion succeeds, or the branch is already absent.

- [ ] **Step 5: Verify the final branch set**

Run: `git branch --format='%(refname:short)'`
Expected: only the approved branches remain locally.

## Execution Notes

- Do not mix unrelated algorithm edits into the migration commits.
- If path-sensitive tests fail, fix path references before changing behavior.
- If branch ownership is unclear, keep the branch and decide later.

## Handoff

Plan complete and saved to `docs/superpowers/plans/2026-03-11-ce-repo-stabilization.md`. Ready to execute?
