# CE Repository Stabilization Design

## Summary

This design defines a minimal-risk cleanup of the `CE` mono-repo so that:

- `WPA` has exactly one active development directory
- all obsolete `WPA` variants are moved into an explicit archive area
- `compare` is archived with the historical material
- Git branches are reduced to a small approved set

The goal is to stop accidental edits in the wrong place without splitting repositories yet.

## Confirmed Decisions

- Keep `CE` as the top-level repository for multiple algorithms.
- Keep `FCA` as an active top-level directory.
- Promote the current active WPA workspace from
  `wpa_python_port/claude_version_cct_rgb_gain_method`
  to the top-level active directory `wpa/`.
- Move the current top-level MATLAB `wpa/` directory into the archive and label it as the MATLAB original version.
- Move obsolete Python WPA variants into the archive.
- Move `compare/` into the archive.
- Keep only `main`, `claude_method`, and a small explicit allowlist of still-used `feature/*` branches.
- Delete all other stale branches after the directory migration is complete.

## Target Repository Layout

```text
CE/
├── fca/
├── wpa/                         # only active WPA directory
├── archive/
│   ├── wpa/
│   │   ├── matlab-original/
│   │   ├── python-port-legacy/
│   │   └── python-simple-legacy/
│   └── compare/
└── README.md
```

## Directory Mapping

### Active

- Current active directory:
  `wpa_python_port/claude_version_cct_rgb_gain_method`
- Destination:
  `wpa/`

This becomes the only place where WPA feature development is allowed.

### Archive

- Current MATLAB original:
  `wpa/`
  -> `archive/wpa/matlab-original/`
- Current Python legacy port:
  `wpa_python_port/claude_version/`
  -> `archive/wpa/python-port-legacy/`
- Current Python simplified version:
  `wpa_simple_version/`
  -> `archive/wpa/python-simple-legacy/`
- Current comparison utilities:
  `compare/`
  -> `archive/compare/`

Archive content is retained for reference only and should not receive new feature work.

## Naming Rules

- The active WPA directory uses the simple canonical name `wpa/`.
- Archived directories use descriptive historical labels rather than process labels such as `new`, `final`, or `simple2`.
- README files must explicitly distinguish between:
  - active development directories
  - archived reference-only directories

## Branch Policy

### Keep

- `main`
- `claude_method`
- explicitly approved `feature/*` branches that are still active

### Remove

- obsolete historical branches
- branches that duplicated experiments already preserved in commit history
- branches whose only purpose was to hold superseded directory variants

Branch deletion happens after the filesystem layout is stabilized and verified.

## Migration Sequence

1. Clean or isolate the current working tree so directory moves do not mix with unrelated edits.
2. Create `archive/` and `archive/wpa/`.
3. Move `compare/` into `archive/compare/`.
4. Move the current top-level MATLAB `wpa/` into `archive/wpa/matlab-original/`.
5. Move obsolete Python WPA directories into `archive/wpa/` with descriptive names.
6. Promote `wpa_python_port/claude_version_cct_rgb_gain_method` to the new top-level `wpa/`.
7. Update documentation, scripts, path references, and any helper commands that still point to the old deep path.
8. Add explicit repository rules to the root README and WPA README.
9. Prune stale branches using the approved keep-list.

## Documentation Rules

The root `README.md` should state:

- `wpa/` is the only active WPA directory
- `archive/` contains historical material
- `archive/wpa/*` and `archive/compare/` are reference-only

The new `wpa/README.md` should repeat:

- active development happens only in this directory
- archived WPA directories are not part of the normal workflow

## Risks

### Path breakage

The largest risk is not Git history loss. It is broken relative paths in:

- scripts
- tests
- docs
- dataset references
- local command snippets

Because of that, directory movement and path repair must be delivered together in one change set.

### Dirty working tree

The current repository already has unrelated in-progress changes. Migration work should be isolated so cleanup changes do not accidentally absorb unfinished algorithm edits.

### Branch deletion timing

Branch cleanup is low risk only after the directory migration is complete. Doing it earlier increases rollback confusion without reducing current operational pain.

## Non-Goals

- Do not split `CE` into separate repositories yet.
- Do not redesign all algorithm folders.
- Do not refactor archived implementations unless needed for migration safety.
- Do not optimize historical naming beyond what is needed to make archive contents understandable.

## Success Criteria

The cleanup is successful when all of the following are true:

- Entering `CE/` makes the active WPA location obvious.
- There is only one active WPA directory.
- Historical WPA and comparison material are clearly archived.
- The root documentation explains the rule unambiguously.
- Only the approved branch set remains.
