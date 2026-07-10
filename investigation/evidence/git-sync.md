# Remote synchronization

Initial state:

```text
$ git status --short --branch
## main...origin/main
 ? tests/data/pleiades_data

$ git fetch --prune origin
60b4b56..19d0ea2  main -> origin/main

$ git rev-list --left-right --count HEAD...origin/main
0 37
```

Update:

```text
$ git merge --ff-only origin/main
Updating 60b4b56..19d0ea2
Fast-forward
```

The pre-existing nested `tests/data/pleiades_data` state was preserved.
Investigation work then moved to `codex/ic-calibration-research`.

Final sync verification after a second `git fetch --prune origin` immediately
before handoff:

```text
$ git rev-parse main origin/main
19d0ea28b967f6772a36025cb4184a3b2e149b0f
19d0ea28b967f6772a36025cb4184a3b2e149b0f

$ git rev-list --left-right --count main...origin/main
0 0

$ git log -1 --oneline main
19d0ea2 Release v0.3.0
```
