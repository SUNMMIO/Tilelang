# Project Workflow Rules

## Preferred Dev/Test Workflow

When working in this repository, use this workflow by default:

1. Edit and implement changes locally in `/Users/chunfeng/Tilelang-Mesh`.
2. Commit only relevant files for the current task.
3. Push changes to remote `serve` branch `tilelang_mesh_main` by default.
4. SSH to `liuchunfeng@123.127.250.154` on port `9104`.
5. Use remote repo path `/home/liuchunfeng/Tilelang-Mesh-Sync-Pc` for build and test.
6. Use conda env `tl` for build/test commands on remote.
7. Configure/build with:
   - `~/.local/bin/cmake -S . -B build -DTILELANG_UPDATE_SUBMODULES=OFF`
   - `~/.local/bin/cmake --build build -j8`
8. Run smoke checks remotely after build:
   - verify symbols in `build/lib/libtilelang.so` when relevant
   - run Python syntax/smoke checks in conda env `tl`

## Git/Remote Notes

- Remote names in local repo:
  - `origin`: GitHub
  - `serve`: `ssh://liuchunfeng@123.127.250.154:9104/home/liuchunfeng/Tilelang-Mesh-Sync-Pc`
- `serve/tilelang_mesh_main` is a sync branch between local and server.
- Before pushing to `serve/tilelang_mesh_main`, check remote dirtiness with:
  - `ssh -p 9104 liuchunfeng@123.127.250.154 'cd /home/liuchunfeng/Tilelang-Mesh-Sync-Pc && git status --short'`
- If remote is dirty, do not discard unknown edits. Prefer to commit or stash remote WIP first, then push.

## Cross-Window Handoff

- Use `docs/SESSION_HANDOFF.md` as the canonical context transfer note between Trae windows/sessions.
- After meaningful milestones, refresh `docs/SESSION_HANDOFF.md` with:
  - branch + commit
  - key files changed
  - exact verification commands
  - latest test result summary
