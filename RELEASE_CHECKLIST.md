# Release checklist

- [ ] `uv run ruff check .`
- [ ] `uv run pytest`
- [ ] `uv run kg verify --offline` passes: exactly the six documented read tools, and
      the `no-write-tools` negative control reports none exposed
- [ ] `uv run kg doctor` passes against a real endpoint, including the live
      embedding-width probe
- [ ] Ladybug disposable-database smoke test passes: `kg-ladybug-setup --apply` on a new
      file, then `kg ask` returns no facts and exits 0 before any ingest
- [ ] Ladybug coexistence: `kg ask` and `kg export` succeed while `graphiti-local` runs
      on the same file, `kg-ingest --apply` lands while the server is up, and the
      running server serves what landed
- [ ] `kg export` then `kg-ingest SNAPSHOT --restore --apply` into an empty database
      reproduces the facts
- [ ] FalkorDB smoke test passes when a test endpoint is available
- [ ] `python scripts/release_audit.py .` reports no findings
- [ ] `CHANGELOG.md` records every breaking change with its migration
- [ ] `pyproject.toml` version matches the tag being cut
- [ ] CI runs the same audit with `--allow-remote` because checkout configures `origin`
- [ ] `git log --format=fuller` contains share-safe author metadata
- [ ] `git remote -v` is empty until the owner chooses a destination
- [ ] Review the source archive and commit before creating a remote
- [ ] Create the GitHub repository as private first
- [ ] Make it public only after the owner explicitly approves the reviewed commit
