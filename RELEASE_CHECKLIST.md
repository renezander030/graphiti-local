# Release checklist

- [ ] `uv run ruff check .`
- [ ] `uv run pytest`
- [ ] MCP registry contains exactly the six documented read tools
- [ ] Ladybug disposable-database smoke test passes
- [ ] FalkorDB smoke test passes when a test endpoint is available
- [ ] `python scripts/release_audit.py .` reports no findings
- [ ] CI runs the same audit with `--allow-remote` because checkout configures `origin`
- [ ] `git log --format=fuller` contains share-safe author metadata
- [ ] `git remote -v` is empty until the owner chooses a destination
- [ ] Review the source archive and commit before creating a remote
- [ ] Create the GitHub repository as private first
- [ ] Make it public only after the owner explicitly approves the reviewed commit
