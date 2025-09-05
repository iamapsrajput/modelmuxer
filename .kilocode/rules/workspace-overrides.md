# Workspace Overrides – ModelMuxer

- Use `make bootstrap`, `make check`, `make dev`.
- Python: ruff, black, mypy --strict, pytest --cov >= 80%.
- TS/React: eslint, prettier, tsc --noEmit must pass.
- K8s: resources/limits required; no `latest` images.
- Router changes require unit tests + telemetry updates.
