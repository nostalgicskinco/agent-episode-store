# agent-episode-store

Replayable episode ledger for AI agent runs. Part of [AIR Blackbox](https://github.com/airblackbox).

Every agent task becomes an **episode** — a complete, replayable record of every LLM call, tool invocation, and decision the agent made. Episodes are the dataset that makes evals, policy enforcement, and reproducible debugging possible.

## How It Fits Together

```
Agent / App
    |
    v
AIR Blackbox Gateway ──► records each LLM call as an AIR record
    |
    v
Episode Store ──► groups AIR records into complete task-level episodes
    |
    ├──► Eval Harness ──► replays episodes, scores regressions
    └──► Policy Engine ──► enforces rules, manages autonomy
```

## Features

**Sprint 1 — Core Ledger**
- Episode schema with Pydantic v2 validation
- SQLite storage with WAL mode (concurrent reads during writes)
- Ingest endpoint (`POST /v1/episodes`)

**Sprint 2 — Query, Replay, Diff, Export**
- Extended query filters: search by model, provider, or tool (`GET /v1/episodes`)
- Episode replay: re-indexed steps for deterministic re-execution (`GET /v1/episodes/{id}/replay`)
- Gateway webhook client for integration with AIR Blackbox Gateway
- Episode diff: step-by-step comparison for regression detection (`GET /v1/episodes/diff`)
- Streamlit dashboard: browse, filter, inspect, and compare episodes
- JSONL export: streaming export for offline analysis (`GET /v1/episodes/export`)

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/v1/episodes` | Ingest a new episode |
| GET | `/v1/episodes` | List episodes (filters: agent, status, model, provider, tool) |
| GET | `/v1/episodes/{id}` | Get full episode with steps |
| GET | `/v1/episodes/{id}/replay` | Replay-ready view (re-indexed, no timestamps) |
| GET | `/v1/episodes/diff?left={id}&right={id}` | Compare two episodes step-by-step |
| GET | `/v1/episodes/export` | Stream all episodes as JSONL |
| GET | `/v1/health` | Health check |

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Run
python -m app.server

# Dashboard
streamlit run dashboard.py
```

### Ingest an Episode

```bash
curl -X POST http://localhost:8100/v1/episodes \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "my-agent",
    "status": "success",
    "steps": [
      {"step_index": 0, "step_type": "llm_call", "model": "gpt-4", "tokens": 150},
      {"step_index": 1, "step_type": "tool_call", "tool_name": "web_search", "tokens": 200}
    ]
  }'
```

### Query with Filters

```bash
curl "http://localhost:8100/v1/episodes?model=gpt-4&status=success"
curl "http://localhost:8100/v1/episodes?tool=web_search"
```

## Testing

```bash
pytest -v   # 66 tests (27 API + 12 model + 27 storage)
```

## Configuration

| Env Var | Default | Description |
|---------|---------|-------------|
| `EPISODE_DB_PATH` | `episodes.db` | Path to SQLite database |
| `EPISODE_HOST` | `0.0.0.0` | Listen host |
| `EPISODE_PORT` | `8100` | Listen port |
| `EPISODE_STORE_URL` | `http://localhost:8100` | Dashboard connection URL |

## Roadmap

- [x] Episode schema & SQLite storage (EL-1, EL-2)
- [x] Ingest endpoint (EL-3)
- [x] Extended query filters — model, provider, tool (EL-4)
- [x] Episode replay endpoint (EL-5)
- [x] Gateway integration webhook (EL-6)
- [x] Episode diff utility (EL-7)
- [x] Streamlit dashboard (EL-8)
- [x] JSONL export (EL-9)
- [ ] → **Next:** Eval Harness consumes episodes for regression testing

## License

Apache-2.0
