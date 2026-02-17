"""
Episode API routes.

Endpoints:
    POST /v1/episodes              — Ingest a new episode (EL-3)
    GET  /v1/episodes              — List episodes with filters (EL-3 + EL-4)
    GET  /v1/episodes/{id}         — Get a single episode with all steps
    GET  /v1/episodes/{id}/replay  — Get replay-ready view (EL-5)
    GET  /v1/episodes/diff         — Compare two episodes (EL-7)
    GET  /v1/episodes/export       — Export episodes as JSONL (EL-9)
    GET  /v1/health                — Health check
"""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from pkg.models import Episode, EpisodeCreate, EpisodeDiff, EpisodeReplay, EpisodeSummary
from pkg.storage import EpisodeStore

import json

router = APIRouter()

# The store gets attached at startup (see app/server.py)
_store: EpisodeStore | None = None


def set_store(store: EpisodeStore) -> None:
    """Called at app startup to inject the store dependency."""
    global _store
    _store = store


def get_store() -> EpisodeStore:
    """Get the store, raising if not initialized."""
    if _store is None:
        raise RuntimeError("Store not initialized")
    return _store


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@router.get("/v1/health")
async def health() -> dict:
    """Health check — returns ok if the store is connected."""
    store = get_store()
    count = await store.count()
    return {
        "status": "ok",
        "service": "agent-episode-store",
        "version": "0.2.0",
        "episodes_stored": count,
    }


# ---------------------------------------------------------------------------
# Ingest (EL-3)
# ---------------------------------------------------------------------------

@router.post("/v1/episodes", status_code=201, response_model=Episode)
async def create_episode(payload: EpisodeCreate) -> Episode:
    """Ingest a new episode.

    This is the main entry point for the gateway webhook.
    The gateway sends a complete episode (agent_id, steps, status)
    and the store assigns an episode_id and computes aggregates.
    """
    store = get_store()
    episode = await store.create(payload)
    return episode


# ---------------------------------------------------------------------------
# Export (EL-9) — must be BEFORE /{episode_id} to avoid path conflict
# ---------------------------------------------------------------------------

@router.get("/v1/episodes/export")
async def export_episodes(
    agent_id: str | None = Query(default=None, description="Filter by agent"),
    status: str | None = Query(default=None, description="Filter by status"),
    since: datetime | None = Query(default=None, description="Episodes after this time"),
    until: datetime | None = Query(default=None, description="Episodes before this time"),
) -> StreamingResponse:
    """Export episodes as JSONL (one JSON object per line).

    Suitable for piping into offline analysis tools, feeding into
    the eval harness, or archiving to S3.
    """
    store = get_store()
    episodes = await store.export_jsonl(
        agent_id=agent_id, status=status, since=since, until=until,
    )

    def generate():
        for ep in episodes:
            yield json.dumps(ep, default=str) + "\n"

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
        headers={"Content-Disposition": "attachment; filename=episodes.jsonl"},
    )


# ---------------------------------------------------------------------------
# Diff (EL-7) — must be BEFORE /{episode_id} to avoid path conflict
# ---------------------------------------------------------------------------

@router.get("/v1/episodes/diff", response_model=EpisodeDiff)
async def diff_episodes(
    left: str = Query(..., description="Baseline episode ID"),
    right: str = Query(..., description="Comparison episode ID"),
) -> EpisodeDiff:
    """Compare two episodes step-by-step.

    Shows which steps differ, plus deltas for tokens, cost, and duration.
    Useful for regression detection and A/B testing agent behavior.
    """
    store = get_store()
    result = await store.diff(left, right)
    if result is None:
        raise HTTPException(status_code=404, detail="One or both episodes not found")
    return result


# ---------------------------------------------------------------------------
# Query (EL-3 + EL-4)
# ---------------------------------------------------------------------------

@router.get("/v1/episodes", response_model=list[EpisodeSummary])
async def list_episodes(
    agent_id: str | None = Query(default=None, description="Filter by agent"),
    status: str | None = Query(default=None, description="Filter by status"),
    since: datetime | None = Query(default=None, description="Episodes after this time"),
    until: datetime | None = Query(default=None, description="Episodes before this time"),
    model: str | None = Query(default=None, description="Filter by LLM model used (EL-4)"),
    provider: str | None = Query(default=None, description="Filter by provider (EL-4)"),
    tool: str | None = Query(default=None, description="Filter by tool used (EL-4)"),
    limit: int = Query(default=50, ge=1, le=500, description="Max results"),
    offset: int = Query(default=0, ge=0, description="Pagination offset"),
) -> list[EpisodeSummary]:
    """List episodes with optional filters.

    EL-4 adds model, provider, and tool filters on top of the
    original agent_id, status, and date range filters.
    """
    store = get_store()
    return await store.list(
        agent_id=agent_id,
        status=status,
        since=since,
        until=until,
        model=model,
        provider=provider,
        tool=tool,
        limit=limit,
        offset=offset,
    )


@router.get("/v1/episodes/{episode_id}", response_model=Episode)
async def get_episode(episode_id: str) -> Episode:
    """Get a single episode by ID, including all steps."""
    store = get_store()
    episode = await store.get(episode_id)
    if episode is None:
        raise HTTPException(status_code=404, detail="Episode not found")
    return episode


# ---------------------------------------------------------------------------
# Replay (EL-5)
# ---------------------------------------------------------------------------

@router.get("/v1/episodes/{episode_id}/replay", response_model=EpisodeReplay)
async def replay_episode(episode_id: str) -> EpisodeReplay:
    """Get a replay-ready view of an episode.

    Strips timestamps and re-indexes steps so they can be fed
    sequentially back through the gateway for eval purposes.
    """
    store = get_store()
    replay = await store.get_replay(episode_id)
    if replay is None:
        raise HTTPException(status_code=404, detail="Episode not found")
    return replay
