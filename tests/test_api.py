"""
Tests for episode store API endpoints.

Covers EL-3 (ingest), EL-4 (extended filters), EL-5 (replay),
EL-7 (diff), and EL-9 (JSONL export).
"""

from __future__ import annotations

import json

import pytest


def _make_episode_payload(
    agent_id: str = "test-agent",
    status: str = "success",
    steps: list | None = None,
    metadata: dict | None = None,
) -> dict:
    """Helper to build a valid episode payload."""
    if steps is None:
        steps = [
            {
                "step_index": 0,
                "step_type": "llm_call",
                "model": "gpt-4",
                "provider": "openai",
                "tokens": 150,
                "cost_usd": 0.005,
                "duration_ms": 800,
            },
            {
                "step_index": 1,
                "step_type": "tool_call",
                "tool_name": "web_search",
                "tokens": 200,
                "cost_usd": 0.006,
                "duration_ms": 1200,
            },
        ]
    return {
        "agent_id": agent_id,
        "status": status,
        "steps": steps,
        "metadata": metadata or {},
    }


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class TestHealth:
    @pytest.mark.asyncio
    async def test_health_check(self, client):
        """Health endpoint returns ok."""
        resp = await client.get("/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["service"] == "agent-episode-store"


# ---------------------------------------------------------------------------
# POST /v1/episodes
# ---------------------------------------------------------------------------

class TestIngest:
    @pytest.mark.asyncio
    async def test_create_episode(self, client):
        """POST creates an episode and returns 201."""
        payload = _make_episode_payload()
        resp = await client.post("/v1/episodes", json=payload)
        assert resp.status_code == 201

        data = resp.json()
        assert data["agent_id"] == "test-agent"
        assert data["status"] == "success"
        assert data["step_count"] == 2
        assert data["total_tokens"] == 350
        assert data["tools_used"] == ["web_search"]
        assert data["episode_id"] is not None

    @pytest.mark.asyncio
    async def test_create_minimal_episode(self, client):
        """POST with only agent_id works."""
        resp = await client.post("/v1/episodes", json={"agent_id": "minimal-agent"})
        assert resp.status_code == 201
        data = resp.json()
        assert data["agent_id"] == "minimal-agent"
        assert data["status"] == "running"
        assert data["step_count"] == 0

    @pytest.mark.asyncio
    async def test_create_rejects_empty_agent(self, client):
        """POST with empty agent_id returns 422."""
        resp = await client.post("/v1/episodes", json={"agent_id": ""})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_create_with_metadata(self, client):
        """POST preserves metadata."""
        payload = _make_episode_payload(metadata={"experiment": "v2", "model_version": "4o"})
        resp = await client.post("/v1/episodes", json=payload)
        assert resp.status_code == 201
        data = resp.json()
        assert data["metadata"]["experiment"] == "v2"

    @pytest.mark.asyncio
    async def test_create_computes_aggregates(self, client):
        """POST computes cost and duration from steps."""
        payload = _make_episode_payload()
        resp = await client.post("/v1/episodes", json=payload)
        data = resp.json()
        assert data["total_cost_usd"] == pytest.approx(0.011)
        assert data["total_duration_ms"] == 2000


# ---------------------------------------------------------------------------
# GET /v1/episodes/{id}
# ---------------------------------------------------------------------------

class TestGetEpisode:
    @pytest.mark.asyncio
    async def test_get_by_id(self, client):
        """GET returns the full episode with steps."""
        # Create first
        payload = _make_episode_payload()
        create_resp = await client.post("/v1/episodes", json=payload)
        ep_id = create_resp.json()["episode_id"]

        # Fetch
        resp = await client.get(f"/v1/episodes/{ep_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["episode_id"] == ep_id
        assert len(data["steps"]) == 2
        assert data["steps"][0]["step_type"] == "llm_call"
        assert data["steps"][1]["tool_name"] == "web_search"

    @pytest.mark.asyncio
    async def test_get_not_found(self, client):
        """GET returns 404 for nonexistent episode."""
        resp = await client.get("/v1/episodes/does-not-exist")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /v1/episodes
# ---------------------------------------------------------------------------

class TestListEpisodes:
    @pytest.mark.asyncio
    async def test_list_empty(self, client):
        """GET returns empty list when no episodes exist."""
        resp = await client.get("/v1/episodes")
        assert resp.status_code == 200
        assert resp.json() == []

    @pytest.mark.asyncio
    async def test_list_after_create(self, client):
        """GET returns episodes after creating them."""
        await client.post("/v1/episodes", json=_make_episode_payload(agent_id="a1"))
        await client.post("/v1/episodes", json=_make_episode_payload(agent_id="a2"))

        resp = await client.get("/v1/episodes")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        # Summaries should NOT include steps
        assert "steps" not in data[0]

    @pytest.mark.asyncio
    async def test_list_filter_by_agent(self, client):
        """GET filters by agent_id query param."""
        await client.post("/v1/episodes", json=_make_episode_payload(agent_id="alpha"))
        await client.post("/v1/episodes", json=_make_episode_payload(agent_id="beta"))
        await client.post("/v1/episodes", json=_make_episode_payload(agent_id="alpha"))

        resp = await client.get("/v1/episodes", params={"agent_id": "alpha"})
        data = resp.json()
        assert len(data) == 2
        assert all(d["agent_id"] == "alpha" for d in data)

    @pytest.mark.asyncio
    async def test_list_filter_by_status(self, client):
        """GET filters by status query param."""
        await client.post("/v1/episodes", json=_make_episode_payload(status="success"))
        await client.post("/v1/episodes", json=_make_episode_payload(status="failure"))

        resp = await client.get("/v1/episodes", params={"status": "failure"})
        data = resp.json()
        assert len(data) == 1
        assert data[0]["status"] == "failure"

    @pytest.mark.asyncio
    async def test_list_pagination(self, client):
        """GET supports limit and offset."""
        for i in range(5):
            await client.post("/v1/episodes", json=_make_episode_payload(agent_id=f"a{i}"))

        resp = await client.get("/v1/episodes", params={"limit": 2, "offset": 0})
        assert len(resp.json()) == 2

        resp2 = await client.get("/v1/episodes", params={"limit": 2, "offset": 2})
        assert len(resp2.json()) == 2


# ---------------------------------------------------------------------------
# EL-4: Extended filters (model, provider, tool)
# ---------------------------------------------------------------------------

class TestExtendedFilters:
    @pytest.mark.asyncio
    async def test_filter_by_model(self, client):
        """GET filters by model name inside steps JSON."""
        await client.post("/v1/episodes", json=_make_episode_payload())  # has gpt-4
        await client.post("/v1/episodes", json=_make_episode_payload(
            steps=[{"step_index": 0, "step_type": "llm_call", "model": "claude-3", "tokens": 100}]
        ))

        resp = await client.get("/v1/episodes", params={"model": "gpt-4"})
        data = resp.json()
        assert len(data) == 1

    @pytest.mark.asyncio
    async def test_filter_by_provider(self, client):
        """GET filters by provider inside steps JSON."""
        await client.post("/v1/episodes", json=_make_episode_payload())  # has openai
        await client.post("/v1/episodes", json=_make_episode_payload(
            steps=[{"step_index": 0, "step_type": "llm_call", "provider": "anthropic", "tokens": 100}]
        ))

        resp = await client.get("/v1/episodes", params={"provider": "anthropic"})
        data = resp.json()
        assert len(data) == 1

    @pytest.mark.asyncio
    async def test_filter_by_tool(self, client):
        """GET filters by tool name in tools_used."""
        await client.post("/v1/episodes", json=_make_episode_payload())  # has web_search
        await client.post("/v1/episodes", json=_make_episode_payload(
            steps=[{"step_index": 0, "step_type": "tool_call", "tool_name": "calculator", "tokens": 50}]
        ))

        resp = await client.get("/v1/episodes", params={"tool": "calculator"})
        data = resp.json()
        assert len(data) == 1
        assert "calculator" in data[0]["tools_used"]

    @pytest.mark.asyncio
    async def test_filter_no_match(self, client):
        """GET returns empty when filter matches nothing."""
        await client.post("/v1/episodes", json=_make_episode_payload())
        resp = await client.get("/v1/episodes", params={"model": "nonexistent-model"})
        assert resp.json() == []


# ---------------------------------------------------------------------------
# EL-5: Replay
# ---------------------------------------------------------------------------

class TestReplay:
    @pytest.mark.asyncio
    async def test_replay_endpoint(self, client):
        """GET /v1/episodes/{id}/replay returns replay-ready view."""
        create_resp = await client.post("/v1/episodes", json=_make_episode_payload())
        ep_id = create_resp.json()["episode_id"]

        resp = await client.get(f"/v1/episodes/{ep_id}/replay")
        assert resp.status_code == 200
        data = resp.json()
        assert data["episode_id"] == ep_id
        assert data["agent_id"] == "test-agent"
        assert data["original_status"] == "success"
        assert len(data["replay_steps"]) == 2
        assert data["replay_steps"][0]["replay_index"] == 0
        assert data["replay_steps"][1]["replay_index"] == 1
        assert data["tools_used"] == ["web_search"]

    @pytest.mark.asyncio
    async def test_replay_not_found(self, client):
        """GET /v1/episodes/{id}/replay returns 404 for missing episode."""
        resp = await client.get("/v1/episodes/nonexistent/replay")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_replay_strips_timestamps(self, client):
        """Replay steps should not contain timestamp fields."""
        create_resp = await client.post("/v1/episodes", json=_make_episode_payload())
        ep_id = create_resp.json()["episode_id"]

        resp = await client.get(f"/v1/episodes/{ep_id}/replay")
        step = resp.json()["replay_steps"][0]
        assert "timestamp" not in step


# ---------------------------------------------------------------------------
# EL-7: Diff
# ---------------------------------------------------------------------------

class TestDiff:
    @pytest.mark.asyncio
    async def test_diff_identical(self, client):
        """Diffing two identical episodes shows zero differences."""
        payload = _make_episode_payload()
        r1 = await client.post("/v1/episodes", json=payload)
        r2 = await client.post("/v1/episodes", json=payload)
        id1, id2 = r1.json()["episode_id"], r2.json()["episode_id"]

        resp = await client.get("/v1/episodes/diff", params={"left": id1, "right": id2})
        assert resp.status_code == 200
        data = resp.json()
        assert data["matching_steps"] == 2
        assert data["differing_steps"] == 0
        assert data["token_delta"] == 0
        assert data["cost_delta"] == 0.0
        assert data["step_diffs"] == []

    @pytest.mark.asyncio
    async def test_diff_different_models(self, client):
        """Diffing episodes with different models shows field diffs."""
        p1 = _make_episode_payload()
        p2 = _make_episode_payload(steps=[
            {"step_index": 0, "step_type": "llm_call", "model": "claude-3", "provider": "anthropic", "tokens": 150, "cost_usd": 0.005, "duration_ms": 800},
            {"step_index": 1, "step_type": "tool_call", "tool_name": "web_search", "tokens": 200, "cost_usd": 0.006, "duration_ms": 1200},
        ])
        r1 = await client.post("/v1/episodes", json=p1)
        r2 = await client.post("/v1/episodes", json=p2)
        id1, id2 = r1.json()["episode_id"], r2.json()["episode_id"]

        resp = await client.get("/v1/episodes/diff", params={"left": id1, "right": id2})
        data = resp.json()
        assert data["differing_steps"] >= 1
        # Should have model and provider diffs at step 0
        fields_changed = [d["field"] for d in data["step_diffs"]]
        assert "model" in fields_changed
        assert "provider" in fields_changed

    @pytest.mark.asyncio
    async def test_diff_different_step_counts(self, client):
        """Diffing episodes with different step counts tracks extras."""
        p1 = _make_episode_payload()  # 2 steps
        p2 = _make_episode_payload(steps=[
            {"step_index": 0, "step_type": "llm_call", "model": "gpt-4", "provider": "openai", "tokens": 150}
        ])  # 1 step
        r1 = await client.post("/v1/episodes", json=p1)
        r2 = await client.post("/v1/episodes", json=p2)
        id1, id2 = r1.json()["episode_id"], r2.json()["episode_id"]

        resp = await client.get("/v1/episodes/diff", params={"left": id1, "right": id2})
        data = resp.json()
        assert data["left_step_count"] == 2
        assert data["right_step_count"] == 1
        assert data["extra_left"] == 1

    @pytest.mark.asyncio
    async def test_diff_not_found(self, client):
        """Diff returns 404 when an episode doesn't exist."""
        resp = await client.get("/v1/episodes/diff", params={"left": "fake-1", "right": "fake-2"})
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# EL-9: JSONL Export
# ---------------------------------------------------------------------------

class TestExport:
    @pytest.mark.asyncio
    async def test_export_jsonl(self, client):
        """GET /v1/episodes/export returns JSONL stream."""
        await client.post("/v1/episodes", json=_make_episode_payload(agent_id="a1"))
        await client.post("/v1/episodes", json=_make_episode_payload(agent_id="a2"))

        resp = await client.get("/v1/episodes/export")
        assert resp.status_code == 200
        assert "application/x-ndjson" in resp.headers["content-type"]

        lines = resp.text.strip().split("\n")
        assert len(lines) == 2
        # Each line should be valid JSON
        for line in lines:
            parsed = json.loads(line)
            assert "episode_id" in parsed
            assert "steps" in parsed  # Full episodes, not summaries

    @pytest.mark.asyncio
    async def test_export_with_filter(self, client):
        """Export respects agent_id filter."""
        await client.post("/v1/episodes", json=_make_episode_payload(agent_id="export-a"))
        await client.post("/v1/episodes", json=_make_episode_payload(agent_id="export-b"))

        resp = await client.get("/v1/episodes/export", params={"agent_id": "export-a"})
        lines = resp.text.strip().split("\n")
        assert len(lines) == 1
        assert json.loads(lines[0])["agent_id"] == "export-a"

    @pytest.mark.asyncio
    async def test_export_empty(self, client):
        """Export returns empty body when no episodes match."""
        resp = await client.get("/v1/episodes/export")
        assert resp.status_code == 200
        assert resp.text.strip() == ""
