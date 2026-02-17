"""
Tests for SQLite storage backend.

Covers EL-2 (WAL mode, save/get/list/count) plus
EL-4 (extended filters), EL-5 (replay), EL-7 (diff), EL-9 (export).
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from pkg.models import Episode, EpisodeCreate, EpisodeStep, EpisodeStatus, StepType


# ---------------------------------------------------------------------------
# WAL mode
# ---------------------------------------------------------------------------

class TestWALMode:
    @pytest.mark.asyncio
    async def test_wal_enabled(self, store):
        """Database should be in WAL journal mode."""
        cursor = await store._db.execute("PRAGMA journal_mode;")
        row = await cursor.fetchone()
        assert row[0] == "wal"


# ---------------------------------------------------------------------------
# Save & Get
# ---------------------------------------------------------------------------

class TestSaveAndGet:
    @pytest.mark.asyncio
    async def test_save_and_retrieve(self, store):
        """Save an episode and get it back by ID."""
        ep = Episode(
            agent_id="agent-1",
            status=EpisodeStatus.SUCCESS,
            steps=[
                EpisodeStep(step_index=0, step_type=StepType.LLM_CALL, tokens=100),
            ],
        )
        saved = await store.save(ep)
        assert saved.step_count == 1
        assert saved.total_tokens == 100

        fetched = await store.get(saved.episode_id)
        assert fetched is not None
        assert fetched.episode_id == saved.episode_id
        assert fetched.agent_id == "agent-1"
        assert len(fetched.steps) == 1

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, store):
        """Getting a nonexistent episode returns None."""
        result = await store.get("does-not-exist")
        assert result is None

    @pytest.mark.asyncio
    async def test_create_from_payload(self, store):
        """Create an episode from an EpisodeCreate payload."""
        payload = EpisodeCreate(
            agent_id="agent-2",
            steps=[
                EpisodeStep(
                    step_index=0, step_type=StepType.TOOL_CALL,
                    tool_name="web_search", tokens=200, cost_usd=0.006,
                ),
            ],
            status=EpisodeStatus.SUCCESS,
            metadata={"experiment": "v1"},
        )
        ep = await store.create(payload)
        assert ep.episode_id is not None
        assert ep.agent_id == "agent-2"
        assert ep.total_tokens == 200
        assert ep.tools_used == ["web_search"]
        assert ep.metadata == {"experiment": "v1"}

    @pytest.mark.asyncio
    async def test_steps_roundtrip(self, store):
        """Steps survive JSON serialization/deserialization."""
        ep = Episode(
            agent_id="agent-3",
            steps=[
                EpisodeStep(
                    step_index=0, step_type=StepType.LLM_CALL,
                    model="gpt-4", provider="openai",
                    input_summary="Hello", output_summary="Hi there",
                    tokens=50, cost_usd=0.001, duration_ms=300,
                    metadata={"temperature": 0.7},
                ),
                EpisodeStep(
                    step_index=1, step_type=StepType.TOOL_CALL,
                    tool_name="calculator",
                    tokens=10, duration_ms=50,
                ),
            ],
        )
        saved = await store.save(ep)
        fetched = await store.get(saved.episode_id)

        assert len(fetched.steps) == 2
        assert fetched.steps[0].model == "gpt-4"
        assert fetched.steps[0].metadata == {"temperature": 0.7}
        assert fetched.steps[1].tool_name == "calculator"


# ---------------------------------------------------------------------------
# List & Count
# ---------------------------------------------------------------------------

class TestListAndCount:
    @pytest.mark.asyncio
    async def test_list_empty(self, store):
        """List returns empty when no episodes exist."""
        results = await store.list()
        assert results == []

    @pytest.mark.asyncio
    async def test_list_with_agent_filter(self, store):
        """List filters by agent_id."""
        await store.create(EpisodeCreate(agent_id="agent-a", status=EpisodeStatus.SUCCESS))
        await store.create(EpisodeCreate(agent_id="agent-b", status=EpisodeStatus.SUCCESS))
        await store.create(EpisodeCreate(agent_id="agent-a", status=EpisodeStatus.FAILURE))

        results = await store.list(agent_id="agent-a")
        assert len(results) == 2
        assert all(r.agent_id == "agent-a" for r in results)

    @pytest.mark.asyncio
    async def test_list_with_status_filter(self, store):
        """List filters by status."""
        await store.create(EpisodeCreate(agent_id="agent-1", status=EpisodeStatus.SUCCESS))
        await store.create(EpisodeCreate(agent_id="agent-1", status=EpisodeStatus.FAILURE))

        results = await store.list(status="success")
        assert len(results) == 1
        assert results[0].status == EpisodeStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_list_pagination(self, store):
        """Limit and offset work correctly."""
        for i in range(5):
            await store.create(EpisodeCreate(agent_id=f"agent-{i}", status=EpisodeStatus.SUCCESS))

        page1 = await store.list(limit=2, offset=0)
        page2 = await store.list(limit=2, offset=2)

        assert len(page1) == 2
        assert len(page2) == 2
        assert page1[0].episode_id != page2[0].episode_id

    @pytest.mark.asyncio
    async def test_count(self, store):
        """Count returns correct totals."""
        assert await store.count() == 0

        await store.create(EpisodeCreate(agent_id="agent-1", status=EpisodeStatus.SUCCESS))
        await store.create(EpisodeCreate(agent_id="agent-2", status=EpisodeStatus.FAILURE))

        assert await store.count() == 2
        assert await store.count(agent_id="agent-1") == 1
        assert await store.count(status="failure") == 1


# ---------------------------------------------------------------------------
# EL-4: Extended filters (model, provider, tool)
# ---------------------------------------------------------------------------

class TestExtendedFilters:
    @pytest.mark.asyncio
    async def test_filter_by_model(self, store):
        """List filters by model name inside steps JSON."""
        await store.create(EpisodeCreate(
            agent_id="a1", status=EpisodeStatus.SUCCESS,
            steps=[EpisodeStep(step_index=0, step_type=StepType.LLM_CALL, model="gpt-4", tokens=100)],
        ))
        await store.create(EpisodeCreate(
            agent_id="a2", status=EpisodeStatus.SUCCESS,
            steps=[EpisodeStep(step_index=0, step_type=StepType.LLM_CALL, model="claude-3", tokens=100)],
        ))

        results = await store.list(model="gpt-4")
        assert len(results) == 1
        assert results[0].agent_id == "a1"

    @pytest.mark.asyncio
    async def test_filter_by_provider(self, store):
        """List filters by provider inside steps JSON."""
        await store.create(EpisodeCreate(
            agent_id="a1", status=EpisodeStatus.SUCCESS,
            steps=[EpisodeStep(step_index=0, step_type=StepType.LLM_CALL, provider="openai", tokens=100)],
        ))
        await store.create(EpisodeCreate(
            agent_id="a2", status=EpisodeStatus.SUCCESS,
            steps=[EpisodeStep(step_index=0, step_type=StepType.LLM_CALL, provider="anthropic", tokens=100)],
        ))

        results = await store.list(provider="anthropic")
        assert len(results) == 1
        assert results[0].agent_id == "a2"

    @pytest.mark.asyncio
    async def test_filter_by_tool(self, store):
        """List filters by tool name in tools_used."""
        await store.create(EpisodeCreate(
            agent_id="a1", status=EpisodeStatus.SUCCESS,
            steps=[EpisodeStep(step_index=0, step_type=StepType.TOOL_CALL, tool_name="web_search", tokens=50)],
        ))
        await store.create(EpisodeCreate(
            agent_id="a2", status=EpisodeStatus.SUCCESS,
            steps=[EpisodeStep(step_index=0, step_type=StepType.TOOL_CALL, tool_name="calculator", tokens=50)],
        ))

        results = await store.list(tool="calculator")
        assert len(results) == 1
        assert results[0].agent_id == "a2"

    @pytest.mark.asyncio
    async def test_combined_filters(self, store):
        """Multiple filters combine with AND logic."""
        await store.create(EpisodeCreate(
            agent_id="a1", status=EpisodeStatus.SUCCESS,
            steps=[EpisodeStep(step_index=0, step_type=StepType.LLM_CALL, model="gpt-4", provider="openai", tokens=100)],
        ))
        await store.create(EpisodeCreate(
            agent_id="a2", status=EpisodeStatus.FAILURE,
            steps=[EpisodeStep(step_index=0, step_type=StepType.LLM_CALL, model="gpt-4", provider="openai", tokens=100)],
        ))

        results = await store.list(model="gpt-4", status="success")
        assert len(results) == 1
        assert results[0].agent_id == "a1"

    @pytest.mark.asyncio
    async def test_filter_no_match(self, store):
        """Returns empty list when no episodes match filter."""
        await store.create(EpisodeCreate(
            agent_id="a1", status=EpisodeStatus.SUCCESS,
            steps=[EpisodeStep(step_index=0, step_type=StepType.LLM_CALL, model="gpt-4", tokens=100)],
        ))

        results = await store.list(model="nonexistent")
        assert results == []


# ---------------------------------------------------------------------------
# EL-5: Replay
# ---------------------------------------------------------------------------

class TestReplay:
    @pytest.mark.asyncio
    async def test_replay_returns_correct_structure(self, store):
        """Replay returns episode with replay-indexed steps."""
        ep = await store.create(EpisodeCreate(
            agent_id="replay-agent", status=EpisodeStatus.SUCCESS,
            steps=[
                EpisodeStep(step_index=0, step_type=StepType.LLM_CALL, model="gpt-4", tokens=100),
                EpisodeStep(step_index=1, step_type=StepType.TOOL_CALL, tool_name="web_search", tokens=50),
            ],
        ))

        replay = await store.get_replay(ep.episode_id)
        assert replay is not None
        assert replay.episode_id == ep.episode_id
        assert replay.agent_id == "replay-agent"
        assert replay.original_status == EpisodeStatus.SUCCESS
        assert len(replay.replay_steps) == 2
        assert replay.replay_steps[0].replay_index == 0
        assert replay.replay_steps[1].replay_index == 1
        assert replay.replay_steps[1].tool_name == "web_search"

    @pytest.mark.asyncio
    async def test_replay_nonexistent(self, store):
        """Replay of nonexistent episode returns None."""
        result = await store.get_replay("does-not-exist")
        assert result is None

    @pytest.mark.asyncio
    async def test_replay_preserves_fields(self, store):
        """Replay carries over model, provider, and token data."""
        ep = await store.create(EpisodeCreate(
            agent_id="agent-rp", status=EpisodeStatus.SUCCESS,
            steps=[
                EpisodeStep(
                    step_index=0, step_type=StepType.LLM_CALL,
                    model="gpt-4", provider="openai",
                    tokens=200, cost_usd=0.01, duration_ms=500,
                ),
            ],
        ))

        replay = await store.get_replay(ep.episode_id)
        step = replay.replay_steps[0]
        assert step.model == "gpt-4"
        assert step.provider == "openai"
        assert step.tokens == 200
        assert step.cost_usd == 0.01


# ---------------------------------------------------------------------------
# EL-7: Diff
# ---------------------------------------------------------------------------

class TestDiff:
    @pytest.mark.asyncio
    async def test_diff_identical(self, store):
        """Diffing identical episodes shows zero differences."""
        steps = [
            EpisodeStep(step_index=0, step_type=StepType.LLM_CALL, model="gpt-4", tokens=100),
            EpisodeStep(step_index=1, step_type=StepType.TOOL_CALL, tool_name="web_search", tokens=50),
        ]
        ep1 = await store.create(EpisodeCreate(agent_id="a", status=EpisodeStatus.SUCCESS, steps=steps))
        ep2 = await store.create(EpisodeCreate(agent_id="a", status=EpisodeStatus.SUCCESS, steps=steps))

        diff = await store.diff(ep1.episode_id, ep2.episode_id)
        assert diff is not None
        assert diff.matching_steps == 2
        assert diff.differing_steps == 0
        assert diff.token_delta == 0
        assert diff.step_diffs == []

    @pytest.mark.asyncio
    async def test_diff_different_models(self, store):
        """Diffing episodes with different models shows field diffs."""
        ep1 = await store.create(EpisodeCreate(
            agent_id="a", status=EpisodeStatus.SUCCESS,
            steps=[EpisodeStep(step_index=0, step_type=StepType.LLM_CALL, model="gpt-4", provider="openai", tokens=100)],
        ))
        ep2 = await store.create(EpisodeCreate(
            agent_id="a", status=EpisodeStatus.SUCCESS,
            steps=[EpisodeStep(step_index=0, step_type=StepType.LLM_CALL, model="claude-3", provider="anthropic", tokens=100)],
        ))

        diff = await store.diff(ep1.episode_id, ep2.episode_id)
        assert diff.differing_steps == 1
        fields = [d.field for d in diff.step_diffs]
        assert "model" in fields
        assert "provider" in fields

    @pytest.mark.asyncio
    async def test_diff_different_step_counts(self, store):
        """Diff tracks extra steps when counts differ."""
        ep1 = await store.create(EpisodeCreate(
            agent_id="a", status=EpisodeStatus.SUCCESS,
            steps=[
                EpisodeStep(step_index=0, step_type=StepType.LLM_CALL, model="gpt-4", tokens=100),
                EpisodeStep(step_index=1, step_type=StepType.TOOL_CALL, tool_name="calc", tokens=50),
            ],
        ))
        ep2 = await store.create(EpisodeCreate(
            agent_id="a", status=EpisodeStatus.SUCCESS,
            steps=[EpisodeStep(step_index=0, step_type=StepType.LLM_CALL, model="gpt-4", tokens=100)],
        ))

        diff = await store.diff(ep1.episode_id, ep2.episode_id)
        assert diff.left_step_count == 2
        assert diff.right_step_count == 1
        assert diff.extra_left == 1
        assert diff.extra_right == 0

    @pytest.mark.asyncio
    async def test_diff_nonexistent(self, store):
        """Diff returns None when either episode doesn't exist."""
        result = await store.diff("fake-1", "fake-2")
        assert result is None

    @pytest.mark.asyncio
    async def test_diff_token_delta(self, store):
        """Diff calculates correct token delta."""
        ep1 = await store.create(EpisodeCreate(
            agent_id="a", status=EpisodeStatus.SUCCESS,
            steps=[EpisodeStep(step_index=0, step_type=StepType.LLM_CALL, tokens=100)],
        ))
        ep2 = await store.create(EpisodeCreate(
            agent_id="a", status=EpisodeStatus.SUCCESS,
            steps=[EpisodeStep(step_index=0, step_type=StepType.LLM_CALL, tokens=300)],
        ))

        diff = await store.diff(ep1.episode_id, ep2.episode_id)
        assert diff.token_delta == 200


# ---------------------------------------------------------------------------
# EL-9: Export JSONL
# ---------------------------------------------------------------------------

class TestExportJSONL:
    @pytest.mark.asyncio
    async def test_export_all(self, store):
        """Export returns all episodes as dicts."""
        await store.create(EpisodeCreate(agent_id="a1", status=EpisodeStatus.SUCCESS))
        await store.create(EpisodeCreate(agent_id="a2", status=EpisodeStatus.SUCCESS))

        rows = await store.export_jsonl()
        assert len(rows) == 2
        assert all("episode_id" in r for r in rows)
        assert all("steps" in r for r in rows)

    @pytest.mark.asyncio
    async def test_export_with_filter(self, store):
        """Export respects agent_id filter."""
        await store.create(EpisodeCreate(agent_id="x", status=EpisodeStatus.SUCCESS))
        await store.create(EpisodeCreate(agent_id="y", status=EpisodeStatus.SUCCESS))

        rows = await store.export_jsonl(agent_id="x")
        assert len(rows) == 1
        assert rows[0]["agent_id"] == "x"

    @pytest.mark.asyncio
    async def test_export_empty(self, store):
        """Export returns empty list when no episodes exist."""
        rows = await store.export_jsonl()
        assert rows == []

    @pytest.mark.asyncio
    async def test_export_includes_steps(self, store):
        """Exported episodes include full step data."""
        await store.create(EpisodeCreate(
            agent_id="a1", status=EpisodeStatus.SUCCESS,
            steps=[
                EpisodeStep(step_index=0, step_type=StepType.LLM_CALL, model="gpt-4", tokens=100),
            ],
        ))

        rows = await store.export_jsonl()
        assert len(rows[0]["steps"]) == 1
        assert rows[0]["steps"][0]["model"] == "gpt-4"
