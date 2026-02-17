"""
SQLite storage backend for episodes.

EL-2: SQLite storage backend with WAL mode

Uses aiosqlite for async access. WAL (Write-Ahead Logging) mode allows
concurrent reads while a write is happening — important for a service
that ingests episodes while dashboards query them.

The schema stores episodes as a row with JSON columns for steps and
metadata. This keeps the storage simple (one table) while still
letting us query by agent_id, status, and date range.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite

from pkg.models import (
    Episode,
    EpisodeCreate,
    EpisodeDiff,
    EpisodeReplay,
    EpisodeStep,
    EpisodeStatus,
    EpisodeSummary,
    ReplayStep,
    StepDiff,
)

# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS episodes (
    episode_id     TEXT PRIMARY KEY,
    agent_id       TEXT NOT NULL,
    status         TEXT NOT NULL DEFAULT 'running',
    steps          TEXT NOT NULL DEFAULT '[]',
    tools_used     TEXT NOT NULL DEFAULT '[]',
    total_tokens   INTEGER NOT NULL DEFAULT 0,
    total_cost_usd REAL NOT NULL DEFAULT 0.0,
    total_duration_ms INTEGER NOT NULL DEFAULT 0,
    step_count     INTEGER NOT NULL DEFAULT 0,
    started_at     TEXT NOT NULL,
    ended_at       TEXT,
    metadata       TEXT NOT NULL DEFAULT '{}',
    created_at     TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_episodes_agent ON episodes(agent_id);",
    "CREATE INDEX IF NOT EXISTS idx_episodes_status ON episodes(status);",
    "CREATE INDEX IF NOT EXISTS idx_episodes_started ON episodes(started_at);",
]

_INSERT = """
INSERT INTO episodes (
    episode_id, agent_id, status, steps, tools_used,
    total_tokens, total_cost_usd, total_duration_ms, step_count,
    started_at, ended_at, metadata
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
"""

_SELECT_BY_ID = "SELECT * FROM episodes WHERE episode_id = ?;"

_SELECT_LIST = """
SELECT episode_id, agent_id, status, tools_used,
       total_tokens, total_cost_usd, total_duration_ms, step_count,
       started_at, ended_at
FROM episodes
WHERE 1=1
"""

_COUNT = "SELECT COUNT(*) FROM episodes WHERE 1=1"


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class EpisodeStore:
    """Async SQLite-backed episode storage.

    Usage:
        store = EpisodeStore("episodes.db")
        await store.init()           # creates tables
        await store.save(episode)    # insert
        ep = await store.get("id")   # fetch one
        results = await store.list() # fetch many
        await store.close()
    """

    def __init__(self, db_path: str = "episodes.db") -> None:
        self.db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def init(self) -> None:
        """Open database, enable WAL mode, create tables."""
        # Ensure parent directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row

        # WAL mode: allows concurrent reads during writes
        await self._db.execute("PRAGMA journal_mode=WAL;")
        # Sync mode NORMAL is safe with WAL and faster than FULL
        await self._db.execute("PRAGMA synchronous=NORMAL;")

        await self._db.execute(_CREATE_TABLE)
        for idx_sql in _CREATE_INDEXES:
            await self._db.execute(idx_sql)
        await self._db.commit()

    async def close(self) -> None:
        """Close the database connection."""
        if self._db:
            await self._db.close()
            self._db = None

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    async def save(self, episode: Episode) -> Episode:
        """Insert an episode into the database.

        Computes aggregates from steps before saving.
        Returns the episode with computed fields.
        """
        assert self._db is not None, "Call init() first"

        episode.compute_aggregates()

        steps_json = json.dumps(
            [s.model_dump(mode="json") for s in episode.steps]
        )
        tools_json = json.dumps(episode.tools_used)
        meta_json = json.dumps(episode.metadata)
        started = episode.started_at.isoformat()
        ended = episode.ended_at.isoformat() if episode.ended_at else None

        await self._db.execute(
            _INSERT,
            (
                episode.episode_id,
                episode.agent_id,
                episode.status.value,
                steps_json,
                tools_json,
                episode.total_tokens,
                episode.total_cost_usd,
                episode.total_duration_ms,
                episode.step_count,
                started,
                ended,
                meta_json,
            ),
        )
        await self._db.commit()
        return episode

    async def create(self, payload: EpisodeCreate) -> Episode:
        """Create a new episode from an ingest payload.

        Assigns episode_id and timestamps, then saves.
        """
        now = datetime.now(timezone.utc)
        ended = now if payload.status != "running" else None

        episode = Episode(
            agent_id=payload.agent_id,
            status=payload.status,
            steps=payload.steps,
            started_at=now,
            ended_at=ended,
            metadata=payload.metadata,
        )
        return await self.save(episode)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    async def get(self, episode_id: str) -> Episode | None:
        """Fetch a single episode by ID. Returns None if not found."""
        assert self._db is not None, "Call init() first"

        cursor = await self._db.execute(_SELECT_BY_ID, (episode_id,))
        row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_episode(row)

    async def list(
        self,
        agent_id: str | None = None,
        status: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        model: str | None = None,
        provider: str | None = None,
        tool: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[EpisodeSummary]:
        """List episodes with optional filters.

        EL-4: Extended filters — model, provider, tool (searched inside
        the JSON steps column using SQLite JSON/LIKE).
        Returns lightweight summaries (no steps) for performance.
        """
        assert self._db is not None, "Call init() first"

        where_clauses: list[str] = []
        params: list[str | int] = []

        if agent_id:
            where_clauses.append("AND agent_id = ?")
            params.append(agent_id)
        if status:
            where_clauses.append("AND status = ?")
            params.append(status)
        if since:
            where_clauses.append("AND started_at >= ?")
            params.append(since.isoformat())
        if until:
            where_clauses.append("AND started_at <= ?")
            params.append(until.isoformat())
        if model:
            where_clauses.append("AND steps LIKE ?")
            params.append(f'%"model": "{model}"%')
        if provider:
            where_clauses.append("AND steps LIKE ?")
            params.append(f'%"provider": "{provider}"%')
        if tool:
            where_clauses.append("AND tools_used LIKE ?")
            params.append(f'%"{tool}"%')

        query = (
            _SELECT_LIST
            + " ".join(where_clauses)
            + " ORDER BY started_at DESC LIMIT ? OFFSET ?"
        )
        params.extend([limit, offset])

        cursor = await self._db.execute(query, params)
        rows = await cursor.fetchall()
        return [self._row_to_summary(r) for r in rows]

    async def count(
        self,
        agent_id: str | None = None,
        status: str | None = None,
    ) -> int:
        """Count episodes matching filters."""
        assert self._db is not None, "Call init() first"

        where_clauses: list[str] = []
        params: list[str] = []

        if agent_id:
            where_clauses.append("AND agent_id = ?")
            params.append(agent_id)
        if status:
            where_clauses.append("AND status = ?")
            params.append(status)

        query = _COUNT + " " + " ".join(where_clauses)
        cursor = await self._db.execute(query, params)
        row = await cursor.fetchone()
        return row[0] if row else 0

    # ------------------------------------------------------------------
    # Replay (EL-5)
    # ------------------------------------------------------------------

    async def get_replay(self, episode_id: str) -> EpisodeReplay | None:
        """Build a replay-ready view of an episode.

        Strips timestamps and re-indexes steps for sequential replay
        through the gateway.
        """
        episode = await self.get(episode_id)
        if episode is None:
            return None

        replay_steps = [
            ReplayStep(
                replay_index=i,
                step_type=s.step_type,
                tool_name=s.tool_name,
                model=s.model,
                provider=s.provider,
                input_summary=s.input_summary,
                output_summary=s.output_summary,
                tokens=s.tokens,
                cost_usd=s.cost_usd,
                duration_ms=s.duration_ms,
                metadata=s.metadata,
            )
            for i, s in enumerate(episode.steps)
        ]

        return EpisodeReplay(
            episode_id=episode.episode_id,
            agent_id=episode.agent_id,
            original_status=episode.status,
            replay_steps=replay_steps,
            total_tokens=episode.total_tokens,
            total_cost_usd=episode.total_cost_usd,
            tools_used=episode.tools_used,
        )

    # ------------------------------------------------------------------
    # Diff (EL-7)
    # ------------------------------------------------------------------

    async def diff(self, left_id: str, right_id: str) -> EpisodeDiff | None:
        """Compare two episodes step-by-step.

        Returns a diff showing which steps changed between the left
        (baseline) and right (comparison) episodes.
        Returns None if either episode doesn't exist.
        """
        left = await self.get(left_id)
        right = await self.get(right_id)
        if left is None or right is None:
            return None

        step_diffs: list[StepDiff] = []
        matching = 0
        differing = 0

        compare_fields = [
            "step_type", "tool_name", "model", "provider",
            "input_summary", "output_summary",
        ]

        min_len = min(len(left.steps), len(right.steps))
        for i in range(min_len):
            ls, rs = left.steps[i], right.steps[i]
            step_has_diff = False
            for field in compare_fields:
                lv = str(getattr(ls, field))
                rv = str(getattr(rs, field))
                if lv != rv:
                    step_diffs.append(StepDiff(
                        step_index=i, field=field, left=lv, right=rv,
                    ))
                    step_has_diff = True
            if step_has_diff:
                differing += 1
            else:
                matching += 1

        return EpisodeDiff(
            left_episode_id=left_id,
            right_episode_id=right_id,
            left_step_count=len(left.steps),
            right_step_count=len(right.steps),
            matching_steps=matching,
            differing_steps=differing,
            extra_left=max(0, len(left.steps) - len(right.steps)),
            extra_right=max(0, len(right.steps) - len(left.steps)),
            token_delta=right.total_tokens - left.total_tokens,
            cost_delta=round(right.total_cost_usd - left.total_cost_usd, 6),
            duration_delta=right.total_duration_ms - left.total_duration_ms,
            step_diffs=step_diffs,
        )

    # ------------------------------------------------------------------
    # Export (EL-9)
    # ------------------------------------------------------------------

    async def export_jsonl(
        self,
        agent_id: str | None = None,
        status: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> list[dict]:
        """Export episodes as a list of dicts (for JSONL serialization).

        Each dict is a complete episode with steps — suitable for
        writing one-per-line to a .jsonl file for offline analysis.
        """
        assert self._db is not None, "Call init() first"

        where_clauses: list[str] = []
        params: list[str | int] = []

        if agent_id:
            where_clauses.append("AND agent_id = ?")
            params.append(agent_id)
        if status:
            where_clauses.append("AND status = ?")
            params.append(status)
        if since:
            where_clauses.append("AND started_at >= ?")
            params.append(since.isoformat())
        if until:
            where_clauses.append("AND started_at <= ?")
            params.append(until.isoformat())

        query = (
            "SELECT * FROM episodes WHERE 1=1 "
            + " ".join(where_clauses)
            + " ORDER BY started_at DESC"
        )
        cursor = await self._db.execute(query, params)
        rows = await cursor.fetchall()
        return [self._row_to_episode(r).model_dump(mode="json") for r in rows]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_episode(row: aiosqlite.Row) -> Episode:
        """Convert a database row to an Episode model."""
        steps_data = json.loads(row["steps"])
        steps = [EpisodeStep(**s) for s in steps_data]

        return Episode(
            episode_id=row["episode_id"],
            agent_id=row["agent_id"],
            status=row["status"],
            steps=steps,
            tools_used=json.loads(row["tools_used"]),
            total_tokens=row["total_tokens"],
            total_cost_usd=row["total_cost_usd"],
            total_duration_ms=row["total_duration_ms"],
            step_count=row["step_count"],
            started_at=row["started_at"],
            ended_at=row["ended_at"],
            metadata=json.loads(row["metadata"]),
        )

    @staticmethod
    def _row_to_summary(row: aiosqlite.Row) -> EpisodeSummary:
        """Convert a database row to a lightweight summary."""
        return EpisodeSummary(
            episode_id=row["episode_id"],
            agent_id=row["agent_id"],
            status=row["status"],
            tools_used=json.loads(row["tools_used"]),
            total_tokens=row["total_tokens"],
            total_cost_usd=row["total_cost_usd"],
            total_duration_ms=row["total_duration_ms"],
            step_count=row["step_count"],
            started_at=row["started_at"],
            ended_at=row["ended_at"],
        )
