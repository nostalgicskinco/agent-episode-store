"""
Episode schema for the Agent Episode Store.

An Episode is a complete record of one agent task — from start to finish.
It groups individual AIR records (LLM calls) into a replayable sequence
of steps, each with its own type, tool invocation, and result.

EL-1: Define Episode schema (run_id, steps[], tools[], result, metadata)
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class EpisodeStatus(str, Enum):
    """Outcome of an episode."""
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    KILLED = "killed"


class StepType(str, Enum):
    """What kind of action this step represents."""
    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    DECISION = "decision"
    ERROR = "error"


# ---------------------------------------------------------------------------
# Step (one action inside an episode)
# ---------------------------------------------------------------------------

class EpisodeStep(BaseModel):
    """A single step in an episode's sequence.

    Each step maps to one action the agent took — an LLM call,
    a tool invocation, or a decision point. The `air_record_id`
    links back to the raw AIR record in air-blackbox-gateway.
    """

    step_index: int = Field(
        ..., ge=0, description="Position in the episode sequence (0-based)"
    )
    step_type: StepType = Field(
        ..., description="What kind of action this step represents"
    )
    air_record_id: str | None = Field(
        default=None,
        description="Links to the AIR record run_id in air-blackbox-gateway",
    )
    tool_name: str | None = Field(
        default=None, description="Name of the tool invoked (if tool_call)"
    )
    model: str | None = Field(
        default=None, description="LLM model used (if llm_call)"
    )
    provider: str | None = Field(
        default=None, description="Provider (openai, anthropic, etc.)"
    )
    input_summary: str | None = Field(
        default=None, description="Brief summary of what went into this step"
    )
    output_summary: str | None = Field(
        default=None, description="Brief summary of what came out of this step"
    )
    tokens: int = Field(default=0, ge=0, description="Tokens used in this step")
    cost_usd: float = Field(default=0.0, ge=0, description="Cost in USD for this step")
    duration_ms: int = Field(default=0, ge=0, description="Step duration in milliseconds")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this step occurred",
    )
    error: str | None = Field(
        default=None, description="Error message if this step failed"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary key-value pairs"
    )


# ---------------------------------------------------------------------------
# Episode (the full agent run)
# ---------------------------------------------------------------------------

class EpisodeCreate(BaseModel):
    """Payload for creating a new episode via POST /v1/episodes.

    The caller provides the agent_id, steps, and result.
    The server assigns episode_id and timestamps.
    """

    agent_id: str = Field(
        ..., min_length=1, description="Which agent ran this episode"
    )
    steps: list[EpisodeStep] = Field(
        default_factory=list, description="Ordered sequence of steps"
    )
    status: EpisodeStatus = Field(
        default=EpisodeStatus.RUNNING, description="Episode outcome"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary tags for this episode"
    )


class Episode(BaseModel):
    """A complete episode — the full record of one agent task.

    This is what gets stored in the database and returned by the API.
    It adds server-assigned fields (episode_id, timestamps, aggregates)
    on top of the EpisodeCreate payload.
    """

    episode_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique episode identifier (server-assigned)",
    )
    agent_id: str = Field(
        ..., min_length=1, description="Which agent ran this episode"
    )
    status: EpisodeStatus = Field(
        default=EpisodeStatus.RUNNING, description="Episode outcome"
    )

    # Steps
    steps: list[EpisodeStep] = Field(
        default_factory=list, description="Ordered sequence of steps"
    )

    # Aggregates (computed from steps)
    tools_used: list[str] = Field(
        default_factory=list,
        description="Deduplicated list of tools invoked during this episode",
    )
    total_tokens: int = Field(default=0, ge=0, description="Sum of all step tokens")
    total_cost_usd: float = Field(default=0.0, ge=0, description="Sum of all step costs")
    total_duration_ms: int = Field(
        default=0, ge=0, description="Total episode duration in milliseconds"
    )
    step_count: int = Field(default=0, ge=0, description="Number of steps")

    # Timestamps
    started_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this episode started",
    )
    ended_at: datetime | None = Field(
        default=None, description="When this episode ended (null if still running)"
    )

    # Metadata
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary key-value pairs for tagging"
    )

    def compute_aggregates(self) -> None:
        """Recompute aggregate fields from the steps list."""
        self.step_count = len(self.steps)
        self.total_tokens = sum(s.tokens for s in self.steps)
        self.total_cost_usd = round(sum(s.cost_usd for s in self.steps), 6)
        self.total_duration_ms = sum(s.duration_ms for s in self.steps)
        seen: set[str] = set()
        tools: list[str] = []
        for s in self.steps:
            if s.tool_name and s.tool_name not in seen:
                tools.append(s.tool_name)
                seen.add(s.tool_name)
        self.tools_used = tools


class EpisodeSummary(BaseModel):
    """Lightweight view for list endpoints — no steps included."""

    episode_id: str
    agent_id: str
    status: EpisodeStatus
    tools_used: list[str]
    total_tokens: int
    total_cost_usd: float
    total_duration_ms: int
    step_count: int
    started_at: datetime
    ended_at: datetime | None


# ---------------------------------------------------------------------------
# Replay (EL-5)
# ---------------------------------------------------------------------------

class ReplayStep(BaseModel):
    """A single step in a replay sequence.

    Adds a 'replay_index' for ordering and strips fields
    that aren't needed for replay (like timestamps).
    """
    replay_index: int
    step_type: StepType
    tool_name: str | None = None
    model: str | None = None
    provider: str | None = None
    input_summary: str | None = None
    output_summary: str | None = None
    tokens: int = 0
    cost_usd: float = 0.0
    duration_ms: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class EpisodeReplay(BaseModel):
    """Replay-ready view of an episode.

    Contains just what you need to feed back through the gateway:
    the agent, the ordered steps, and the original result.
    """
    episode_id: str
    agent_id: str
    original_status: EpisodeStatus
    replay_steps: list[ReplayStep]
    total_tokens: int
    total_cost_usd: float
    tools_used: list[str]


# ---------------------------------------------------------------------------
# Diff (EL-7)
# ---------------------------------------------------------------------------

class StepDiff(BaseModel):
    """Difference between two steps at the same index."""
    step_index: int
    field: str
    left: str | None = None
    right: str | None = None


class EpisodeDiff(BaseModel):
    """Result of comparing two episodes step-by-step."""
    left_episode_id: str
    right_episode_id: str
    left_step_count: int
    right_step_count: int
    matching_steps: int
    differing_steps: int
    extra_left: int
    extra_right: int
    token_delta: int
    cost_delta: float
    duration_delta: int
    step_diffs: list[StepDiff]
