"""
Gateway integration webhook client.

EL-6: Gateway integration — emit episode events from air-blackbox-gateway

This module provides a client that the air-blackbox-gateway can use
to send completed episodes to the episode store. It batches AIR records
by run_id into episodes and POSTs them to the store's ingest endpoint.

Usage from the gateway side (Go):
    The gateway calls POST http://episode-store:8100/v1/episodes
    with a JSON body containing agent_id, steps, status, and metadata.

Usage from Python (for testing or companion services):
    webhook = GatewayWebhook("http://localhost:8100")
    episode = await webhook.send_episode(agent_id="my-agent", steps=[...], status="success")
"""

from __future__ import annotations

import httpx

from pkg.models import Episode, EpisodeStep, EpisodeStatus


class GatewayWebhook:
    """HTTP client for sending episodes to the episode store.

    This is the integration point between air-blackbox-gateway
    and the episode store. The gateway collects AIR records for
    a task, groups them into steps, and sends the episode here.
    """

    def __init__(self, base_url: str = "http://localhost:8100") -> None:
        self.base_url = base_url.rstrip("/")

    async def send_episode(
        self,
        agent_id: str,
        steps: list[dict],
        status: str = "success",
        metadata: dict | None = None,
    ) -> Episode:
        """Send a completed episode to the store.

        Args:
            agent_id: Which agent ran this episode.
            steps: List of step dicts matching the EpisodeStep schema.
            status: Episode outcome (success, failure, timeout, killed).
            metadata: Optional tags for this episode.

        Returns:
            The created Episode with server-assigned ID and aggregates.

        Raises:
            httpx.HTTPStatusError: If the store returns an error.
        """
        payload = {
            "agent_id": agent_id,
            "steps": steps,
            "status": status,
            "metadata": metadata or {},
        }

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.base_url}/v1/episodes",
                json=payload,
                timeout=10.0,
            )
            resp.raise_for_status()
            return Episode(**resp.json())

    async def health_check(self) -> dict:
        """Check if the episode store is reachable."""
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{self.base_url}/v1/health",
                timeout=5.0,
            )
            resp.raise_for_status()
            return resp.json()
