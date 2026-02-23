"""
Agent Episode Store — FastAPI server.

Starts the episode store service with SQLite backend.
Configurable via environment variables:

    EPISODE_DB_PATH  — Path to SQLite database (default: episodes.db)
    EPISODE_HOST     — Listen host (default: 0.0.0.0)
    EPISODE_PORT     — Listen port (default: 8100)
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from pkg.api.routes import router, set_store
from pkg.storage import EpisodeStore


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: open DB and create tables. Shutdown: close DB."""
    db_path = os.environ.get("EPISODE_DB_PATH", "episodes.db")
    store = EpisodeStore(db_path)
    await store.init()
    set_store(store)
    print(f"Episode store ready — db={db_path}")
    yield
    await store.close()
    print("Episode store shut down")


app = FastAPI(
    title="Agent Episode Store",
    description="Replayable episode ledger for AI agent runs",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("EPISODE_HOST", "0.0.0.0")
    port = int(os.environ.get("EPISODE_PORT", "8100"))
    uvicorn.run("cmd.server:app", host=host, port=port, reload=True)