"""FastAPI application factory with lifespan management."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from evalkit import __version__
from evalkit.api.routes import router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager: initialize resources on startup, clean up on shutdown."""
    storage = app.state.storage
    logger.info("EvalKit API v%s starting up.", __version__)

    # If the storage backend supports async initialisation, trigger it
    if storage is not None and hasattr(storage, "_get_engine"):
        try:
            await storage._get_engine()
            logger.info("Storage backend initialised.")
        except Exception as exc:
            logger.warning("Storage backend initialisation failed: %s", exc)

    yield

    # Shutdown
    logger.info("EvalKit API shutting down.")
    if storage is not None and hasattr(storage, "close"):
        try:
            await storage.close()
        except Exception as exc:
            logger.warning("Error closing storage backend: %s", exc)


def create_app(
    storage: Any = None,
    runner_config: Any = None,
    title: str = "EvalKit API",
    allow_origins: list[str] | None = None,
    debug: bool = False,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        storage: A StorageBackend instance (JSONFileBackend or SQLiteBackend).
                 If None, endpoints that require storage will return 503.
        runner_config: Optional RunnerConfig for shared runner settings.
        title: API title shown in the OpenAPI docs.
        allow_origins: CORS allowed origins. Defaults to ["*"] (open).
        debug: Enable FastAPI debug mode.

    Returns:
        A configured FastAPI application instance.
    """
    app = FastAPI(
        title=title,
        version=__version__,
        description=(
            "EvalKit — LLM evaluation and comparison framework REST API.\n\n"
            "Run evaluation suites, compare model outputs, and manage human preference "
            "judgments via HTTP."
        ),
        debug=debug,
        lifespan=_lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # Attach shared state
    app.state.storage = storage
    app.state.runner_config = runner_config

    # CORS middleware
    origins = allow_origins or ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routes
    app.include_router(router, prefix="/api/v1")

    @app.get("/", tags=["system"], include_in_schema=False)
    async def root() -> dict[str, str]:
        return {
            "service": "EvalKit API",
            "version": __version__,
            "docs": "/docs",
        }

    logger.info("EvalKit FastAPI app created (debug=%s, storage=%s)", debug, type(storage).__name__)
    return app
