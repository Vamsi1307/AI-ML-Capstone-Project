"""Main application entry point."""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.logging_config import setup_logging, get_logger
from app.core.providers import provider_manager
from app.api.routes import router as api_router

# Setup logging
setup_logging(log_level=settings.LOG_LEVEL, log_file=settings.LOG_FILE)
logger = get_logger(__name__)


# Lifespan event handlers
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown."""
    # Startup
    logger.info("Application starting up")
    primary = provider_manager.get_primary_provider()
    logger.info(
        "Configuration loaded",
        provider=primary.provider_type.value,
        model=primary.model,
        vector_store_type=settings.VECTOR_STORE_TYPE,
    )

    # Log provider status
    provider_manager.log_provider_status()

    yield
    # Shutdown
    logger.info("Application shutting down")


# Create FastAPI application
app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION,
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure based on your requirements
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Middleware for request/response logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all HTTP requests and responses."""
    logger.info(
        "Incoming request",
        method=request.method,
        path=request.url.path,
    )

    response = await call_next(request)

    logger.info(
        "Response sent",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
    )

    return response


# Exception handler
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle uncaught exceptions."""
    logger.error(
        "Unhandled exception",
        method=request.method,
        path=request.url.path,
        error=str(exc),
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


# Include routes
app.include_router(api_router, prefix="/api/v1", tags=["genai"])


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "app": settings.API_TITLE,
        "version": settings.API_VERSION,
        "docs": "/docs",
        "openapi": "/openapi.json",
    }


if __name__ == "__main__":
    import uvicorn

    logger.info(
        "Starting server",
        host=settings.HOST,
        port=settings.PORT,
        debug=settings.DEBUG,
    )

    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
    )
