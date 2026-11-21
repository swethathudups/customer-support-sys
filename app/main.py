"""FastAPI application factory (Category 1 — Tool APIs)."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI

from app import __version__
from app.api.deps import get_principal
from app.api.v1.router import api_router
from app.api.v1.routers import health as health_router
from app.core.config import settings
from app.core.correlation import CorrelationIdMiddleware
from app.core.exceptions import register_exception_handlers
from app.core.logging import configure_logging, get_logger
from app.core.rate_limit import RateLimitMiddleware
from app.db.session import dispose_engine, get_engine, ping

logger = get_logger("app.main")

_DESCRIPTION = """
Agentic Customer Support System — **Category 1: Tool APIs**.

Secure, read-mostly REST endpoints over the commerce database that the support
agent (added later) will consume as tools:

* **customers** — profile, orders, addresses, store credit
* **orders** — details, status, tracking, items, payment, history, invoice,
  delivery estimate, and refund/return/cancellation **eligibility** (read-only)
* **order-actions** — refunds, returns, claims, cancellation, goodwill credit
  (writes; idempotent; policy + confirmation enforced)
* **products** — catalog
* **tickets** — support tickets

**Category 2 — Agentic APIs** (`/agent/*`): `POST /agent/chat` runs the LangGraph support
agent, which calls the Tool APIs above (as a customer-scoped service principal);
`/agent/sessions|runs/...` expose messages, traces, steps and tool calls.

Auth: send `X-API-Key`. Act on behalf of a customer with `X-Customer-Id`
(customer-ownership is enforced on every order/customer resource).
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    get_engine()
    db_ok = await ping()
    logger.info("startup", extra={"db_ok": db_ok,
                                  "db_scheme": settings.effective_database_url.split("://")[0]})
    yield
    await dispose_engine()
    logger.info("shutdown")


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description=_DESCRIPTION,
        lifespan=lifespan,
        openapi_tags=[
            {"name": "customers", "description": "Customer profile, orders, addresses, store credit."},
            {"name": "orders", "description": "Order details, tracking, payments, eligibility (read-only)."},
            {"name": "order-actions", "description": "Refunds, returns, claims, cancellation (writes)."},
            {"name": "products", "description": "Product catalog."},
            {"name": "tickets", "description": "Support tickets."},
            {"name": "agent", "description": "Agentic APIs: chat + execution observability."},
            {"name": "health", "description": "Liveness/health."},
        ],
    )

    # Middleware (outermost first): correlation id, then rate limiting.
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(CorrelationIdMiddleware)

    register_exception_handlers(app)

    app.include_router(health_router.router)
    # Every /api/v1 route requires a valid API key principal.
    app.include_router(api_router, prefix=settings.api_v1_prefix,
                       dependencies=[Depends(get_principal)])
    return app


app = create_app()
