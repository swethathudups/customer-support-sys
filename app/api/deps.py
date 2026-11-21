"""FastAPI dependencies: auth principal, DB session, services, ownership, idempotency."""
from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Header
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import (
    API_KEY_HEADER,
    CUSTOMER_ID_HEADER,
    Principal,
    authenticate,
    ensure_owns_customer,
)
from app.db.agent_models import ensure_agent_schema
from app.db.session import get_engine, get_session
from app.repositories.agent_repo import AgentRepository
from app.services.agent_service import AgentService
from app.repositories.customers_repo import CustomersRepository
from app.repositories.orders_repo import OrdersRepository
from app.repositories.products_repo import ProductsRepository
from app.repositories.resolutions_repo import ResolutionsRepository
from app.repositories.support_repo import SupportRepository
from app.services.customer_service import CustomerService
from app.services.order_service import OrderService
from app.services.product_service import ProductService
from app.services.resolution_service import ResolutionService
from app.services.ticket_service import TicketService
from app.core.config import settings


async def get_principal(
    x_api_key: Annotated[str | None, Header(alias=API_KEY_HEADER)] = None,
    x_customer_id: Annotated[str | None, Header(alias=CUSTOMER_ID_HEADER)] = None,
) -> Principal:
    return authenticate(x_api_key, x_customer_id)


def get_idempotency_key(
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
) -> str | None:
    return idempotency_key


DbSession = Annotated[AsyncSession, Depends(get_session)]
CurrentPrincipal = Annotated[Principal, Depends(get_principal)]


# --------------------------- service factories ---------------------------
def get_customer_service(db: DbSession) -> CustomerService:
    return CustomerService(CustomersRepository(db))


def get_order_service(db: DbSession) -> OrderService:
    return OrderService(OrdersRepository(db))


def get_product_service(db: DbSession) -> ProductService:
    return ProductService(ProductsRepository(db))


def get_resolution_service(db: DbSession) -> ResolutionService:
    return ResolutionService(
        OrdersRepository(db), ResolutionsRepository(db), SupportRepository(db),
        approval_limit_cents=settings.agent_refund_approval_limit_cents,
    )


def get_ticket_service(db: DbSession) -> TicketService:
    return TicketService(SupportRepository(db), OrdersRepository(db))


async def get_agent_service(db: DbSession) -> AgentService:
    await ensure_agent_schema(get_engine())   # idempotent, once per engine
    return AgentService(AgentRepository(db))


AgentSvc = Annotated[AgentService, Depends(get_agent_service)]
CustomerSvc = Annotated[CustomerService, Depends(get_customer_service)]
OrderSvc = Annotated[OrderService, Depends(get_order_service)]
ProductSvc = Annotated[ProductService, Depends(get_product_service)]
ResolutionSvc = Annotated[ResolutionService, Depends(get_resolution_service)]
TicketSvc = Annotated[TicketService, Depends(get_ticket_service)]


# --------------------------- ownership guard ---------------------------
async def get_owned_order(order_id: int, principal: CurrentPrincipal, db: DbSession) -> dict:
    """Load an order and enforce that the caller is allowed to see this customer's data."""
    from app.core.exceptions import NotFoundError
    owner = await OrdersRepository(db).get_owner(order_id)
    if not owner:
        raise NotFoundError("Order not found.", details={"order_id": order_id})
    ensure_owns_customer(principal, owner["customer_id"])
    return owner


OwnedOrder = Annotated[dict, Depends(get_owned_order)]


async def ensure_customer_access(customer_id: int, principal: CurrentPrincipal) -> int:
    ensure_owns_customer(principal, customer_id)
    return customer_id


CustomerAccess = Annotated[int, Depends(ensure_customer_access)]
