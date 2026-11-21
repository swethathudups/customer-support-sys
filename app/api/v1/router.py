"""Aggregate the v1 API surface.

Category 1 — Tool APIs (backend business APIs): customers, orders, order-actions,
products, tickets. Category 2 — Agentic APIs (agent orchestration): /agent/*.
"""
from __future__ import annotations

from fastapi import APIRouter

from app.api.v1.routers import actions, agent, customers, orders, products, tickets

api_router = APIRouter()
api_router.include_router(customers.router)
api_router.include_router(orders.router)
api_router.include_router(actions.router)
api_router.include_router(products.router)
api_router.include_router(tickets.router)
api_router.include_router(agent.router)
