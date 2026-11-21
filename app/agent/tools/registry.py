"""Agent tool catalogue.

Each tool = clear name + description + typed (Pydantic) input schema + the Tool
API endpoint it maps to + its safety metadata. The LLM planner receives these as
tool definitions; the rule planner calls them by name. Neither can reach SQL or
the DB - the only capability the agent has is "call one of these endpoints".

Notable design points:
  * `customer_id` is NEVER a tool argument - it is injected from the
    authenticated chat context, so a prompt cannot redirect a call to someone else.
  * The model refers to orders by `order_number` (what customers know); the
    executor resolves it against *this customer's* orders only.
  * Write tools declare `requires_confirmation` and a `prerequisite` read tool
    that must have run first (e.g. eligibility check before a refund).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field

ORDER_NUMBER_PATTERN = r"^ORD-\d{3,}$"


# ------------------------------ input schemas ------------------------------
class NoInput(BaseModel):
    pass


class OrderInput(BaseModel):
    order_number: str = Field(..., pattern=ORDER_NUMBER_PATTERN,
                              description="Customer-facing order number, e.g. ORD-100001.")


class ListOrdersInput(BaseModel):
    limit: int = Field(10, ge=1, le=50, description="Max orders to return (newest first).")
    status: str | None = Field(None, description="Optional status filter, e.g. 'shipped'.")


class SearchProductsInput(BaseModel):
    q: str | None = Field(None, max_length=100, description="Keywords to match title/description.")
    category: str | None = Field(None, max_length=60, description="Category slug.")
    limit: int = Field(5, ge=1, le=20)


class GetProductInput(BaseModel):
    product_id: int = Field(..., ge=1)


class GetTicketInput(BaseModel):
    ticket_number: str = Field(..., max_length=40)


class SearchKBInput(BaseModel):
    query: str = Field(..., min_length=2, max_length=300,
                       description="Natural-language policy question or keywords.")


class InitiateRefundInput(OrderInput):
    reason: str = Field(..., min_length=3, max_length=500)
    method: Literal["original_payment", "store_credit", "gift_card"] = "original_payment"


class CreateReturnInput(OrderInput):
    reason: Literal["change_of_mind", "wrong_size", "defective", "damaged", "wrong_item",
                    "missing_item", "not_as_described", "other"]
    order_item_ids: list[int] = Field(..., min_length=1,
                                      description="order_item_id values from the order's items.")


class CreateClaimInput(OrderInput):
    type: Literal["damaged", "defective", "wrong_item", "missing_item", "missing_part"]
    order_item_id: int | None = None
    description: str = Field(..., min_length=3, max_length=1000)
    photos_provided: bool = False


class CancelOrderInput(OrderInput):
    reason: str = Field(..., min_length=3, max_length=500)


class CreateTicketInput(BaseModel):
    subject: str = Field(..., min_length=3, max_length=200)
    message: str = Field(..., min_length=1, max_length=4000)
    category: Literal["refunds", "returns", "shipping", "delays", "damaged", "payments", "account",
                      "warranty", "promotions", "order_management", "other"] | None = None
    order_number: str | None = Field(None, pattern=ORDER_NUMBER_PATTERN)
    priority: Literal["low", "normal", "high", "urgent"] = "normal"


# --------------------------------- spec ---------------------------------
@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_model: type[BaseModel]
    method: str = "GET"
    path: str = ""                      # template: {customer_id} / {order_id}
    query_fields: tuple[str, ...] = ()  # input fields sent as query params (GET)
    body_fn: Callable[[BaseModel], dict[str, Any]] | None = None
    is_write: bool = False
    requires_confirmation: bool = False
    prerequisite: str | None = None     # read tool that must have succeeded for this order first
    risk: Literal["low", "medium", "high"] = "low"
    local: bool = False                 # handled in-process (KB), not via HTTP

    def input_schema(self) -> dict:
        schema = self.input_model.model_json_schema()
        schema.pop("title", None)
        schema["additionalProperties"] = False
        return schema

    @property
    def order_scoped(self) -> bool:
        return "{order_id}" in self.path or "order_number" in self.input_model.model_fields


def _body(*fields: str, **extra: Any) -> Callable[[BaseModel], dict[str, Any]]:
    def build(m: BaseModel) -> dict[str, Any]:
        data = m.model_dump()
        return {**{f: data[f] for f in fields}, **extra}
    return build


def _order(name: str, desc: str, suffix: str) -> ToolSpec:
    return ToolSpec(name, desc, OrderInput, path=f"/orders/{{order_id}}{suffix}")


_TOOLS: list[ToolSpec] = [
    # ---- customer ----
    ToolSpec("get_customer_profile", "Get the signed-in customer's profile (name, loyalty tier, status).",
             NoInput, path="/customers/{customer_id}"),
    ToolSpec("list_customer_orders",
             "List the customer's orders, newest first (order_number, status, totals, promised date). "
             "Use first when the customer has not named a specific order.",
             ListOrdersInput, path="/customers/{customer_id}/orders", query_fields=("limit", "status")),
    ToolSpec("get_customer_addresses", "List the customer's saved addresses.", NoInput,
             path="/customers/{customer_id}/addresses"),
    ToolSpec("get_store_credit_balance", "Get the customer's store-credit balance.", NoInput,
             path="/customers/{customer_id}/store-credit"),
    # ---- order (read) ----
    _order("get_order_details", "Full order detail: status, totals, items, promised delivery.", ""),
    _order("get_order_status", "Current status of one order.", "/status"),
    _order("get_order_tracking", "Shipment, carrier, tracking number and carrier event timeline.", "/tracking"),
    _order("get_order_items", "Line items (order_item_id, title, qty, price) for one order.", "/items"),
    _order("get_order_payment", "Payment records for an order (card brand + last4 only).", "/payment"),
    _order("get_order_history", "Order status history timeline.", "/history"),
    _order("get_order_invoice", "Invoice for an order.", "/invoice"),
    _order("get_delivery_estimate", "Delivery estimate and how many days late the order is, if any.",
           "/delivery-estimate"),
    _order("check_refund_eligibility",
           "READ-ONLY verdict on whether the order can be refunded (window, prior refund, dispute, "
           "$100 agent limit). Does NOT issue a refund.", "/refund-eligibility"),
    _order("check_return_eligibility", "READ-ONLY per-item return eligibility. Does NOT create a return.",
           "/return-eligibility"),
    _order("check_cancellation_eligibility", "READ-ONLY: can this order still be cancelled?",
           "/cancellation-eligibility"),
    _order("get_order_refunds", "Existing refund records for the order (status, amount, approver).", "/refunds"),
    _order("get_order_returns", "Existing return/RMA records for the order.", "/returns"),
    _order("get_order_claims", "Existing damage/wrong-item claims for the order.", "/claims"),
    # ---- catalogue / tickets / KB ----
    ToolSpec("search_products", "Search the product catalogue.", SearchProductsInput, path="/products",
             query_fields=("q", "category", "limit")),
    ToolSpec("get_product", "Get one product's details.", GetProductInput, path="/products/{product_id}"),
    ToolSpec("get_support_ticket", "Retrieve one of the customer's support tickets and its messages.",
             GetTicketInput, path="/tickets/{ticket_number}"),
    ToolSpec("search_knowledge_base",
             "Search company policy articles (refunds, returns, shipping, delays, warranty, payments...). "
             "Use for any 'what is your policy' question and to ground eligibility explanations.",
             SearchKBInput, local=True),
    # ---- writes ----
    ToolSpec("create_support_ticket",
             "Open a support ticket (e.g. wrong/damaged item, escalation to a human). Low risk.",
             CreateTicketInput, method="POST", path="/tickets", is_write=True,
             body_fn=_body("subject", "category", "order_number", "message", "priority")),
    ToolSpec("initiate_refund",
             "ISSUE a refund. Only after eligibility was checked and the customer EXPLICITLY asked for it; "
             "the system will ask the customer to confirm before it runs.",
             InitiateRefundInput, method="POST", path="/orders/{order_id}/refunds", is_write=True,
             requires_confirmation=True, prerequisite="check_refund_eligibility", risk="high",
             body_fn=_body("reason", "method", customer_confirmed=True)),
    ToolSpec("create_return",
             "Create a return (RMA) for specific items. Requires an explicit customer request and "
             "confirmation; call check_return_eligibility first.",
             CreateReturnInput, method="POST", path="/orders/{order_id}/returns", is_write=True,
             requires_confirmation=True, prerequisite="check_return_eligibility", risk="medium",
             body_fn=_body("reason", "order_item_ids", customer_confirmed=True)),
    ToolSpec("cancel_order",
             "Cancel an unshipped order. Requires an explicit request and confirmation; call "
             "check_cancellation_eligibility first.",
             CancelOrderInput, method="POST", path="/orders/{order_id}/cancel", is_write=True,
             requires_confirmation=True, prerequisite="check_cancellation_eligibility", risk="high",
             body_fn=_body("reason", customer_confirmed=True)),
    ToolSpec("create_claim", "Open a damaged/defective/wrong/missing item claim (needs confirmation).",
             CreateClaimInput, method="POST", path="/orders/{order_id}/claims", is_write=True,
             requires_confirmation=True, risk="medium",
             body_fn=_body("type", "order_item_id", "description", "photos_provided")),
    ToolSpec("grant_delay_credit",
             "Grant the policy goodwill credit for a late order ($0/$5/$15, one per order). Needs "
             "confirmation; check get_delivery_estimate first.",
             OrderInput, method="POST", path="/orders/{order_id}/goodwill-credit", is_write=True,
             requires_confirmation=True, prerequisite="get_delivery_estimate", risk="medium"),
]

TOOLS: dict[str, ToolSpec] = {t.name: t for t in _TOOLS}


def anthropic_tool_defs() -> list[dict]:
    """Tool definitions in the Anthropic Messages API format."""
    return [{"name": t.name, "description": t.description, "input_schema": t.input_schema()}
            for t in TOOLS.values()]
