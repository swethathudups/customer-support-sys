# Agent API design (Category 2)

## Two API categories and how they talk
```
Frontend --HTTP--> Agentic APIs  /api/v1/agent/*          (X-API-Key, customer scope)
                       |  AgentService (auth, session, run lifecycle, trace persistence)
                       v
                  LangGraph graph (guardrails, tool loop, confirmation, escalation)
                       |  Planner (Claude | rules)   KB search (_KB, BM25)
                       v
                  ToolExecutor -- gates: schema > order ownership > prerequisite > confirmation
                       |  HTTP + timeout/retry/circuit breaker, as `agent` principal + X-Customer-Id
                       v
                  Tool APIs  /api/v1/customers|orders|products|tickets   (re-check ownership, rules, audit)
                       v
                  Services > Repositories > DB
```
The frontend never reaches tools or the DB; the agent never reaches the DB, only Tool API endpoints.

## Agentic endpoints
See README section 9. Schemas: `app/schemas/agent.py`. Errors use the shared envelope
`{"error": {code, message, details, request_id}}`: 401 missing/invalid key, 403 ownership, 404 unknown
customer/session/run, 422 validation, 503 backend down. Agent-level failures return 200 with
`status: "failed" | "escalated"` and a safe message (never exception text).

## Tool catalogue (`app/agent/tools/registry.py`)
| Tool | Endpoint | Kind |
|---|---|---|
| get_customer_profile, list_customer_orders, get_customer_addresses, get_store_credit_balance | `/customers/{id}...` | read |
| get_order_details, status, tracking, items, payment, history, invoice, delivery_estimate | `/orders/{id}...` | read |
| check_refund_eligibility, check_return_eligibility, check_cancellation_eligibility | `/orders/{id}/...-eligibility` | read (verdict only) |
| get_order_refunds, get_order_returns, get_order_claims | `/orders/{id}/...` | read |
| search_products, get_product, get_support_ticket | `/products...`, `/tickets/{n}` | read |
| search_knowledge_base | in-process KB | read |
| create_support_ticket | `POST /tickets` | write, no confirmation (low risk) |
| initiate_refund, create_return, cancel_order, create_claim, grant_delay_credit | `POST /orders/{id}/...` | write, **confirmation + prerequisite** |

## Trace event types
`intake, guardrail, intent, decision, retrieval, tool_call, validation, confirmation, escalation,
response, error` - each with status (`ok|blocked|error`), redacted `data`, and duration.
Tool calls carry method, masked path, HTTP status, error code, redacted request/response summary,
attempts and idempotency key.

## Example tool selection (rules planner; the LLM planner chooses freely within the same gates)
| Question | Tools |
|---|---|
| Where is my order? | list_customer_orders, get_order_tracking, get_delivery_estimate |
| Can I get a refund? | (list orders), get_order_details, search_knowledge_base, check_refund_eligibility, then explain |
| I want a refund | same, then *stage* initiate_refund; on "yes" re-check eligibility, then initiate_refund |
| What is your refund policy? | search_knowledge_base |
| Why was my refund rejected? | get_order_refunds, search_knowledge_base |
| Wrong item received | get_order_details, get_order_items, search_knowledge_base, create_support_ticket |
