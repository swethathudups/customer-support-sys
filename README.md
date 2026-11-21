# Customer Support API — Category 1: Tool APIs (FastAPI)

Production-grade, API-driven backend for the Agentic Customer Support System.
This is **Category 1 — the Tool APIs**: secure, mostly read-only REST endpoints
over the commerce database that the support agent (Category 2, a later phase)
will consume as tools. The agent never touches the database; it will go through
exactly these endpoints.

- **Knowledge base**: `_KB/` (82 Markdown policy articles) — built.
- **Commerce DB**: `_DB/ecommerce.db` (SQLite, 34 tables + 7 policy views) — built & seeded.
- **This component**: the FastAPI layer in `app/`.

> The policy views in the DB (`v_refund_eligibility`, `v_delivery_status`, …)
> encode the business rules once; the API reads them so the API and the KB can
> never disagree (30-day window, one refund/order, $100 agent limit, delay
> tiers $0/$5/$15, dispute blocks).

---

## 1. Architecture

Layered, with strict separation of concerns:

```
HTTP request
  │
  ▼
Router (app/api/v1/routers/*)      HTTP only: validation, auth deps, status codes
  │
  ▼
Service (app/services/*)           business logic, policy enforcement, schema mapping
  │
  ▼
Repository (app/repositories/*)    all SQL (parameterized), nothing else
  │
  ▼
Async SQLAlchemy 2.0  →  SQLite (_DB/ecommerce.db)   [Postgres-portable]
```

Cross-cutting (in `app/core/`): environment config, structured JSON logging,
request **correlation IDs**, centralized exception handling (no internal
leakage), API-key **auth + customer-ownership**, in-memory **rate limiting**,
and pagination/sort helpers. Mutating endpoints support **idempotency**.

### Request/response contract
- Success: the resource (Pydantic v2 schema). Lists are wrapped in `{items, meta}`.
- Error: a consistent envelope `{"error": {code, message, details, request_id}}`.
- Money is integer **cents** everywhere, surfaced as `{cents, currency, display}`.

---

## 2. Endpoint catalog

Base prefix: `/api/v1`. All endpoints require `X-API-Key`. Customer/order
endpoints additionally require the caller to be authorized for that customer
(see Auth).

### Customers
| Method | Path | Purpose |
|---|---|---|
| GET | `/customers/{customer_id}` | Customer profile |
| GET | `/customers/{customer_id}/orders` | Customer's orders (paginated, filter/sort) |
| GET | `/customers/{customer_id}/addresses` | Saved addresses |
| GET | `/customers/{customer_id}/store-credit` | Store-credit balance |

### Orders — read-only
| Method | Path | Purpose |
|---|---|---|
| GET | `/orders/{order_id}` | Full order detail + items |
| GET | `/orders/{order_id}/status` | Current status |
| GET | `/orders/{order_id}/tracking` | Shipment + carrier timeline (WISMO) |
| GET | `/orders/{order_id}/items` | Line items |
| GET | `/orders/{order_id}/payment` | Payment records (brand + last4 only) |
| GET | `/orders/{order_id}/history` | Status history |
| GET | `/orders/{order_id}/invoice` | Invoice |
| GET | `/orders/{order_id}/delivery-estimate` | Delivery estimate / lateness |
| GET | `/orders/{order_id}/refund-eligibility` | **Verdict only — does not refund** |
| GET | `/orders/{order_id}/return-eligibility` | Per-item return rules |
| GET | `/orders/{order_id}/cancellation-eligibility` | Can it still be cancelled |
| GET | `/orders/{order_id}/refunds` | Existing refund records |
| GET | `/orders/{order_id}/returns` | Existing return records |
| GET | `/orders/{order_id}/claims` | Existing claim records |

### Orders — writes (POST, idempotent, policy-gated)
| Method | Path | Purpose |
|---|---|---|
| POST | `/orders/{order_id}/refunds` | Initiate refund (requires confirmation; **>$100 → human approval**) |
| POST | `/orders/{order_id}/returns` | Create a return (RMA) |
| POST | `/orders/{order_id}/claims` | Open a damaged/wrong/missing claim |
| POST | `/orders/{order_id}/cancel` | Cancel an unshipped order (requires confirmation) |
| POST | `/orders/{order_id}/goodwill-credit` | Grant delay goodwill ($0/$5/$15, max one/order) |

### Products & Tickets
| Method | Path | Purpose |
|---|---|---|
| GET | `/products` | Search/list catalog (paginated) |
| GET | `/products/{product_id}` | Product detail |
| POST | `/tickets` | Create a support ticket |
| GET | `/tickets/{ticket_number}` | Retrieve a ticket + messages |

Interactive docs: **`/docs`** (Swagger) and **`/redoc`**. Liveness: **`/health`**.

### Key business rules enforced
- **A refund question is never consent.** `POST .../refunds` requires
  `customer_confirmed=true`; otherwise `409 confirmation_required`.
- **$100 agent limit.** Order value ≤ $100 and in-window → agent-issued
  (`approved_by_type=ai_agent`). Over $100 → **not issued**; a `pending` refund +
  human **approval** + escalation **ticket** are created.
- **One refund per order / open dispute / outside window / not delivered** →
  the write is declined with a specific error code; `refund-eligibility` returns
  the matching verdict.
- **GETs are strictly read-only.** All mutations are POST and audited.

---

## 3. Auth & authorization

Send `X-API-Key` (configured in `API_KEYS`). To act for a specific customer,
also send `X-Customer-Id`. Ownership is then enforced on every customer/order
resource (fail-closed `403 ownership_violation` on mismatch). The future agent
will call these same endpoints with its own key, scoped per chat customer — so
it can never read another customer's data.

```bash
curl -s localhost:8000/api/v1/customers/1 \
  -H "X-API-Key: dev-public-key" -H "X-Customer-Id: 1" | jq
```

---

## 4. Setup & run

Prerequisites: Python 3.11+, [uv](https://docs.astral.sh/uv/), and the seeded DB
(`bash _DB/build.sh`, already present).

```bash
uv sync                           # creates .venv and installs deps from pyproject.toml
cp .env.example .env              # optional; sensible defaults otherwise

uv run uvicorn app.main:app --reload     # http://localhost:8000/docs
```

Dependencies live in `pyproject.toml` (`uv sync` installs the `dev` group too;
use `uv sync --no-dev` for a runtime-only env).

Environment (all optional; see `.env.example`): `DATABASE_URL`, `API_KEYS`,
`RATE_LIMIT_PER_MINUTE`, `LOG_JSON`, `AGENT_REFUND_APPROVAL_LIMIT_CENTS`.

### Example

```bash
export K="-H X-API-Key:dev-public-key -H X-Customer-Id:1"
curl -s $K localhost:8000/api/v1/customers/1/orders | jq '.items[0]'
# refund eligibility (read):
curl -s $K localhost:8000/api/v1/orders/1/refund-eligibility | jq
# issue a refund (write, needs confirmation + idempotency key):
curl -s -X POST $K -H "Idempotency-Key: r-1" \
  -d '{"reason":"changed my mind","customer_confirmed":true}' \
  -H "Content-Type: application/json" \
  localhost:8000/api/v1/orders/1/refunds | jq
```

---

## 5. Tests

```bash
uv run pytest    # 36 tests: auth, ownership, pagination, all read paths,
                 # refund verdicts, confirmation/$100/idempotency, returns,
                 # cancellation, goodwill, claims, tickets
```

Each test runs against a fresh copy of `_DB/ecommerce.db`, asserting against the
seeded eval fixtures `ORD-100001..100011`.

---

## 6. Docker

```bash
docker compose up --build        # http://localhost:8000/docs
```

Runs as a non-root user with a `/health` healthcheck.

---

## 7. Portability (SQLite → PostgreSQL)

The code is DB-agnostic above the engine. Set
`DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/db` and add `asyncpg` to
`requirements.txt`. The schema (`_DB/schema.sql`) is written to port cleanly
(see `_DB/README.md`); the FK `PRAGMA` is SQLite-only and auto-skipped.

---

## 8. Project structure

```
app/
  core/        config, logging, correlation, security, exceptions, rate_limit, pagination
  db/          async engine/session (FK pragma per connection)
  schemas/     Pydantic v2 request/response models
  repositories/ parameterized SQL only (SQLAlchemy 2.0 async)
  services/    business logic + policy enforcement + idempotency
  api/         deps (DI, auth, ownership) + v1 routers
  main.py      app factory (middleware, exception handlers, routers)
tests/         pytest (ASGI transport, per-test DB copy)
Dockerfile, docker-compose.yml, pyproject.toml, .env.example
```

## 9. Category 2 — Agentic APIs (the support agent)

The frontend only ever calls `/api/v1/agent/*`. The agent (LangGraph) calls the Tool APIs above
as the **agent service principal**, scoped to the chat customer with `X-Customer-Id`, so the
Tool API's ownership checks apply to everything the agent does. Design: `docs/agent-api-design.md`.

| Method | Path | Purpose |
|---|---|---|
| POST | `/agent/chat` | Run one agent turn `{customer_id, session_id, message}` |
| GET | `/agent/sessions/{session_id}` | Session info (+ `awaiting_confirmation`) |
| GET | `/agent/sessions/{session_id}/messages` | Conversation history (paginated) |
| GET | `/agent/sessions/{session_id}/traces` | Execution traces for every run (paginated) |
| GET | `/agent/runs/{run_id}` | Run status + metadata |
| GET | `/agent/runs/{run_id}/steps` | Structured execution steps |
| GET | `/agent/runs/{run_id}/tools` | Tool calls (redacted request/response metadata) |
| GET | `/agent/health` | Agent + dependency health (DB, Tool API + circuit breaker, KB, LLM) |

```
screen -(abuse/legal/safety)-----------------> escalate (ticket) -> END
   |-(pending action + "yes")-> execute_confirmed -> END   (re-checks eligibility, idempotent)
   |-(pending action + "no")--> cancel_pending -> END
   '-> understand -(needs human)-> escalate
         '-> plan <--> act       tool loop, <= AGENT_MAX_STEPS; reads run, writes are only *staged*
               '-> respond -> END
```

**Safety properties** (enforced in code outside the LLM; covered by `tests/test_agent.py`)
- A refund/return/cancel *question* never changes data. Writes are staged as a pending action and
  the customer must reply "yes"; consent is parsed deterministically, expires after
  `AGENT_PENDING_TTL_MINUTES`, and is discarded if the customer changes topic.
- Every write needs its eligibility read first (`prerequisite`), is re-validated at confirm time and
  sent with an idempotency key (`{session}:{pending_id}`), so a stray second "yes" cannot double-refund.
- `customer_id` is never a tool argument; orders are addressed by `order_number` and resolved only
  against *this customer's* orders. The LLM has no SQL/DB access, only the typed tool catalogue.
- Abuse, legal threats, safety reports, chargebacks, privacy/fraud requests, tool failures and
  unresolved ambiguity (2 clarifications) open a ticket and hand off honestly (KB `GUARD-001`,
  `ESC-*`); the agent never guesses a status when a tool fails.
- Traces hold structured events only: PII keys redacted, card-like numbers masked (Luhn),
  payloads size-capped, no chain-of-thought. Observability endpoints are ownership-checked.

**Planners** (`AGENT_PLANNER=auto|llm|rules`): with `ANTHROPIC_API_KEY` set the agent uses Claude
(`LLM_MODEL`, default `claude-sonnet-5-5`, tool use, server-side refusal fallback on by default;
`LLM_REFUSAL_FALLBACK=false` disables it). Without a key it runs the deterministic offline
rule-based planner, so everything works and is testable with no credentials. Both sit behind the
same `Planner` interface and the same guardrails.

> **Note:** the agent stores sessions/runs/steps in its own `agent_*` tables, created automatically
> in the configured database on first use. With the default SQLite file that modifies
> `_DB/ecommerce.db` (a tracked file); point `DATABASE_URL` at a copy for local experiments.

### Example
```bash
K='-H X-API-Key:dev-public-key -H Content-Type:application/json'
curl -s $K localhost:8000/api/v1/agent/chat \
  -d '{"customer_id":1,"session_id":"SESSION1001","message":"I want a refund for ORD-100001"}'
# -> status "awaiting_confirmation", pending_confirmation{tool:"initiate_refund",
#    description:"refund $60.00 for order ORD-100001 to the original payment"}
curl -s $K localhost:8000/api/v1/agent/chat \
  -d '{"customer_id":1,"session_id":"SESSION1001","message":"yes"}'
# -> "Done - Refund of $60.00 to original payment approved within policy (REF-001)."
curl -s $K localhost:8000/api/v1/agent/sessions/SESSION1001/traces | jq '.items[0].steps[].summary'
```
`customer_id` is the numeric DB id (the brief's `CUST1001`-style ids don't exist in the schema).
In production the identity should come from the verified login/JWT, not the request body; sending
`X-Customer-Id` already forces the body value to match.

### Known limits
Rate limiter, idempotency cache and circuit breaker are per-process (swap for Redis to scale out);
KB retrieval is BM25 over `_KB` (swappable behind `kb.search`); the LLM planner path is unit-tested
with a fake client and has **not** been run against the live Anthropic API.
