# RFC: A Real-Time Budget Decision Plane for AI Agent Runs

**Status:** Draft v3, for feedback · **Author:** Ajay Rajput · **Date:** July 2026
**Audience:** Platform/infra engineers running LLM gateways in production

---

## 1. Problem

AI agents don't consume tokens the way chat does. An agent runs a loop: observe, think, act, repeat, and each iteration resends the accumulated context. By step 20 of a run with file reads, a single call can exceed 50K input tokens. Reported cases from the past year include a developer hitting $4,200 in API fees over one weekend of autonomous refactoring and a 35-engineer team receiving an $87K monthly bill; one audit of 30 teams found a 20x spread between p10 and p90 per-developer cost for the same tooling. These figures come from industry writeups rather than primary incident reports, but the mechanism they describe, unbounded loops resending growing context, is structural and reproducible.

Three gaps make this hard to control today:

**Budgets attach to the wrong unit.** Existing gateway budgets attach to API keys, users, or teams, over accounting periods measured in days or months. The damage unit for agents is the **run**: one autonomous session that needs a ceiling in dollars, not a monthly quota it can exhaust in an hour. No mainstream gateway enforces per-run ceilings.

**Budget enforcement is implicit and fragile in incumbent gateways.** Recent budget-enforcement regressions in LiteLLM (e.g. [#26672](https://github.com/BerriAI/litellm/issues/26672), [#27381](https://github.com/BerriAI/litellm/issues/27381), [#27480](https://github.com/BerriAI/litellm/issues/27480), with new budget issues continuing to appear) illustrate the underlying design problem: enforcement implemented as scattered callbacks with no explicit authorization step is hard to test and easy to silently break. Separately, models with missing price metadata have been treated as free, bypassing all budget checks, and team-level enforcement sits behind an enterprise paywall. The lesson is not "gateway X is buggy"; it is that budget authority should be an explicit, testable decision point with stated guarantees.

**Enforcement is blind, so agents can't adapt.** When a budget check fails today, the request dies with an opaque error. The agent never learns it was *approaching* a limit, so it can't do what a cost-conscious human does: downshift to a cheaper model for routine steps, narrow context, or wrap up. Visibility without a feedback channel produces bill shock; blocking without one produces broken runs. Neither changes agent behavior.

## 2. Thesis

AI agent spend control needs a **real-time budget decision plane**. This RFC defines a run-scoped budget authority that atomically reserves estimated spend before provider calls, reconciles against actual usage after calls, fails closed on unknown prices, and exposes machine-readable budget state so agents can adapt before they exhaust the run.

## 3. Goals

1. **Per-run budget ceilings** enforced *before* the provider call, with stated correctness guarantees under concurrency.
2. **A machine-readable budget-state protocol** (response headers plus RFC 9457 problem-detail errors) that lets agents adapt mid-run instead of failing blind, without mutating successful provider responses.
3. **Per-run cost attribution** rolling up to user, feature, and team, without depending on provider-side billing tags.
4. **Fail-closed pricing:** a model with no known price is unroutable unless an explicit tenant override exists.

### Non-goals

- **Not a model gateway or provider abstraction.** This is a budget-decision plane that can be embedded as a gateway hook, sidecar, or SDK middleware. Successful provider responses pass through unmodified.
- Not agent-efficiency tooling (context compaction, caching, sub-agents). The frameworks and labs own that layer.
- Not post-hoc cost dashboards alone. Attribution exists here to make enforcement trustworthy.

## 4. Design

### 4.1 Concepts

- **Run**: one agent session. See identity rules in 4.2.
- **Ceiling**: a USD limit attached to a scope (run, user, team, key, feature tag).
- **Budget Decision**: the central primitive. The pre-call authorization result produced by the authority: `allow`, `downgrade`, `advisory_warn`, or `block` (reservation is the internal action backing `allow`/`downgrade`, not a client-facing decision value). Every decision has an ID and is logged with its inputs (scopes, estimates, effective output cap, price table version).
- **Reservation**: an atomic hold of estimated cost against one or more scopes, made before forwarding, committed or released after.
- **Ledger** (per scope): `committed_usd`, `reserved_usd`, `available_usd = limit_usd − committed_usd − reserved_usd`, plus reservation records (`reservation_id`, `expires_at`, `price_table_version`).
- **Estimate**: pre-call cost projection from the price table and token counts. In hard-gate mode the reservation basis is worst-case: actual input tokens plus `effective_max_output_tokens` at output price (see 4.4).

### 4.2 Run identity

Client-supplied run IDs are convenient and untrustworthy. Rules:

- `X-Run-Id` is accepted only from authenticated callers and is bound server-side to the authenticated key/user/team. A run ID cannot be attached to a different principal than the one that created it.
- Absent a run ID, the authority issues a server-side run ID and returns it in response headers.
- All ledger writes bind the full tuple: `run_id + user_id + key_id + team_id + feature_id`.
- Cardinality controls apply (see 4.9): max active runs per principal, run TTL.

### 4.3 Decision flow (reserve → commit → refund)

```
request → resolve scopes (run, user, team, key, feature)
        → compute effective_max_output_tokens (4.4)
        → estimate cost of requested model (worst-case basis in hard_gate)
        → ATOMIC: reserve estimate against ALL applicable scopes, or fail
            ├─ all scopes fit                → forward request
            ├─ blocked, valid alternative    → downgrade (policy-controlled) or block with alternatives
            └─ blocked, no alternative       → block with problem-detail error
        → on provider success: commit actual cost, release unused reserve
        → on provider failure: release reserve
        → on missing result: reservation expires after TTL, reconciled asynchronously
        → attach budget-state headers to every response
```

Reservation across multiple scopes is a single atomic transaction (one Redis Lua script or one SQL transaction): all scopes reserve or none do. Sequential per-scope locking is explicitly rejected (partial reservations, deadlocks). A request may fit the run ceiling and still be blocked by the user ceiling; the decision reports the **blocking scope**, but the transaction touches all scopes.

**Downgrade semantics.** If downgrade is selected, the authority reserves against the selected alternative model, not the originally requested model. Auto-downgrade is allowed only when the alternative satisfies capability contracts (4.8) and tenant policy permits downgrade for the request class.

**Reservation state machine.** Idempotency is defined over explicit states:

```
reserved → forwarded → committed
reserved → released
reserved → expired → reconciled
```

All transitions are idempotent. A commit for an already-committed reservation is ignored. A release after commit is ignored. A retry carrying the same idempotency key returns the existing decision rather than creating a new reservation.

### 4.4 Effective output cap

The hard-gate guarantee depends on who controls `max_output_tokens`. The authority estimates against `effective_max_output_tokens`, never blindly against the client-supplied value:

```
effective_max_output_tokens =
  min(
    client_requested_max_output_tokens,
    tenant_policy_max_output_tokens,
    model_context_remaining_output_limit,
    budget_derived_max_output_tokens   # opt-in only, see below
  )

budget_derived_max_output_tokens is treated as ∞ unless tenant opt-in
clamping is enabled; the min() never budget-clamps by default.
```

If the client omits `max_output_tokens`, enforce mode applies a tenant default or rejects the request. If the client requests a value above tenant policy, the authority clamps or rejects according to policy.

**Budget-derived clamping is opt-in tenant policy, never default.** Shrinking a generation to fit remaining budget is a behavior change disguised as accounting: a truncated diff or half-written JSON can be worse than a clean block. When enabled and applied, clamping must be visible (`X-Budget-Output-Clamped: true`, plus both values in the decision record and problem body).

### 4.5 Enforcement modes

Per tenant/scope, answering "how wrong can the estimate be?" explicitly:

| Mode | Behavior |
|---|---|
| `advisory_estimate` | Log and emit headers only. Nothing blocked. |
| `soft_gate` | Block only if estimate exceeds remaining by a configured safety margin. |
| `hard_gate` | Atomic worst-case reservation required before forwarding. |
| `actuals_only` | Allow until committed spend reaches the limit, then block new calls. No estimation trust required. |

Mode and decision are different dimensions: mode is configuration, decision is the per-request outcome (see header set in 4.7). The recommended adoption path is staged: advisory → downgrade-permitted → blocking. Teams watch advisory numbers before enabling anything that can touch a production run.

### 4.6 Ledger implementation invariants

- **Money precision:** all ledger amounts are stored as integer micro-USD. API responses render decimal strings. Floating-point arithmetic is forbidden in reservation, commit, and reconciliation paths.
- **Redis Cluster:** all ledger keys touched by one reservation script must share a hash tag (e.g. `{tenant_id}:budget:scope:...`) so the multi-scope Lua transaction stays single-slot.
- **SQLite fallback** is single-node/dev mode only and is not a multi-instance production ledger.
- **Price versioning:** every decision records provider, model, input price, output price, cache-read and cache-write prices, currency, and `price_table_version`. Tenant-specific price overrides are supported. Unknown price is unroutable unless an explicit tenant override exists.

### 4.7 Budget-state protocol

**Successful provider responses are not body-mutated by default.** The authority must not break OpenAI-compatible response contracts, SDKs, streaming clients, or eval harnesses. Headers expose the decision and the tightest applicable scope; full multi-scope state is available through logs, audit export, or a lookup endpoint:

```
GET /budget/decisions/{decision_id}
```

Decision records are retained for a configurable window (default 30 days) so the lookup promise is meaningful. An **optional envelope mode** may include full budget state in response bodies for clients that explicitly opt in.

Response headers:

```
X-Budget-Decision: allow | downgrade | advisory_warn | block
X-Budget-Decision-Id: bdgdec_01J...
X-Budget-Reservation-Id: rsv_01J...       (when a reservation was made)
X-Budget-Enforcement-Mode: advisory_estimate | soft_gate | hard_gate | actuals_only
X-Budget-Blocking-Scope: run              (present when constrained)
X-Budget-Remaining-USD: 2.86              (tightest applicable scope)
X-Budget-Requested-Model: claude-sonnet   (on downgrade)
X-Budget-Selected-Model: claude-haiku     (on downgrade)
X-Budget-Output-Clamped: true             (only when clamping applied)
X-Budget-Price-Table-Version: 2026-07-04
X-Run-Id: run_abc                         (echoed or server-issued)
```

Blocked requests return `402 Payment Required` by default (status configurable, since RFC 9110 still reserves 402 for future use) with an RFC 9457 `application/problem+json` body:

```json
{
  "type": "https://modelmuxer.dev/problems/budget-exceeded",
  "title": "Budget exceeded",
  "status": 402,
  "detail": "Estimated request cost exceeds the remaining run budget.",
  "code": "run_ceiling_reached",
  "budget": {
    "scope": "run",
    "run_id": "run_abc",
    "limit_usd": "5.00",
    "committed_usd": "4.91",
    "reserved_usd": "0.00",
    "remaining_usd": "0.09",
    "estimate_usd": "0.31",
    "effective_max_output_tokens": 4096,
    "client_requested_max_output_tokens": 8192,
    "price_table_version": "2026-07-04"
  },
  "alternatives": [
    {
      "model": "claude-haiku",
      "estimate_usd": "0.04",
      "fits_budget": true,
      "supports_tools": true,
      "supports_json_schema": true,
      "context_window_tokens": 200000,
      "downgrade_class": "routine_reasoning",
      "policy_reason": "allowed_for_non_write_step"
    }
  ]
}
```

### 4.8 Capability-valid alternatives

`alternatives` are **policy-valid alternatives**, not merely cheaper models: an alternative must satisfy the request's capability requirements (tool calling, JSON mode, context length, modality, tenant allow-lists) before it is offered. This is the feedback channel that turns enforcement into a steering signal: agents downshift for routine steps and reserve expensive models for hard ones.

### 4.9 Streaming

- Pre-call estimates require an output cap; 4.4 defines resolution when the client omits one.
- For streaming responses, actual usage is committed when final usage metadata is received.
- Mid-stream termination is **out of scope for v1**; where providers expose incremental usage telemetry, mid-stream enforcement is a stated future extension.

### 4.10 Abuse and failure modes

The authority is itself an attack surface and a dependency. v1 must define:

- **Cardinality:** limits on distinct run IDs and max active runs per key/user; run TTL.
- **Idempotency:** decision requests carry idempotency keys; the state machine in 4.3 makes retries safe.
- **Reservation hygiene:** TTL on all reservations; async reconciliation job for orphans; clock-skew tolerance on expiry.
- **Identity:** header spoofing prevented by binding run IDs to authenticated principals (4.2); tenant isolation on all ledger keys.
- **Backend outage (Redis/DB down):** advisory and downgrade modes fail open. Blocking modes are configurable fail-open or fail-closed per tenant, default fail-closed, loudly alarmed either way.

## 5. Correctness guarantees

In hard-gate mode, the authority guarantees that **no request is forwarded unless its estimated cost has been atomically reserved against every applicable ceiling.** Concurrent requests within the same run share one reservation pool; the "ten parallel calls each see $4.80/$5.00 and all pass" race cannot occur, because reservation, not read-then-check, is the gate.

The authority bounds forwarded exposure by reserving worst-case cost before the provider call. **Final invoice exposure is bounded when the provider honors `effective_max_output_tokens` and the active price table correctly models all billable token classes for that request** (including cached input, cache writes, reasoning tokens, and modality-specific token types). Where a provider bills dimensions the price table does not model, the bound does not hold for those dimensions; unmodeled billing classes are a price-table completeness problem, surfaced by reconciliation, not silently absorbed.

Reservations expire if no provider result is observed within the configured TTL and are reconciled asynchronously.

## 6. Delivery and v1 scope

Phase 1 is a LiteLLM integration (`async_pre_call_hook` for decisions, post-call callback for commit/refund), installable next to an existing deployment **without replacing the gateway or changing provider integrations** (configuration, identity binding, price tables, and ledger deployment are still required). Phase 2 is a standalone sidecar for teams not on LiteLLM. The protocol spec is published independently so any gateway or agent framework can implement it. Everything is Apache 2.0.

**In v1:** enforceable ceilings for run, user, and key; attribution tags for feature and team; atomic reservation ledger (Redis, SQLite for dev); all four enforcement modes; problem-detail errors; capability-checked alternatives; price table versioning; decision lookup endpoint; attribution API.
**Explicitly after v1:** enforceable team and feature ceilings; mid-stream enforcement; multi-region ledger; spend-velocity (trajectory-based) controls; non-USD currencies; optional response envelope mode.

## 7. Open questions, feedback wanted

1. Would your team ever enable **blocking** for production agent runs, or is auto-downgrade the ceiling of acceptable enforcement? What guarantees would blocking require beyond section 5?
2. How do you identify an agent run today: header, OTel trace/span, framework-level ID, or not at all?
3. Which enforcement mode would you actually start with, and is `actuals_only` (no estimation trust required) the necessary on-ramp?
4. Is worst-case reservation (`effective_max_output_tokens` basis) acceptable, or does it over-reserve so aggressively for your parallel workloads that soft-gate margins are preferable?
5. Would clients tolerate **headers-only budget state on successful responses**, with full state via the decision lookup endpoint, or do they need the opt-in response envelope?
6. Where should this live: gateway plugin, sidecar, or SDK middleware inside the agent framework?
7. What's missing from the problem-detail schema for your agents to act on it automatically?

## 8. Risks and alternatives considered

**Incumbents fix enforcement**: likely over time; the durable value is run-scoped ceilings, reservation semantics, and the feedback protocol, which are absent from incumbent designs rather than merely buggy. **Agents ignore the headers**: mitigated by adapter snippets for common frameworks, and the protocol remains useful to humans (dashboards, audit) even when agents ignore it. **Estimation is wrong**: bounded by mode choice; `actuals_only` requires no estimation trust at all. **Over-reservation starves parallel runs**: real trade-off of worst-case basis; mitigated by per-tenant margin tuning and fast commit-refund cycles. **Provider billing dimensions drift**: reconciliation surfaces unmodeled token classes as price-table gaps rather than absorbing them silently. **"Just use provider spend limits"**: per-account and monthly; they cannot see runs, features, or users, and they fail open across multiple providers.
