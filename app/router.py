# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""
Heuristic routing logic for selecting the best LLM provider and model.
"""

import logging
import operator
import re
import time
from typing import Callable, TypedDict, cast

from app.core.costing import (Estimator, LatencyPriors, estimate_tokens,
                              load_price_table)
from app.core.exceptions import BudgetExceededError, NoProvidersAvailableError
from app.core.intent import classify_intent
from app.models import ChatMessage
from app.providers.base import LLMProviderAdapter, ProviderResponse
from app.settings import settings
from app.telemetry.metrics import (LLM_ROUTER_BUDGET_EXCEEDED_TOTAL,
                                   LLM_ROUTER_COST_ESTIMATE_USD_SUM,
                                   LLM_ROUTER_DOWN_ROUTE_TOTAL,
                                   LLM_ROUTER_ETA_MS_BUCKET,
                                   LLM_ROUTER_SELECTED_COST_ESTIMATE_USD,
                                   LLM_ROUTER_UNPRICED_MODELS_SKIPPED,
                                   ROUTER_DECISION_LATENCY, ROUTER_FALLBACKS,
                                   ROUTER_INTENT_TOTAL, ROUTER_REQUESTS)
from app.telemetry.tracing import \
    start_span as \
    start_span  # backward-compat attribute expected by some tests
from app.telemetry.tracing import start_span_async

# Note: providers will be imported at runtime to avoid circular imports

logger = logging.getLogger(__name__)


class HeuristicRouter:
    """Intelligent router that selects the best LLM based on prompt characteristics."""

    def __init__(self, provider_registry_fn: Callable[[], dict[str, object]] | None = None):
        """
        Initialize HeuristicRouter with optional provider registry dependency.

        Args:
            provider_registry_fn: Function that returns provider registry dict.
                                 If None, uses default from providers.registry.
        """
        from app.settings import Settings as _Settings

        self.settings: _Settings = settings
        if provider_registry_fn is None:
            # Avoid capturing a direct function reference so tests can patch the module attribute
            import app.providers.registry as providers_registry

            self.provider_registry_fn = lambda: providers_registry.get_provider_registry()
        else:
            self.provider_registry_fn = provider_registry_fn
        try:
            from app.core.costing import Price

            self.price_table: dict[str, Price] = load_price_table(
                self.settings.pricing.price_table_path
            )
            self.latency_priors: LatencyPriors = LatencyPriors(
                self.settings.pricing.latency_priors_window_s
            )
            self.estimator: Estimator = Estimator(
                self.price_table, self.latency_priors, self.settings
            )
            if not self.price_table:
                logger.warning(
                    "Price table empty at %s; all models will be treated as unpriced and may trigger budget gating. Configure PRICING via PRICE_TABLE_PATH.",
                    self.settings.pricing.price_table_path,
                )
                if self.settings.features.mode == "production":
                    from app.core.exceptions import RouterConfigurationError

                    raise RouterConfigurationError(
                        f"Price table is empty in production mode at {self.settings.pricing.price_table_path}"
                    )
        except Exception as e:
            logger.error("Failed to initialize cost estimation: %s", e)
            self.price_table = {}
            self.latency_priors = LatencyPriors()
            self.estimator = Estimator({}, self.latency_priors, self.settings)
            logger.warning(
                "Price table empty at %s; all models will be treated as unpriced and may trigger budget gating. Configure PRICING via PRICE_TABLE_PATH.",
                self.settings.pricing.price_table_path,
            )

        self.code_patterns: list[str] = [
            r"```[\s\S]*?```",
            r"`[^`\n]+`",
            r"\bdef\s+\w+\s*\(",
            r"\bclass\s+\w+\s*[:\(]",
            r"\bfunction\s+\w+\s*\(",
            r"\bpublic\s+\w+\s+\w+\s*\(",
            r"\bimport\s+\w+",
            r"\bfrom\s+\w+\s+import",
            r"#include\s*<\w+>",
            r"\$\w+\s*=",
            r"SELECT\s+.*\s+FROM",
            r"CREATE\s+TABLE",
            r"<\w+[^>]*>.*</\w+>",
            r'{\s*"[\w":\s,\[\]{}]+\s*}',
        ]
        self.programming_keywords: list[str] = [
            "function",
            "class",
            "import",
            "export",
            "const",
            "let",
            "var",
            "def",
            "return",
            "if",
            "else",
            "elif",
            "for",
            "while",
            "try",
            "except",
            "public",
            "private",
            "protected",
            "static",
            "void",
            "int",
            "string",
            "array",
            "list",
            "dict",
            "object",
            "null",
            "undefined",
            "true",
            "false",
            "async",
            "await",
            "promise",
            "callback",
            "lambda",
            "yield",
            "SELECT",
            "INSERT",
            "UPDATE",
            "DELETE",
            "CREATE",
            "DROP",
            "ALTER",
            "html",
            "css",
            "javascript",
            "python",
            "java",
            "cpp",
            "sql",
            "json",
        ]
        self._compiled_keyword_patterns: list[re.Pattern[str]] = [
            re.compile(r"\b" + re.escape(keyword.lower()) + r"\b", re.IGNORECASE)
            for keyword in self.programming_keywords
        ]
        self.complexity_keywords: list[str] = [
            "analyze",
            "analysis",
            "explain",
            "explanation",
            "debug",
            "debugging",
            "reasoning",
            "reason",
            "complex",
            "complicated",
            "algorithm",
            "algorithms",
            "optimize",
            "optimization",
            "architecture",
            "design pattern",
            "patterns",
            "performance",
            "scalability",
            "trade-off",
            "tradeoffs",
            "comparison",
            "evaluate",
            "assessment",
            "review",
            "critique",
            "detailed",
            "comprehensive",
            "step-by-step",
            "methodology",
            "approach",
            "strategy",
            "framework",
            "implementation",
            "solution",
            "problem-solving",
            "troubleshoot",
        ]
        self.simple_indicators: list[str] = [
            "what is",
            "who is",
            "when is",
            "where is",
            "how much",
            "how many",
            "define",
            "definition",
            "meaning",
            "translate",
            "translation",
            "calculate",
            "convert",
            "list",
            "name",
            "tell me",
            "show me",
        ]
        self.model_preferences: dict[str, list[tuple[str, str]]] = {
            "code": [
                ("anthropic", "claude-3-5-sonnet-latest"),
                ("anthropic", "claude-3-5-sonnet-20241022"),
                ("anthropic", "claude-3-sonnet"),
                ("openai", "gpt-4o"),
                ("openai", "o1"),
                ("openai", "gpt-4"),
            ],
            "complex": [
                ("anthropic", "claude-3-opus-20240229"),
                ("anthropic", "claude-3-5-sonnet-latest"),
                ("openai", "o1"),
                ("openai", "gpt-4o"),
                ("anthropic", "claude-3-sonnet"),
            ],
            "simple": [
                ("openai", "gpt-3.5-turbo"),
                ("openai", "gpt-4o-mini"),
                ("anthropic", "claude-3-haiku-20240307"),
                ("anthropic", "claude-3-5-haiku-20241022"),
            ],
            "general": [
                ("openai", "gpt-3.5-turbo"),
                ("openai", "gpt-4o-mini"),
                ("anthropic", "claude-3-haiku-20240307"),
                ("anthropic", "claude-3-5-haiku-20241022"),
            ],
        }
        self._compiled_code_patterns: list[re.Pattern[str]] = [
            re.compile(pattern, re.IGNORECASE | re.MULTILINE) for pattern in self.code_patterns
        ]
        self._compiled_complexity_patterns: list[re.Pattern[str]] = [
            re.compile(r"\b" + re.escape(keyword.lower()) + r"\b", re.IGNORECASE)
            for keyword in self.complexity_keywords
        ]
        self._validate_model_keys()
        # Provide legacy attribute expected by direct tests
        self.direct_model_preferences = self.model_preferences

    def _validate_model_keys(self) -> None:
        candidate_keys: set[str] = {
            f"{provider}:{model}"
            for plist in self.model_preferences.values()
            for provider, model in plist
        }
        price_table_keys: set[str] = set(self.price_table.keys())
        missing_keys: set[str] = candidate_keys - price_table_keys
        if missing_keys:
            error_msg = (
                f"Model keys in preferences missing from price table: {sorted(missing_keys)}. "
                f"This will cause 'no_pricing' errors during routing. "
                f"Please add pricing for these models in {self.settings.pricing.price_table_path} or update model preferences."
            )
            if self.settings.features.mode == "production":
                logger.error(error_msg)
                from app.core.exceptions import RouterConfigurationError

                raise RouterConfigurationError(error_msg)
            else:
                logger.warning(error_msg)
        available_providers = self.provider_registry_fn()
        required_providers: set[str] = {
            provider for prefs in self.model_preferences.values() for provider, _ in prefs
        }
        missing_providers: set[str] = required_providers - set(available_providers.keys())
        if missing_providers:
            error_msg = (
                "Required providers not available: "
                f"{sorted(missing_providers)}. Available providers: {sorted(available_providers.keys())}. "
                f"This will prevent routing to preferred models."
            )
            if self.settings.features.mode == "production":
                logger.error(error_msg)
                from app.core.exceptions import RouterConfigurationError

                raise RouterConfigurationError(error_msg)
            else:
                logger.warning(error_msg)

    class Analysis(TypedDict):
        total_length: int
        message_count: int
        has_code: bool
        code_confidence: float
        has_complexity: bool
        complexity_confidence: float
        is_simple: bool
        simple_confidence: float
        detected_languages: list[str]
        task_type: str

    def analyze_prompt(self, messages: list[ChatMessage]) -> "HeuristicRouter.Analysis":
        full_text = " ".join([msg.content for msg in messages if msg.content])
        full_text_lower = full_text.lower()
        analysis: HeuristicRouter.Analysis = {
            "total_length": len(full_text),
            "message_count": len(messages),
            "has_code": False,
            "code_confidence": 0.0,
            "has_complexity": False,
            "complexity_confidence": 0.0,
            "is_simple": False,
            "simple_confidence": 0.0,
            "detected_languages": [],
            "task_type": "general",
        }
        code_matches = 0
        for pattern in self._compiled_code_patterns:
            matches = pattern.findall(full_text)
            code_matches += len(matches)
        programming_matches = 0
        for pattern in self._compiled_keyword_patterns:
            if pattern.search(full_text_lower):
                programming_matches += 1
        analysis["code_confidence"] = min(1.0, (code_matches * 0.3 + programming_matches * 0.1))
        analysis["has_code"] = (
            analysis["code_confidence"] >= settings.router.code_detection_threshold
        )
        complexity_matches = 0
        for pattern in self._compiled_complexity_patterns:
            if pattern.search(full_text_lower):
                complexity_matches += 1
        analysis["complexity_confidence"] = min(1.0, complexity_matches * 0.1)
        analysis["has_complexity"] = (
            analysis["complexity_confidence"] >= settings.router.complexity_threshold
        )
        simple_matches = 0
        for indicator in self.simple_indicators:
            if indicator.lower() in full_text_lower:
                simple_matches += 1
        analysis["simple_confidence"] = min(1.0, simple_matches * 0.2)
        analysis["is_simple"] = (
            analysis["simple_confidence"] >= settings.router.simple_query_threshold
            and analysis["total_length"] < settings.router.simple_query_max_length
            and analysis["message_count"] <= 2
        )
        if analysis["has_code"]:
            analysis["task_type"] = "code"
        elif analysis["has_complexity"]:
            analysis["task_type"] = "complex"
        elif analysis["is_simple"]:
            analysis["task_type"] = "simple"
        else:
            analysis["task_type"] = "general"
        return analysis

    async def select_model(
        self,
        messages: list[ChatMessage],
        user_id: str | None = None,
        budget_constraint: float | None = None,
        max_tokens: int | None = None,
    ) -> tuple[str, str, str, dict[str, object], dict[str, object]]:
        preferences: list[tuple[str, str]] = []
        intent_metadata: dict[str, object] = {
            "label": "unknown",
            "confidence": 0.0,
            "signals": {},
            "method": "disabled",
        }
        try:
            intent_metadata = await classify_intent(messages)
            ROUTER_INTENT_TOTAL.labels(cast(str, intent_metadata["label"]).__str__()).inc()
        except Exception as e:
            logger.warning("Intent classification failed: %s", e)
        analysis = self.analyze_prompt(messages)
        task_type = analysis["task_type"]
        route_label: str = "chat"
        ROUTER_REQUESTS.labels(route_label).inc()
        t0 = time.perf_counter()
        # Normalize intent confidence to float for tracing
        _conf_val = intent_metadata.get("confidence", 0.0)
        _conf: float = float(_conf_val) if isinstance(_conf_val, int | float) else 0.0
        async with start_span_async(
            "router.decide",
            route=route_label,
            task_type=task_type,
            code_confidence=analysis["code_confidence"],
            complexity_confidence=analysis["complexity_confidence"],
            simple_confidence=analysis["simple_confidence"],
            route_intent_label=str(intent_metadata.get("label", "unknown")),
            route_intent_confidence=_conf,
        ) as span:
            tokens_in, tokens_out = estimate_tokens(
                messages, self.settings, self.settings.pricing.min_tokens_in_floor
            )
            if max_tokens is not None:
                tokens_out = int(max_tokens)
            if span is not None:
                span.set_attribute("route.tokens_in", tokens_in)
                span.set_attribute("route.tokens_out", tokens_out)
                span.set_attribute("route.direct_providers_only", True)
                if user_id is not None:
                    span.set_attribute("route.user_id_len", len(user_id))
            preferences = self.model_preferences.get(
                task_type, self.model_preferences.get("general", [])
            )
            logger.debug(
                "Selected direct provider preferences for task '%s': %d models",
                task_type,
                len(preferences),
            )
            if not preferences:
                available_providers = self.provider_registry_fn()
                budget = (
                    budget_constraint
                    if budget_constraint is not None
                    else self.settings.router_thresholds.max_estimated_usd_per_request
                )
                if available_providers:
                    ROUTER_FALLBACKS.labels(route_label, "no_preferred_provider_available").inc()
                    raise BudgetExceededError(
                        f"No preferred providers available within budget limit of ${budget}",
                        limit=budget,
                        estimates=[],
                        reason="no_preferred_provider_available",
                    )
                else:
                    ROUTER_FALLBACKS.labels(route_label, "no_providers").inc()
                    raise NoProvidersAvailableError("No LLM providers available")
            original_preferences: list[tuple[str, str]] = preferences.copy()
            from app.core.costing import Estimate

            affordable_preferences: list[tuple[str, str, Estimate, float]] = []
            priced_models_count: int = 0
            for provider, model in preferences:
                model_key = f"{provider}:{model}"
                try:
                    estimate = self.estimator.estimate(model_key, tokens_in, tokens_out)
                except Exception as e:
                    logger.warning("Estimator failed for %s: %s", model_key, e)
                    continue
                if estimate.usd is None:
                    logger.warning("No pricing available for %s, skipping", model_key)
                    if span is not None:
                        span.add_event("unpriced_model_skipped", {"model_key": model_key})
                    LLM_ROUTER_UNPRICED_MODELS_SKIPPED.labels(route_label).inc()
                    continue
                priced_models_count += 1
                budget = (
                    budget_constraint
                    if budget_constraint is not None
                    else self.settings.router_thresholds.max_estimated_usd_per_request
                )
                # estimate.usd is not None due to guard above
                if estimate.usd <= budget:
                    usd_value: float = float(estimate.usd)
                    affordable_preferences.append((provider, model, estimate, usd_value))
                    LLM_ROUTER_COST_ESTIMATE_USD_SUM.labels(route_label, model_key, "true").inc(
                        usd_value
                    )
                    LLM_ROUTER_ETA_MS_BUCKET.labels(route_label, model_key).observe(
                        float(estimate.eta_ms)
                    )
                else:
                    budget = (
                        budget_constraint
                        if budget_constraint is not None
                        else self.settings.router_thresholds.max_estimated_usd_per_request
                    )
                    logger.info(
                        "Budget exceeded for %s: $%.4f > $%s", model_key, estimate.usd, budget
                    )
                    LLM_ROUTER_COST_ESTIMATE_USD_SUM.labels(route_label, model_key, "false").inc(
                        float(estimate.usd)
                    )
                    LLM_ROUTER_BUDGET_EXCEEDED_TOTAL.labels(route_label, "pre_gate_exceeded").inc()
            # sort by usd value (index 3)
            affordable_preferences.sort(key=operator.itemgetter(3))
            if not affordable_preferences:
                budget = (
                    budget_constraint
                    if budget_constraint is not None
                    else self.settings.router_thresholds.max_estimated_usd_per_request
                )
                if span is not None:
                    span.set_attribute("route.budget.limit", budget)
                    span.add_event("budget_exceeded")
                if priced_models_count == 0:
                    LLM_ROUTER_BUDGET_EXCEEDED_TOTAL.labels(route_label, "no_pricing").inc()
                    raise BudgetExceededError(
                        "No models have pricing; update PRICE_TABLE_PATH",
                        limit=budget,
                        estimates=[],
                        reason="no_pricing",
                    )
                else:
                    LLM_ROUTER_BUDGET_EXCEEDED_TOTAL.labels(route_label, "budget_exceeded").inc()
                    raise BudgetExceededError(
                        f"No models within budget limit of ${budget}",
                        limit=budget,
                        estimates=[
                            (
                                f"{p}:{m}",
                                self.estimator.estimate(f"{p}:{m}", tokens_in, tokens_out).usd,
                            )
                            for p, m in preferences[:3]
                        ],
                    )
            available_providers = self.provider_registry_fn()
            for provider, model, estimate, usd_value in affordable_preferences:
                adapter = available_providers.get(provider)
                # In selection phase, consider any available adapter acceptable (tests patch with MagicMock)
                if adapter is not None and not getattr(adapter, "circuit_open", False):
                    if span is not None:
                        span.set_attribute("route.estimate.usd", usd_value)
                        span.set_attribute("route.estimate.eta_ms", estimate.eta_ms)
                        span.set_attribute("route.tokens_in", estimate.tokens_in)
                        span.set_attribute("route.tokens_out", estimate.tokens_out)
                        span.set_attribute("route.model_key", estimate.model_key)
                        if self.settings.server.debug:
                            span.set_attribute(
                                "_route.estimate", f"${estimate.usd:.4f} | {estimate.eta_ms}ms"
                            )
                    model_key = f"{provider}:{model}"
                    LLM_ROUTER_SELECTED_COST_ESTIMATE_USD.labels(route_label, model_key).inc(
                        usd_value
                    )
                    if len(original_preferences) > 0 and original_preferences[0] != (
                        provider,
                        model,
                    ):
                        original_first = original_preferences[0]
                        try:
                            original_first_key = f"{original_first[0]}:{original_first[1]}"
                            original_first_estimate = self.estimator.estimate(
                                original_first_key, tokens_in, tokens_out
                            )
                            if original_first_estimate.usd is not None and usd_value < float(
                                original_first_estimate.usd
                            ):
                                LLM_ROUTER_DOWN_ROUTE_TOTAL.labels(
                                    route_label, original_first_key, f"{provider}:{model}"
                                ).inc()
                        except Exception:
                            pass
                    ROUTER_DECISION_LATENCY.labels(route_label).observe(
                        (time.perf_counter() - t0) * 1000
                    )
                    return (
                        provider,
                        model,
                        self._generate_reasoning(analysis),
                        intent_metadata,
                        {
                            "usd": usd_value,
                            "eta_ms": estimate.eta_ms,
                            "tokens_in": estimate.tokens_in,
                            "tokens_out": estimate.tokens_out,
                            "model_key": estimate.model_key,
                        },
                    )
            # Fallback: if we had affordable options but no adapter matched, pick the cheapest affordable anyway
            if affordable_preferences:
                provider, model, estimate, usd_value = affordable_preferences[0]
                if span is not None:
                    span.add_event(
                        "adapter_missing_fallback", {"provider": provider, "model": model}
                    )
                return (
                    provider,
                    model,
                    self._generate_reasoning(analysis),
                    intent_metadata,
                    {
                        "usd": usd_value,
                        "eta_ms": estimate.eta_ms,
                        "tokens_in": estimate.tokens_in,
                        "tokens_out": estimate.tokens_out,
                        "model_key": estimate.model_key,
                    },
                )
        available_providers = self.provider_registry_fn()
        preferred_providers_available: bool = False
        if preferences:
            preferred_providers_available = any(
                provider in available_providers for provider, _ in preferences
            )
        if not preferred_providers_available and available_providers:
            ROUTER_FALLBACKS.labels(route_label, "no_preferred_provider_available").inc()
            ROUTER_DECISION_LATENCY.labels(route_label).observe((time.perf_counter() - t0) * 1000)
            budget = (
                budget_constraint
                if budget_constraint is not None
                else self.settings.router_thresholds.max_estimated_usd_per_request
            )
            raise BudgetExceededError(
                f"No preferred providers available within budget limit of ${budget}",
                limit=budget,
                estimates=[],
                reason="no_preferred_provider_available",
            )
        elif available_providers:
            ROUTER_FALLBACKS.labels(route_label, "no_affordable_available").inc()
            ROUTER_DECISION_LATENCY.labels(route_label).observe((time.perf_counter() - t0) * 1000)
            budget = (
                budget_constraint
                if budget_constraint is not None
                else self.settings.router_thresholds.max_estimated_usd_per_request
            )
            raise BudgetExceededError(
                f"No affordable models available within budget limit of ${budget}",
                limit=budget,
                estimates=[],
                reason="no_affordable_available",
            )
        else:
            ROUTER_FALLBACKS.labels(route_label, "no_providers").inc()
            ROUTER_DECISION_LATENCY.labels(route_label).observe((time.perf_counter() - t0) * 1000)
            raise NoProvidersAvailableError("No LLM providers available")

    async def invoke_via_adapter(
        self, provider: str, model: str, prompt: str, **kwargs: object
    ) -> ProviderResponse:
        available = self.provider_registry_fn()
        adapter = available.get(provider)
        if not adapter:
            raise NoProvidersAvailableError(
                f"Provider adapter not available: {provider}",
                details={
                    "requested_provider": provider,
                    "requested_model": model,
                    "available_providers": list(available.keys()),
                },
            )
        # In direct-provider tests, adapters may be simple mocks; accept objects with 'invoke'
        if not isinstance(adapter, LLMProviderAdapter):
            import inspect

            if not hasattr(adapter, "invoke") or not inspect.iscoroutinefunction(adapter.invoke):
                raise NoProvidersAvailableError(
                    f"Provider does not implement adapter interface: {provider}",
                    details={"provider": provider},
                )
        return await adapter.invoke(model=model, prompt=prompt, **kwargs)

    def _generate_reasoning(self, analysis: "HeuristicRouter.Analysis") -> str:
        reasons: list[str] = []
        if analysis["has_code"]:
            reasons.append(f"Code detected (confidence: {analysis['code_confidence']:.2f})")
        if analysis["has_complexity"]:
            reasons.append(
                f"Complex analysis required (confidence: {analysis['complexity_confidence']:.2f})"
            )
        if analysis["is_simple"]:
            reasons.append(f"Simple query detected (length: {analysis['total_length']} chars)")
        if analysis["total_length"] > 1000:
            reasons.append("Long prompt requires capable model")
        if analysis["message_count"] > 5:
            reasons.append("Multi-turn conversation")
        task_reason = f"Task type: {analysis['task_type']}"
        if reasons:
            task_reason += f" ({', '.join(reasons)})"
        return task_reason

    def record_latency(self, model_key: str, ms: int) -> None:
        self.latency_priors.update(model_key, ms)

    def get_routing_stats(self) -> dict[str, object]:
        return {
            "total_requests": 0,
            "routing_by_task_type": {"code": 0, "complex": 0, "simple": 0, "general": 0},
            "routing_by_provider": {"openai": 0, "anthropic": 0, "mistral": 0},
        }


router = None
