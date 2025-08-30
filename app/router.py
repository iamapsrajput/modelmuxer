# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""
Heuristic routing logic for selecting the best LLM provider and model.
"""

import logging
import re
from typing import Any

from app.settings import settings
from app.core.costing import Estimator, LatencyPriors, load_price_table, estimate_tokens
from app.core.exceptions import BudgetExceededError, NoProvidersAvailableError
from app.models import ChatMessage

from app.telemetry.tracing import start_span
from app.telemetry.metrics import (
    ROUTER_REQUESTS,
    ROUTER_DECISION_LATENCY,
    ROUTER_FALLBACKS,
    ROUTER_INTENT_TOTAL,
    LLM_ROUTER_COST_ESTIMATE_USD_SUM,
    LLM_ROUTER_ETA_MS_BUCKET,
    LLM_ROUTER_BUDGET_EXCEEDED_TOTAL,
    LLM_ROUTER_DOWN_ROUTE_TOTAL,
    LLM_ROUTER_UNPRICED_MODELS_SKIPPED,
    LLM_ROUTER_SELECTED_COST_ESTIMATE_USD,
)
from app.core.intent import classify_intent
import time

# Note: providers will be imported at runtime to avoid circular imports

logger = logging.getLogger(__name__)


class HeuristicRouter:
    """Intelligent router that selects the best LLM based on prompt characteristics."""

    def __init__(self, provider_registry_fn=None):
        """
        Initialize HeuristicRouter with optional provider registry dependency.

        Args:
            provider_registry_fn: Function that returns provider registry dict.
                                 If None, uses default from providers.registry.
        """
        self.settings = settings
        # Store provider registry function to avoid circular import
        if provider_registry_fn is None:
            from app.providers.registry import get_provider_registry

            self.provider_registry_fn = get_provider_registry
        else:
            self.provider_registry_fn = provider_registry_fn
        # Initialize cost estimation components
        try:
            self.price_table = load_price_table(self.settings.pricing.price_table_path)
            self.latency_priors = LatencyPriors(self.settings.pricing.latency_priors_window_s)
            self.estimator = Estimator(self.price_table, self.latency_priors, self.settings)

            # Check if price table is empty and warn
            if not self.price_table:
                logger.warning(
                    "Price table empty at %s; all models will be treated as unpriced and may trigger budget gating. Configure PRICING via PRICE_TABLE_PATH.",
                    self.settings.pricing.price_table_path,
                )
                # Optionally raise ConfigurationError when running in production mode
                if self.settings.features.mode == "production":
                    from app.core.exceptions import ConfigurationError

                    raise ConfigurationError(
                        f"Price table is empty in production mode at {self.settings.pricing.price_table_path}"
                    )

        except Exception as e:
            logger.error(f"Failed to initialize cost estimation: {e}")
            # Fallback to empty price table
            self.price_table = {}
            self.latency_priors = LatencyPriors()
            self.estimator = Estimator({}, self.latency_priors, self.settings)

            # Warn about empty fallback price table
            logger.warning(
                "Price table empty at %s; all models will be treated as unpriced and may trigger budget gating. Configure PRICING via PRICE_TABLE_PATH.",
                self.settings.pricing.price_table_path,
            )

        # Code detection patterns
        self.code_patterns = [
            r"```[\s\S]*?```",  # Code blocks
            r"`[^`\n]+`",  # Inline code
            r"\bdef\s+\w+\s*\(",  # Python function definitions
            r"\bclass\s+\w+\s*[:\(]",  # Class definitions
            r"\bfunction\s+\w+\s*\(",  # JavaScript functions
            r"\bpublic\s+\w+\s+\w+\s*\(",  # Java/C# methods
            r"\bimport\s+\w+",  # Import statements
            r"\bfrom\s+\w+\s+import",  # Python imports
            r"#include\s*<\w+>",  # C/C++ includes
            r"\$\w+\s*=",  # Variable assignments (PHP, shell)
            r"SELECT\s+.*\s+FROM",  # SQL queries
            r"CREATE\s+TABLE",  # SQL DDL
            r"<\w+[^>]*>.*</\w+>",  # HTML/XML tags
            r'{\s*"[\w":\s,\[\]{}]+\s*}',  # JSON objects
        ]

        # Programming language keywords
        self.programming_keywords = [
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

        # Pre-compile regex patterns for efficiency
        self._compiled_keyword_patterns = [
            re.compile(r"\b" + re.escape(keyword.lower()) + r"\b", re.IGNORECASE)
            for keyword in self.programming_keywords
        ]

        # Complexity indicators
        self.complexity_keywords = [
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

        # Simple query indicators
        self.simple_indicators = [
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

        # Direct Provider Model Preferences
        # Format: ("provider_name", "model_name")
        # Provider names must match registry keys: openai, anthropic, mistral, groq, google, cohere, together
        # Model names must be supported by the respective provider adapters
        # ModelMuxer now uses direct provider connections as the primary routing method
        # This provides better control, lower latency, and enhanced error handling
        self.model_preferences = {
            "code": [
                # Best Claude models for code (direct provider connections)
                ("anthropic", "claude-3-5-sonnet-latest"),  # Latest Claude 3.5
                ("anthropic", "claude-3-5-sonnet-20241022"),  # Specific Claude 3.5
                ("anthropic", "claude-3-sonnet"),  # Standard Claude 3 Sonnet
                ("openai", "gpt-4o"),  # GPT-4o for code
                ("openai", "o1"),  # OpenAI O1 reasoning model
                ("openai", "gpt-4"),  # Standard GPT-4
            ],
            "complex": [
                # Best models for complex reasoning (direct provider connections)
                ("anthropic", "claude-3-opus-20240229"),  # Claude Opus for deep thinking
                ("anthropic", "claude-3-5-sonnet-latest"),  # Latest Claude 3.5
                ("openai", "o1"),  # OpenAI O1 reasoning model
                ("openai", "gpt-4o"),  # GPT-4o for complex tasks
                ("anthropic", "claude-3-sonnet"),  # Standard Claude 3 Sonnet
            ],
            "simple": [
                # Cost-effective models for simple queries (direct provider connections)
                ("openai", "gpt-3.5-turbo"),  # Standard GPT-3.5
                ("openai", "gpt-4o-mini"),  # Cheaper GPT-4o variant
                ("anthropic", "claude-3-haiku-20240307"),  # Fast Claude
                ("anthropic", "claude-3-5-haiku-20241022"),  # Latest Claude Haiku
            ],
            "general": [
                # Balanced models for general use (direct provider connections)
                ("openai", "gpt-3.5-turbo"),  # Reliable general model
                ("openai", "gpt-4o-mini"),  # Better general model
                ("anthropic", "claude-3-haiku-20240307"),  # Claude for variety
                ("anthropic", "claude-3-5-haiku-20241022"),  # Latest Claude Haiku
            ],
        }

        # Pre-compile regex patterns for efficiency after all attributes are defined
        self._compiled_code_patterns = [
            re.compile(pattern, re.IGNORECASE | re.MULTILINE) for pattern in self.code_patterns
        ]

        self._compiled_complexity_patterns = [
            re.compile(r"\b" + re.escape(keyword.lower()) + r"\b", re.IGNORECASE)
            for keyword in self.complexity_keywords
        ]

        # Validate model key compatibility after all attributes are defined
        self._validate_model_keys()

    def _validate_model_keys(self) -> None:
        """Validate that model preferences align with available pricing."""
        # Compute candidate keys from model preferences
        candidate_keys = set()
        for plist in self.model_preferences.values():
            for provider, model in plist:
                candidate_keys.add(f"{provider}:{model}")

        # Compare with price table keys
        price_table_keys = set(self.price_table.keys())
        missing_keys = candidate_keys - price_table_keys

        if missing_keys:
            error_msg = (
                f"Model keys in preferences missing from price table: {sorted(missing_keys)}. "
                f"This will cause 'no_pricing' errors during routing. "
                f"Please add pricing for these models in {self.settings.pricing.price_table_path} or update model preferences."
            )
            if self.settings.features.mode == "production":
                logger.error(error_msg)
                # Raise exception to fail startup validation in production mode only
                from app.core.exceptions import RouterConfigurationError

                raise RouterConfigurationError(error_msg)
            else:
                logger.warning(error_msg)

        # Validate direct provider availability
        available_providers = self.provider_registry_fn()
        required_providers = set()
        for prefs in self.model_preferences.values():
            for provider, _ in prefs:
                required_providers.add(provider)

        missing_providers = required_providers - set(available_providers.keys())
        if missing_providers:
            error_msg = (
                "Required providers not available: "
                f"{sorted(missing_providers)}. Available providers: {sorted(available_providers.keys())}. "
                f"This will prevent routing to preferred models."
            )
            if self.settings.features.mode == "production":
                logger.error(error_msg)
                # Raise exception to fail startup validation in production mode only
                from app.core.exceptions import RouterConfigurationError

                raise RouterConfigurationError(error_msg)
            else:
                logger.warning(error_msg)

    def analyze_prompt(self, messages: list[ChatMessage]) -> dict[str, Any]:
        """Analyze prompt characteristics to determine routing strategy."""
        # Combine all message content for analysis
        full_text = " ".join([msg.content for msg in messages if msg.content])
        full_text_lower = full_text.lower()

        analysis = {
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

        # Code detection
        code_matches = 0
        for pattern in self._compiled_code_patterns:
            matches = pattern.findall(full_text)
            code_matches += len(matches)

        # Programming keyword detection (using word boundaries to avoid false positives)
        programming_matches = 0
        for pattern in self._compiled_keyword_patterns:
            if pattern.search(full_text_lower):
                programming_matches += 1

        # Calculate code confidence
        analysis["code_confidence"] = min(1.0, (code_matches * 0.3 + programming_matches * 0.1))
        analysis["has_code"] = analysis["code_confidence"] >= settings.router.code_detection_threshold

        # Complexity detection (using word boundaries to avoid false positives)
        # Complexity analysis
        complexity_matches = 0
        for pattern in self._compiled_complexity_patterns:
            if pattern.search(full_text_lower):
                complexity_matches += 1

        # Calculate complexity confidence
        analysis["complexity_confidence"] = min(1.0, complexity_matches * 0.1)
        analysis["has_complexity"] = analysis["complexity_confidence"] >= settings.router.complexity_threshold

        # Simple query detection
        simple_matches = 0
        for indicator in self.simple_indicators:
            if indicator.lower() in full_text_lower:
                simple_matches += 1

        # Calculate simple confidence
        analysis["simple_confidence"] = min(1.0, simple_matches * 0.2)
        analysis["is_simple"] = (
            analysis["simple_confidence"] >= settings.router.simple_query_threshold
            and analysis["total_length"] < settings.router.simple_query_max_length
            and analysis["message_count"] <= 2
        )

        # Determine task type
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
    ) -> tuple[str, str, str, dict[str, Any], dict[str, Any]]:
        """
        Select the best provider and model for the given messages.

        Returns:
            Tuple of (provider, model, reasoning, intent_metadata, estimate_metadata)
        """
        # Initialize preferences to avoid UnboundLocalError on exceptions
        preferences = []

        # Classify intent first
        intent_metadata = {"label": "unknown", "confidence": 0.0, "signals": {}, "method": "disabled"}
        try:
            intent_metadata = await classify_intent(messages)
            ROUTER_INTENT_TOTAL.labels(intent_metadata["label"]).inc()
        except Exception as e:
            # Log error but continue with routing
            logger.warning(f"Intent classification failed: {e}")

        analysis = self.analyze_prompt(messages)
        task_type = analysis["task_type"]
        route_label = "chat"
        ROUTER_REQUESTS.labels(route_label).inc()
        t0 = time.perf_counter()
        with start_span(
            "router.decide",
            route=route_label,
            task_type=task_type,
            code_confidence=analysis.get("code_confidence"),
            complexity_confidence=analysis.get("complexity_confidence"),
            simple_confidence=analysis.get("simple_confidence"),
            route_intent_label=intent_metadata.get("label"),
            route_intent_confidence=intent_metadata.get("confidence"),
        ) as span:
            # Estimate tokens for cost calculation using encapsulated helper
            tokens_in, tokens_out = estimate_tokens(messages, self.settings, self.settings.pricing.min_tokens_in_floor)

            # Override output tokens if max_tokens is specified
            if max_tokens is not None:
                tokens_out = int(max_tokens)

            # Set span attributes for tokens after calculation
            if span is not None:
                span.set_attribute("route.tokens_in", tokens_in)
                span.set_attribute("route.tokens_out", tokens_out)
                # Add span attribute to track routing mode
                span.set_attribute("route.direct_providers_only", True)

            # Use direct provider preferences
            preferences = self.model_preferences.get(task_type, self.model_preferences.get("general", []))

            # Add logging for preference selection
            logger.debug(f"Selected direct provider preferences for task '{task_type}': {len(preferences)} models")

            # After setting `preferences` and before iterating for budget gating
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

            original_preferences = preferences.copy()  # Preserve original order for down-routing metrics

            # Apply budget gate using new Estimator
            affordable_preferences = []
            priced_models_count = 0
            for provider, model in preferences:
                model_key = f"{provider}:{model}"
                try:
                    estimate = self.estimator.estimate(model_key, tokens_in, tokens_out)
                except Exception as e:
                    logger.warning(f"Estimator failed for {model_key}: {e}")
                    continue

                # Check budget constraint - skip models with no pricing
                if estimate.usd is None:
                    # Log unknown model pricing
                    logger.warning(f"No pricing available for {model_key}, skipping")
                    # Add span event if span exists
                    if span is not None:
                        span.add_event("unpriced_model_skipped", {"model_key": model_key})
                    # Increment metric
                    LLM_ROUTER_UNPRICED_MODELS_SKIPPED.labels(route_label).inc()
                    continue

                priced_models_count += 1

                # Use per-request budget override if provided, otherwise use default threshold
                budget = (
                    budget_constraint
                    if budget_constraint is not None
                    else self.settings.router_thresholds.max_estimated_usd_per_request
                )
                if estimate.usd <= budget:
                    affordable_preferences.append((provider, model, estimate))
                    # Record cost estimate metrics for affordable candidates
                    LLM_ROUTER_COST_ESTIMATE_USD_SUM.labels(route_label, model_key, "true").inc(estimate.usd)
                    LLM_ROUTER_ETA_MS_BUCKET.labels(route_label, model_key).observe(estimate.eta_ms)
                else:
                    # Log down-routing attempt
                    budget = (
                        budget_constraint
                        if budget_constraint is not None
                        else self.settings.router_thresholds.max_estimated_usd_per_request
                    )
                    logger.info(f"Budget exceeded for {model_key}: ${estimate.usd:.4f} > ${budget}")
                    # Record cost estimate metrics for over-budget candidates
                    LLM_ROUTER_COST_ESTIMATE_USD_SUM.labels(route_label, model_key, "false").inc(estimate.usd)
                    # Record budget exceeded metric
                    LLM_ROUTER_BUDGET_EXCEEDED_TOTAL.labels(route_label, "pre_gate_exceeded").inc()

            # Sort by cost (cheapest first)
            affordable_preferences.sort(key=lambda x: x[2].usd)

            if not affordable_preferences:
                # Add span attributes for budget failure
                budget = (
                    budget_constraint
                    if budget_constraint is not None
                    else self.settings.router_thresholds.max_estimated_usd_per_request
                )
                if span is not None:
                    span.set_attribute("route.budget.limit", budget)
                    span.add_event("budget_exceeded")

                if priced_models_count == 0:
                    # No models have pricing - raise specific error
                    LLM_ROUTER_BUDGET_EXCEEDED_TOTAL.labels(route_label, "no_pricing").inc()
                    raise BudgetExceededError(
                        "No models have pricing; update PRICE_TABLE_PATH",
                        limit=budget,
                        estimates=[],
                        reason="no_pricing",
                    )
                else:
                    # Models have pricing but exceed budget - regular budget exceeded error
                    raise BudgetExceededError(
                        f"No models within budget limit of ${budget}",
                        limit=budget,
                        estimates=[
                            (f"{p}:{m}", self.estimator.estimate(f"{p}:{m}", tokens_in, tokens_out).usd)
                            for p, m in preferences[:3]
                        ],  # Top 3 estimates
                    )

            # Get providers from registry to check availability
            available_providers = self.provider_registry_fn()

            # Select the first available model from affordable preferences
            for i, (provider, model, estimate) in enumerate(affordable_preferences):
                # Check if the provider is actually available
                if provider in available_providers:
                    # Set telemetry attributes within span context (if span is available)
                    if span is not None:
                        span.set_attribute("route.estimate.usd", estimate.usd)
                        span.set_attribute("route.estimate.eta_ms", estimate.eta_ms)
                        span.set_attribute("route.tokens_in", estimate.tokens_in)
                        span.set_attribute("route.tokens_out", estimate.tokens_out)
                        span.set_attribute("route.model_key", estimate.model_key)

                        # Add debug metadata with underscore prefix when in debug mode
                        if self.settings.server.debug:
                            span.set_attribute("_route.estimate", f"${estimate.usd:.4f} | {estimate.eta_ms}ms")

                    # Record selected model cost estimate metric
                    model_key = f"{provider}:{model}"
                    LLM_ROUTER_SELECTED_COST_ESTIMATE_USD.labels(route_label, model_key).inc(estimate.usd)

                    # Record down-routing if we selected a model other than the first original preference
                    # and the selected model is cheaper than the first original preference
                    if len(original_preferences) > 0 and original_preferences[0] != (provider, model):
                        original_first = original_preferences[0]
                        try:
                            original_first_key = f"{original_first[0]}:{original_first[1]}"
                            original_first_estimate = self.estimator.estimate(original_first_key, tokens_in, tokens_out)
                            if original_first_estimate.usd is not None and estimate.usd < original_first_estimate.usd:
                                LLM_ROUTER_DOWN_ROUTE_TOTAL.labels(
                                    route_label, original_first_key, f"{provider}:{model}"
                                ).inc()
                        except Exception:
                            # If we can't estimate the original first choice, skip down-routing metric
                            pass

                    ROUTER_DECISION_LATENCY.labels(route_label).observe((time.perf_counter() - t0) * 1000)
                    return (
                        provider,
                        model,
                        self._generate_reasoning(analysis, provider, model),
                        intent_metadata,
                        {
                            "usd": estimate.usd,
                            "eta_ms": estimate.eta_ms,
                            "tokens_in": estimate.tokens_in,
                            "tokens_out": estimate.tokens_out,
                            "model_key": estimate.model_key,
                        },
                    )

        # Fallback: Check if no preferred providers are available vs no affordable models
        available_providers = self.provider_registry_fn()

        # Check if any of the preferred providers are available (guard against empty preferences)
        preferred_providers_available = False
        if preferences:
            preferred_providers_available = any(provider in available_providers for provider, _ in preferences)

        if not preferred_providers_available and available_providers:
            # Preferred providers not available but other providers exist
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
            # No affordable models available - raise BudgetExceededError
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

    async def invoke_via_adapter(self, provider: str, model: str, prompt: str, **kwargs: Any):
        """Invoke a model via the unified adapter registry."""
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
        return await adapter.invoke(model=model, prompt=prompt, **kwargs)

    def _generate_reasoning(self, analysis: dict, provider: str, model: str) -> str:
        """Generate human-readable reasoning for the routing decision."""
        reasons = []

        if analysis["has_code"]:
            reasons.append(f"Code detected (confidence: {analysis['code_confidence']:.2f})")

        if analysis["has_complexity"]:
            reasons.append(f"Complex analysis required (confidence: {analysis['complexity_confidence']:.2f})")

        if analysis["is_simple"]:
            reasons.append(f"Simple query detected (length: {analysis['total_length']} chars)")

        if analysis["total_length"] > 1000:
            reasons.append("Long prompt requires capable model")

        if analysis["message_count"] > 5:
            reasons.append("Multi-turn conversation")

        task_reason = f"Task type: {analysis['task_type']}"
        if reasons:
            task_reason += f" ({', '.join(reasons)})"

        model_reason = f"Selected {provider}/{model} for optimal {analysis['task_type']} performance"

        return f"{task_reason}. {model_reason}"

    def record_latency(self, model_key: str, ms: int) -> None:
        """
        Record a latency measurement for a model.

        Args:
            model_key: Model identifier (e.g., "openai:gpt-4o")
            ms: Latency measurement in milliseconds
        """
        self.latency_priors.update(model_key, ms)

    def get_routing_stats(self) -> dict[str, Any]:
        """Get statistics about routing decisions (for monitoring)."""
        # In a real implementation, this would query the database
        # for routing statistics
        return {
            "total_requests": 0,
            "routing_by_task_type": {"code": 0, "complex": 0, "simple": 0, "general": 0},
            "routing_by_provider": {"openai": 0, "anthropic": 0, "mistral": 0},
        }


# Global router instance - will be initialized with provider registry
router = None
