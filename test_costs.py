# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.

from app.core.costing import Estimator, LatencyPriors, load_price_table
from app.settings import settings

# Load price table
price_table = load_price_table(settings.pricing.price_table_path)
latency_priors = LatencyPriors()
estimator = Estimator(price_table, latency_priors, settings)

# Test with simple message
tokens_in, tokens_out = 10, 20  # Simple message like 'What is the capital of France?'

# Check costs for cheapest models
cheap_models = [
    "groq:llama-3.1-8b-instant",
    "groq:gemma-7b-it",
    "google:gemini-1.5-flash",
    "together:llama-2-7b-chat",
]

print("Model costs for 10 input tokens, 20 output tokens:")
for model in cheap_models:
    try:
        estimate = estimator.estimate(model, tokens_in, tokens_out)
        print(f"{model}: ${estimate.usd:.8f}")
    except Exception as e:
        print(f"{model}: ERROR - {e}")
