# Frequently Asked Questions

## Installation and Setup

### Q: What Python version is required?

A: ModelMuxer requires Python 3.11 or higher.

### Q: Which LLM providers are supported?

A: Currently supported providers include OpenAI, Anthropic, and Mistral.

### Q: How do I get API keys for providers?

A: Visit each provider's website to sign up and generate API keys:

- **OpenAI**: Create an account at [platform.openai.com](https://platform.openai.com/) and navigate to API Keys
- **Anthropic**: Sign up at [console.anthropic.com](https://docs.anthropic.com/en/api/getting-started) and access your API keys in account settings
- **Mistral**: Register at [console.mistral.ai](https://docs.mistral.ai/) and generate API keys in your dashboard

## Configuration

### Q: How do I configure routing thresholds?

A: Set routing thresholds in your environment variables. See the configuration guide for details on CODE_DETECTION_THRESHOLD and COMPLEXITY_THRESHOLD.

### Q: What routing strategy is used?

A: ModelMuxer currently uses heuristic routing that analyzes prompts to classify them as code, complex, simple, or general tasks, then selects the optimal model accordingly.

## Usage

### Q: How does intelligent routing work?

A: ModelMuxer analyzes your prompts and automatically selects the most appropriate model based on factors like cost, quality requirements, and task complexity.

### Q: Is streaming supported?

A: Yes, ModelMuxer supports streaming responses from all supported providers.

## Troubleshooting

For technical issues, see the [Troubleshooting Guide](troubleshooting.md).

## Getting Help

If you have questions not covered here:

- Check the [documentation](../README.md)
- Open an issue on [GitHub Issues](https://github.com/iamapsrajput/ModelMuxer/issues)
- Contact the maintainers via email
