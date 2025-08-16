#!/usr/bin/env python3
"""
Custom Claude CLI wrapper for ModelMuxer
Provides a Claude-like command-line interface that routes through ModelMuxer
"""

import os
import sys
import json
import requests
import argparse
from typing import Optional, Dict, Any
from urllib.parse import urlparse

try:
    import readline  # For better input handling on Unix/Linux
except ImportError:
    pass  # readline not available on Windows, but not required


class ClaudeCLI:
    def __init__(self, base_url: str = "http://localhost:8000", api_key: str = None):
        # Validate and sanitize base URL to mitigate SSRF
        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("Invalid --url provided. Must be http(s)://host[:port]")
        # Allow only localhost by default to avoid SSRF. Override with env var for trusted CI/dev.
        host = parsed.hostname or ""
        if (
            host not in {"localhost", "127.0.0.1"}
            and os.getenv("ALLOW_EXTERNAL_CLI_URLS", "false").lower() != "true"
        ):
            raise ValueError(
                "Refusing non-localhost URL. Set ALLOW_EXTERNAL_CLI_URLS=true to override."
            )
        self.base_url = f"{parsed.scheme}://{parsed.netloc}"
        # Prefer central settings if available, otherwise fall back to env
        try:
            from app.settings import settings as app_settings

            default_key = app_settings.api.api_keys[0] if app_settings.api.api_keys else None
        except Exception:
            default_key = None
        self.api_key = (
            api_key or default_key or os.getenv("MODELMUXER_API_KEY", "your-api-key-here")
        )
        self.session = requests.Session()
        self.session.headers.update(
            {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        )

    def chat(self, message: str, model: Optional[str] = None, stream: bool = True) -> None:
        """Send a chat message to ModelMuxer"""

        # Prepare the request payload in OpenAI format (ModelMuxer handles both)
        payload = {
            "model": model or "claude-3-5-sonnet-20241022",  # Default to Claude
            "messages": [{"role": "user", "content": message}],
            "stream": stream,
            "max_tokens": 4000,
            "temperature": 0.7,
        }

        try:
            if stream:
                self._stream_response(payload)
            else:
                self._non_stream_response(payload)

        except requests.exceptions.RequestException as e:
            print(f"Error connecting to ModelMuxer: {e}")
            sys.exit(1)
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
            sys.exit(0)

    def _stream_response(self, payload: Dict[Any, Any]) -> None:
        """Handle streaming response"""
        try:
            response = self.session.post(
                f"{self.base_url}/v1/chat/completions", json=payload, stream=True, timeout=60
            )
            response.raise_for_status()

            print("\nClaude (via ModelMuxer):")
            print("-" * 40)

            for line in response.iter_lines():
                if line:
                    line = line.decode("utf-8")
                    if line.startswith("data: "):
                        data = line[6:]  # Remove 'data: ' prefix
                        if data.strip() == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            if "choices" in chunk and len(chunk["choices"]) > 0:
                                delta = chunk["choices"][0].get("delta", {})
                                if "content" in delta:
                                    content = delta["content"]
                                    print(content, end="", flush=True)
                        except json.JSONDecodeError:
                            continue

            print("\n" + "-" * 40)

        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            if hasattr(e, "response") and e.response is not None:
                print(f"Response: {e.response.text}")

    def _non_stream_response(self, payload: Dict[Any, Any]) -> None:
        """Handle non-streaming response"""
        payload["stream"] = False
        response = self.session.post(
            f"{self.base_url}/v1/chat/completions", json=payload, timeout=60
        )
        response.raise_for_status()

        data = response.json()
        if "choices" in data and len(data["choices"]) > 0:
            content = data["choices"][0]["message"]["content"]
            print("\nClaude (via ModelMuxer):")
            print("-" * 40)
            print(content)
            print("-" * 40)
        else:
            print("No response received")

    def interactive_mode(self):
        """Run in interactive mode"""
        print("Claude CLI (powered by ModelMuxer)")
        print("Type 'exit' or 'quit' to end the session")
        print("Type '/help' for commands")
        print("-" * 50)

        while True:
            try:
                user_input = input("\nYou: ").strip()

                if user_input.lower() in ["exit", "quit", "q"]:
                    print("Goodbye!")
                    break

                if user_input == "/help":
                    self._show_help()
                    continue

                if user_input == "/models":
                    self._list_models()
                    continue

                if user_input == "/status":
                    self._show_status()
                    continue

                if not user_input:
                    continue

                self.chat(user_input)

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except EOFError:
                print("\nGoodbye!")
                break

    def _show_help(self):
        """Show help information"""
        help_text = """
Available commands:
  /help     - Show this help message
  /models   - List available models
  /status   - Show ModelMuxer status
  /exit     - Exit the CLI

Just type your message and press Enter to chat with Claude via ModelMuxer.
ModelMuxer will intelligently route your request to the best available model.
        """
        print(help_text)

    def _list_models(self):
        """List available models from ModelMuxer"""
        try:
            response = self.session.get(f"{self.base_url}/v1/models")
            if response.status_code == 200:
                models = response.json()
                print("\nAvailable models:")
                print("-" * 30)
                for model in models.get("data", []):
                    print(f"  {model.get('id', 'Unknown')}")
            else:
                print(f"Could not fetch models: {response.status_code}")
        except Exception as e:
            print(f"Error fetching models: {e}")

    def _show_status(self):
        """Show ModelMuxer health status"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            if response.status_code == 200:
                status = response.json()
                print(f"\nModelMuxer Status: {status.get('status', 'Unknown')}")
                if "mode" in status:
                    print(f"Mode: {status['mode']}")
                if "providers" in status:
                    print(f"Active providers: {len(status['providers'])}")
            else:
                print(f"ModelMuxer health check failed: {response.status_code}")
        except Exception as e:
            print(f"Error checking status: {e}")


def main():
    parser = argparse.ArgumentParser(description="Claude CLI via ModelMuxer")
    parser.add_argument("--message", "-m", help="Single message to send")
    parser.add_argument("--model", help="Specific model to use")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming")
    parser.add_argument("--url", default="http://localhost:8000", help="ModelMuxer URL")
    parser.add_argument("--api-key", help="API key for ModelMuxer")

    args = parser.parse_args()

    cli = ClaudeCLI(base_url=args.url, api_key=args.api_key)

    if args.message:
        # Single message mode
        cli.chat(args.message, model=args.model, stream=not args.no_stream)
    else:
        # Interactive mode
        cli.interactive_mode()


if __name__ == "__main__":
    main()
