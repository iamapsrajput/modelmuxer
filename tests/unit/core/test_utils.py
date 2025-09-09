# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""Unit tests for core utilities."""

import hashlib
import time
from unittest.mock import MagicMock, patch

import pytest

from app.core.utils import (calculate_similarity, detect_programming_language,
                            estimate_tokens, extract_code_blocks, format_cost,
                            generate_request_id, hash_prompt, parse_model_name,
                            round_cost, sanitize_input, truncate_text)
from app.models import ChatMessage


class TestGenerateRequestId:
    """Test request ID generation."""

    def test_request_id_format(self):
        """Test request ID has correct format."""
        request_id = generate_request_id()
        assert request_id.startswith("req_")
        parts = request_id.split("_")
        assert len(parts) == 3
        assert parts[1].isdigit()  # Timestamp
        assert len(parts[2]) == 8  # Random hex

    def test_request_id_uniqueness(self):
        """Test request IDs are unique."""
        ids = [generate_request_id() for _ in range(100)]
        assert len(set(ids)) == 100


class TestHashPrompt:
    """Test prompt hashing."""

    def test_hash_single_message(self):
        """Test hashing a single message."""
        messages = [ChatMessage(role="user", content="Hello world")]
        hash_result = hash_prompt(messages)

        # Verify it's a SHA256 hash (64 hex chars)
        assert len(hash_result) == 64
        assert all(c in "0123456789abcdef" for c in hash_result)

    def test_hash_multiple_messages(self):
        """Test hashing multiple messages."""
        messages = [
            ChatMessage(role="user", content="Hello"),
            ChatMessage(role="assistant", content="Hi there!"),
            ChatMessage(role="user", content="How are you?"),
        ]
        hash_result = hash_prompt(messages)

        assert len(hash_result) == 64

    def test_hash_consistency(self):
        """Test hash is consistent for same input."""
        messages = [ChatMessage(role="user", content="Test message")]
        hash1 = hash_prompt(messages)
        hash2 = hash_prompt(messages)

        assert hash1 == hash2

    def test_hash_different_messages(self):
        """Test different messages produce different hashes."""
        messages1 = [ChatMessage(role="user", content="Message 1")]
        messages2 = [ChatMessage(role="user", content="Message 2")]

        hash1 = hash_prompt(messages1)
        hash2 = hash_prompt(messages2)

        assert hash1 != hash2

    def test_unsupported_algorithm(self):
        """Test unsupported hash algorithm raises error."""
        messages = [ChatMessage(role="user", content="Test")]

        with pytest.raises(ValueError) as exc_info:
            hash_prompt(messages, algorithm="md5")

        assert "Only 'sha256' is supported" in str(exc_info.value)


class TestEstimateTokens:
    """Test token estimation."""

    @patch("tiktoken.encoding_for_model")
    def test_estimate_tokens_gpt4(self, mock_encoding):
        """Test token estimation for GPT-4."""
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = [1, 2, 3, 4, 5]
        mock_encoding.return_value = mock_encoder

        count = estimate_tokens("test text", model="gpt-4")
        assert count == 5
        mock_encoding.assert_called_with("gpt-4")

    @patch("tiktoken.encoding_for_model")
    def test_estimate_tokens_gpt35(self, mock_encoding):
        """Test token estimation for GPT-3.5."""
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = [1, 2, 3]
        mock_encoding.return_value = mock_encoder

        count = estimate_tokens("test", model="gpt-3.5-turbo")
        assert count == 3
        mock_encoding.assert_called_with("gpt-3.5-turbo")

    @patch("tiktoken.get_encoding")
    def test_estimate_tokens_other_model(self, mock_get_encoding):
        """Test token estimation for other models."""
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = [1, 2, 3, 4]
        mock_get_encoding.return_value = mock_encoder

        count = estimate_tokens("test text", model="claude-3")
        assert count == 4
        mock_get_encoding.assert_called_with("cl100k_base")

    @patch("tiktoken.encoding_for_model", side_effect=Exception("Encoding error"))
    def test_estimate_tokens_fallback(self, mock_encoding):
        """Test fallback token estimation."""
        # Fallback uses len(text) // 4
        count = estimate_tokens("This is a test message")
        assert count == 22 // 4  # 5


class TestFormatCost:
    """Test cost formatting."""

    def test_format_very_small_cost(self):
        """Test formatting very small costs."""
        assert format_cost(0.0000001) == "<$0.000001 USD"
        assert format_cost(0.0000005) == "<$0.000001 USD"

    def test_format_small_cost(self):
        """Test formatting small costs."""
        assert format_cost(0.001234) == "$0.001234 USD"
        assert format_cost(0.00999) == "$0.009990 USD"

    def test_format_regular_cost(self):
        """Test formatting regular costs."""
        assert format_cost(0.1234) == "$0.1234 USD"
        assert format_cost(1.2345) == "$1.2345 USD"
        assert format_cost(10.99) == "$10.9900 USD"

    def test_format_cost_with_currency(self):
        """Test formatting with different currency."""
        assert format_cost(0.01, "EUR") == "$0.0100 EUR"


class TestSanitizeInput:
    """Test input sanitization."""

    def test_sanitize_normal_text(self):
        """Test sanitizing normal text."""
        text = "Hello world! This is a test."
        assert sanitize_input(text) == text

    def test_sanitize_control_characters(self):
        """Test removing control characters."""
        text = "Hello\x00World\x07Test\x1f"
        assert sanitize_input(text) == "HelloWorldTest"

    def test_sanitize_preserve_newlines(self):
        """Test preserving newlines and tabs."""
        text = "Line 1\nLine 2\tTabbed"
        assert sanitize_input(text) == text

    def test_sanitize_normalize_line_endings(self):
        """Test normalizing line endings."""
        text = "Line 1\r\nLine 2\rLine 3"
        assert sanitize_input(text) == "Line 1\nLine 2\nLine 3"

    def test_sanitize_excessive_newlines(self):
        """Test reducing excessive newlines."""
        text = "Line 1\n\n\n\n\nLine 2"
        assert sanitize_input(text) == "Line 1\n\n\nLine 2"

    def test_sanitize_max_length(self):
        """Test max length validation."""
        text = "a" * 100001
        with pytest.raises(ValueError) as exc_info:
            sanitize_input(text)
        assert "Input too long" in str(exc_info.value)

    def test_sanitize_strip_whitespace(self):
        """Test stripping leading/trailing whitespace."""
        text = "  Hello World  \n"
        assert sanitize_input(text) == "Hello World"


class TestCalculateSimilarity:
    """Test text similarity calculation."""

    def test_identical_texts(self):
        """Test similarity of identical texts."""
        similarity = calculate_similarity("hello world", "hello world")
        assert similarity == 1.0

    def test_completely_different_texts(self):
        """Test similarity of completely different texts."""
        similarity = calculate_similarity("hello world", "foo bar")
        assert similarity == 0.0

    def test_partial_similarity(self):
        """Test partial text similarity."""
        similarity = calculate_similarity("hello world test", "hello world")
        # Jaccard similarity: 2/3 = 0.666...
        assert 0.66 < similarity < 0.67

    def test_case_insensitive(self):
        """Test case-insensitive comparison."""
        similarity = calculate_similarity("Hello World", "hello world")
        assert similarity == 1.0

    def test_empty_texts(self):
        """Test empty text similarity."""
        assert calculate_similarity("", "") == 0.0
        assert calculate_similarity("hello", "") == 0.0
        assert calculate_similarity("", "world") == 0.0


class TestExtractCodeBlocks:
    """Test code block extraction."""

    def test_extract_single_code_block(self):
        """Test extracting a single code block."""
        text = "Here is code:\n```python\nprint('hello')\n```"
        blocks = extract_code_blocks(text)

        assert len(blocks) == 1
        assert blocks[0]["language"] == "python"
        assert blocks[0]["content"] == "print('hello')"

    def test_extract_multiple_code_blocks(self):
        """Test extracting multiple code blocks."""
        text = """
```python
def hello():
    print('hello')
```
Some text
```javascript
console.log('world');
```
"""
        blocks = extract_code_blocks(text)

        assert len(blocks) == 2
        assert blocks[0]["language"] == "python"
        assert "def hello" in blocks[0]["content"]
        assert blocks[1]["language"] == "javascript"
        assert "console.log" in blocks[1]["content"]

    def test_extract_code_block_no_language(self):
        """Test extracting code block without language."""
        text = "Code:\n```\nsome code\n```"
        blocks = extract_code_blocks(text)

        assert len(blocks) == 1
        assert blocks[0]["language"] == "unknown"
        assert blocks[0]["content"] == "some code"

    def test_no_code_blocks(self):
        """Test when no code blocks are present."""
        text = "This is just plain text"
        blocks = extract_code_blocks(text)

        assert blocks == []


class TestDetectProgrammingLanguage:
    """Test programming language detection."""

    def test_detect_python(self):
        """Test detecting Python code."""
        code = """
def main():
    import sys
    print("Hello world")
    if __name__ == "__main__":
        main()
"""
        assert detect_programming_language(code) == "python"

    def test_detect_javascript(self):
        """Test detecting JavaScript code."""
        code = """
function greet(name) {
    const message = `Hello ${name}`;
    console.log(message);
}
const result = () => { return 42; };
"""
        assert detect_programming_language(code) == "javascript"

    def test_detect_java(self):
        """Test detecting Java code."""
        code = """
public class HelloWorld {
    public static void main(String[] args) {
        System.out.println("Hello World");
    }
}
"""
        assert detect_programming_language(code) == "java"

    def test_detect_sql(self):
        """Test detecting SQL."""
        code = """
SELECT * FROM users
WHERE age > 18;
UPDATE users SET active = true;
"""
        assert detect_programming_language(code) == "sql"

    def test_detect_unknown_language(self):
        """Test when language cannot be detected."""
        code = "This is just plain text without any code"
        assert detect_programming_language(code) is None

    def test_detect_insufficient_patterns(self):
        """Test when not enough patterns match."""
        code = "def something"  # Only one Python pattern
        assert detect_programming_language(code) is None


class TestTruncateText:
    """Test text truncation."""

    def test_truncate_short_text(self):
        """Test text shorter than max length is not truncated."""
        text = "Hello world"
        assert truncate_text(text, max_length=20) == text

    def test_truncate_exact_length(self):
        """Test text exactly at max length."""
        text = "Hello"
        assert truncate_text(text, max_length=5) == text

    def test_truncate_long_text(self):
        """Test truncating long text."""
        text = "This is a very long text that needs to be truncated"
        result = truncate_text(text, max_length=20)
        assert result == "This is a very lo..."
        assert len(result) == 20

    def test_truncate_custom_suffix(self):
        """Test truncation with custom suffix."""
        text = "Hello world!"
        result = truncate_text(text, max_length=10, suffix=" [...]")
        assert result == "Hell [...]"

    def test_truncate_default_length(self):
        """Test default max length (1000)."""
        text = "a" * 2000
        result = truncate_text(text)
        assert len(result) == 1000
        assert result.endswith("...")


class TestParseModelName:
    """Test model name parsing."""

    def test_parse_with_provider(self):
        """Test parsing model name with provider."""
        result = parse_model_name("openai/gpt-4")
        assert result == {"provider": "openai", "model": "gpt-4"}

    def test_parse_without_provider(self):
        """Test parsing model name without provider."""
        result = parse_model_name("gpt-4")
        assert result == {"provider": "unknown", "model": "gpt-4"}

    def test_parse_multiple_slashes(self):
        """Test parsing with multiple slashes."""
        result = parse_model_name("provider/model/version/variant")
        assert result == {"provider": "provider", "model": "model/version/variant"}

    def test_parse_empty_string(self):
        """Test parsing empty string."""
        result = parse_model_name("")
        assert result == {"provider": "unknown", "model": ""}


class TestRoundCost:
    """Test cost rounding."""

    def test_round_cost_default(self):
        """Test default rounding to 6 decimal places."""
        assert round_cost(0.1234567890) == 0.123457
        assert round_cost(0.1234564890) == 0.123456

    def test_round_cost_custom_places(self):
        """Test custom decimal places."""
        assert round_cost(0.123456, decimal_places=2) == 0.12
        assert round_cost(0.126456, decimal_places=2) == 0.13
        assert round_cost(0.999999, decimal_places=4) == 1.0

    def test_round_cost_half_up(self):
        """Test rounding uses ROUND_HALF_UP."""
        assert round_cost(0.1234565, decimal_places=6) == 0.123457
        assert round_cost(0.125, decimal_places=2) == 0.13

    def test_round_cost_zero(self):
        """Test rounding zero."""
        assert round_cost(0.0) == 0.0
        assert round_cost(0.0, decimal_places=2) == 0.0

    def test_round_cost_large_numbers(self):
        """Test rounding large costs."""
        assert round_cost(123.456789, decimal_places=3) == 123.457
        assert round_cost(9999.9999999, decimal_places=5) == 10000.0
