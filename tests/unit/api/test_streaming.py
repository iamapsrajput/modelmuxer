#!/usr/bin/env python3
# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""
Test script to verify ModelMuxer streaming functionality.
This will test the live response streaming that you're looking for.
"""

import json
import time

import requests

BASE_URL = "http://localhost:8000"
API_KEY = "sk-test-claude-dev"


def test_streaming_request(content):
    """Test a streaming request to see the live writing effect."""
    print("🚀 Testing STREAMING Response")
    print(f"Query: {content[:100]}...")
    print("=" * 60)

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }

    payload = {
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 150,
        "stream": True,  # Enable streaming!
        "temperature": 0.7,
    }

    start_time = time.time()

    try:
        with requests.post(
            f"{BASE_URL}/v1/chat/completions",
            headers=headers,
            json=payload,
            stream=True,  # Important for streaming
            timeout=30,
        ) as response:
            print(f"Status Code: {response.status_code}")
            print(f"Content-Type: {response.headers.get('content-type', 'Unknown')}")

            if response.status_code != 200:
                print(f"❌ FAILED: {response.text}")
                return False

            print("\n🤖 AI Response (streaming live):")
            print("-" * 50)

            # Process the streaming response
            full_response = ""
            chunk_count = 0

            for line in response.iter_lines(decode_unicode=True):
                if line:
                    # Server-Sent Events format: "data: {...}"
                    if line.startswith("data: "):
                        data_str = line[6:]  # Remove "data: " prefix

                        if data_str.strip() == "[DONE]":
                            print("\n\n✅ Stream completed!")
                            break

                        try:
                            chunk_data = json.loads(data_str)

                            # Extract content from the chunk
                            if "choices" in chunk_data and chunk_data["choices"]:
                                delta = chunk_data["choices"][0].get("delta", {})
                                content_chunk = delta.get("content", "")

                                if content_chunk:
                                    print(
                                        content_chunk, end="", flush=True
                                    )  # Live streaming effect!
                                    full_response += content_chunk
                                    chunk_count += 1

                        except json.JSONDecodeError:
                            print(f"\n⚠️ Failed to parse chunk: {data_str}")

            end_time = time.time()
            response_time = end_time - start_time

            print("\n" + "-" * 50)
            print(f"⏱️  Total Response Time: {response_time:.2f}s")
            print(f"📦 Chunks Received: {chunk_count}")
            print(f"📝 Total Length: {len(full_response)} characters")

            if chunk_count > 0:
                print("✅ STREAMING SUCCESS - Response was delivered live!")
                return True
            else:
                print("❌ STREAMING FAILED - No content chunks received")
                return False

    except requests.exceptions.RequestException as e:
        print(f"❌ REQUEST FAILED: {e}")
        return False
    except Exception as e:
        print(f"❌ UNEXPECTED ERROR: {e}")
        return False


def test_non_streaming_for_comparison(content):
    """Test non-streaming for comparison."""
    print("\n🐌 Testing NON-STREAMING Response (for comparison)")
    print(f"Query: {content[:100]}...")
    print("=" * 60)

    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    payload = {
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 150,
        "stream": False,  # No streaming
        "temperature": 0.7,
    }

    start_time = time.time()

    try:
        response = requests.post(
            f"{BASE_URL}/v1/chat/completions", headers=headers, json=payload, timeout=30
        )

        end_time = time.time()
        response_time = end_time - start_time

        if response.status_code == 200:
            data = response.json()
            content = data["choices"][0]["message"]["content"]

            print("🤖 AI Response (all at once):")
            print("-" * 50)
            print(content)
            print("-" * 50)
            print(f"⏱️  Total Response Time: {response_time:.2f}s")
            print(f"📝 Total Length: {len(content)} characters")
            print("✅ NON-STREAMING SUCCESS - Response delivered all at once")
            return True
        else:
            print(f"❌ FAILED: {response.text}")
            return False

    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


def main():
    print("🧪 ModelMuxer Streaming vs Non-Streaming Test")
    print(f"Base URL: {BASE_URL}")
    print(f"API Key: {API_KEY}")

    # Test query - something that will generate a decent response
    test_query = "Write a Python function to implement a binary search algorithm. Include comments explaining each step."

    # Test streaming first
    streaming_success = test_streaming_request(test_query)

    print("\n" + "=" * 80 + "\n")

    # Test non-streaming for comparison
    non_streaming_success = test_non_streaming_for_comparison(test_query)

    # Summary
    print(f"\n{'=' * 60}")
    print("TEST SUMMARY")
    print(f"{'=' * 60}")
    print(f"Streaming:     {'✅ PASS' if streaming_success else '❌ FAIL'}")
    print(f"Non-Streaming: {'✅ PASS' if non_streaming_success else '❌ FAIL'}")

    if streaming_success:
        print("\n🎉 Streaming is working! You should see the live response effect.")
        print(
            "💡 To enable streaming in Claude Dev, make sure it sends 'stream': true in requests."
        )
    else:
        print("\n⚠️ Streaming test failed. Let's debug the issue.")

    if streaming_success and non_streaming_success:
        print("\n📋 RECOMMENDATION:")
        print("   - For better user experience, always use streaming")
        print("   - Configure Claude Dev to request streaming responses")
        print("   - Streaming provides the 'live writing' effect you want")


if __name__ == "__main__":
    main()
