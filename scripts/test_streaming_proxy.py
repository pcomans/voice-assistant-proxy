#!/usr/bin/env python3
"""
Test script for HTTP streaming proxy (Option C)
Tests the proxy's ability to forward chunks immediately and stream responses back.
"""

import asyncio
import base64
import json
import wave
import time
from pathlib import Path
import httpx


async def test_streaming_proxy(
    audio_file: str = "test_data/test_audio_16khz.wav",
    proxy_url: str = "http://192.168.7.75:8000/v1/audio",
    token: str = None
):
    """Test proxy streaming with real recorded audio"""

    print("=" * 70)
    print("Testing HTTP Streaming Proxy (Option C)")
    print("=" * 70)

    # Load recorded audio
    audio_path = Path(audio_file)
    if not audio_path.exists():
        print(f"\n❌ Audio file not found: {audio_file}")
        print("\nAvailable audio files:")
        for f in Path(".").glob("*.wav"):
            print(f"  - {f}")
        return

    with wave.open(str(audio_path), 'rb') as wf:
        print(f"\n📼 Loaded audio: {audio_file}")
        print(f"   Sample rate: {wf.getframerate()} Hz")
        print(f"   Channels: {wf.getnchannels()}")
        print(f"   Duration: {wf.getnframes() / wf.getframerate():.2f}s")
        print(f"   Format: {wf.getsampwidth() * 8}-bit PCM")

        if wf.getframerate() != 16000:
            print(f"   ⚠️  Warning: Audio should be 16kHz, got {wf.getframerate()}Hz")

        if wf.getnchannels() != 1:
            print(f"   ⚠️  Warning: Audio should be mono, got {wf.getnchannels()} channels")

        pcm_data = wf.readframes(wf.getnframes())

    # Split into chunks (100ms at 16kHz = 1600 samples = 3200 bytes for 16-bit)
    chunk_size = 3200  # 100ms chunks
    chunks = [pcm_data[i:i+chunk_size] for i in range(0, len(pcm_data), chunk_size)]

    session_id = f"test-{int(time.time())}"

    print(f"\n🚀 Simulating device streaming:")
    print(f"   Session ID: {session_id}")
    print(f"   Total chunks: {len(chunks)} (100ms each)")
    print(f"   Chunk size: {chunk_size} bytes")
    print(f"   Proxy URL: {proxy_url}")
    print(f"\n{'─' * 70}\n")

    # Prepare token header
    headers = {}
    if token:
        headers["X-Assistant-Token"] = token

    # Send chunks
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            start_time = time.time()

            for i, chunk in enumerate(chunks):
                is_final = (i == len(chunks) - 1)

                payload = {
                    "session_id": session_id,
                    "chunk_index": i,
                    "pcm_base64": base64.b64encode(chunk).decode(),
                    "is_final": is_final
                }

                chunk_start = time.time()

                if not is_final:
                    # Non-final chunk - expect quick acknowledgment
                    try:
                        response = await client.post(
                            proxy_url,
                            json=payload,
                            headers=headers
                        )
                        result = response.json()

                        elapsed = (time.time() - chunk_start) * 1000
                        status = result.get('status', 'unknown')

                        # Simple progress indicator
                        bar_length = 40
                        progress = (i + 1) / len(chunks)
                        filled = int(bar_length * progress)
                        bar = '█' * filled + '░' * (bar_length - filled)

                        print(f"\r[{bar}] Chunk {i+1:2d}/{len(chunks)} ({status:8s}, {elapsed:4.0f}ms)", end='', flush=True)

                        if status != "partial":
                            print(f"\n   ⚠️  Expected 'partial', got '{status}'")

                        # Simulate device recording delay (100ms between chunks)
                        await asyncio.sleep(0.1)

                    except Exception as e:
                        print(f"\n❌ Error sending chunk {i}: {e}")
                        return

                else:
                    # Final chunk - expect streaming response
                    print(f"\n\n✓ All chunks sent! Waiting for OpenAI response...\n")

                    try:
                        first_chunk_time = None
                        response_chunks = []

                        async with client.stream(
                            "POST",
                            proxy_url,
                            json=payload,
                            headers=headers
                        ) as response:

                            if response.status_code != 200:
                                error_text = await response.aread()
                                print(f"❌ HTTP {response.status_code}: {error_text.decode()}")
                                return

                            async for line in response.aiter_lines():
                                if line.strip():
                                    if first_chunk_time is None:
                                        first_chunk_time = time.time()
                                        ttfb = (first_chunk_time - start_time) * 1000
                                        print(f"🎉 First response chunk received!")
                                        print(f"   Time to first byte: {ttfb:.0f}ms\n")

                                    try:
                                        chunk_data = json.loads(line)
                                        response_chunks.append(chunk_data)

                                        # Show streaming progress
                                        print(f"  ← Response chunk {len(response_chunks):2d}", end='\r', flush=True)

                                    except json.JSONDecodeError as e:
                                        print(f"\n   ⚠️  Invalid JSON: {line[:100]}")

                        print()  # New line after progress

                        total_time = (time.time() - start_time) * 1000

                        print(f"\n{'─' * 70}")
                        print(f"✅ Streaming complete!")
                        print(f"{'─' * 70}")
                        print(f"   Total time: {total_time:.0f}ms ({total_time/1000:.2f}s)")
                        print(f"   Upload time: {(first_chunk_time - start_time)*1000:.0f}ms")
                        print(f"   Response chunks: {len(response_chunks)}")

                        # Decode and save audio as WAV
                        if response_chunks:
                            # Decode each chunk's base64 separately, then join the bytes
                            pcm_bytes = b"".join(
                                base64.b64decode(c.get("audio_delta", ""))
                                for c in response_chunks
                                if c.get("audio_delta")
                            )

                            if pcm_bytes:
                                # Save as WAV file (24kHz, mono, 16-bit PCM)
                                output_file = "response.wav"
                                with wave.open(output_file, 'wb') as wf:
                                    wf.setnchannels(1)  # Mono
                                    wf.setsampwidth(2)  # 16-bit = 2 bytes
                                    wf.setframerate(24000)  # 24kHz
                                    wf.writeframes(pcm_bytes)

                                duration = len(pcm_bytes) / (24000 * 2)  # bytes / (sample_rate * bytes_per_sample)
                                print(f"   Response size: {len(pcm_bytes):,} bytes")
                                print(f"   Duration: {duration:.2f}s")
                                print(f"   Format: 24kHz mono PCM16")
                                print(f"   Saved to: {output_file}")
                                print(f"\n🔊 Play response:")
                                print(f"   ffplay {output_file}")
                            else:
                                print(f"   ⚠️  No audio data in response")
                        else:
                            print(f"   ⚠️  No response chunks received")

                    except httpx.ReadTimeout:
                        print(f"❌ Timeout waiting for response")
                    except Exception as e:
                        print(f"❌ Error receiving response: {e}")
                        import traceback
                        traceback.print_exc()

    except httpx.ConnectError:
        print(f"\n❌ Cannot connect to proxy at {proxy_url}")
        print(f"   Is the proxy running?")
        print(f"   Start proxy: cd smart_assistant_proxy && python -m uvicorn smart_assistant_proxy.app:app --reload")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import sys

    # Parse command line arguments
    audio_file = "test_data/test_audio_16khz.wav"
    proxy_url = "http://192.168.7.75:8000/v1/audio"
    token = None

    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
    if len(sys.argv) > 2:
        proxy_url = sys.argv[2]
    if len(sys.argv) > 3:
        token = sys.argv[3]

    print("\nUsage: poetry run python scripts/test_streaming_proxy.py [audio_file] [proxy_url] [token]")
    print(f"Using: {audio_file}\n")

    asyncio.run(test_streaming_proxy(audio_file, proxy_url, token))
