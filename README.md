# Smart Assistant Local Proxy

LAN service that bridges the ESP32-S3 smart assistant firmware and OpenAI’s Realtime API.

## Why a Local Proxy?
- **Keeps secrets off the device**: the ESP32 never stores your `OPENAI_API_KEY`; it only talks to the proxy using a short shared secret.
- **Simplifies the protocol**: the board sends raw PCM over simple HTTPS; the proxy handles the more complex Realtime streaming workflow and JSON framing expected by OpenAI.
- **Eases iteration**: you can inspect requests, inject mock responses, or run without internet while still exercising the firmware pipeline.
- **Adds a security layer**: the proxy can enforce device auth, rate limits, or other policies before forwarding traffic to OpenAI.

## Quick Start
- Install dependencies:
  ```bash
  poetry install
  ```
- Copy the example environment file and fill in your secrets:
  ```bash
  cp .env.example .env
  ```
- Launch the API (reload optional while iterating):
  ```bash
  poetry run uvicorn smart_assistant_proxy.app:app --host 0.0.0.0 --port 8000 --reload
  ```

## Environment
- `OPENAI_API_KEY` – API key used to contact OpenAI.
- `ASSISTANT_SHARED_SECRET` – token that devices must present in the `X-Assistant-Token` header.

## Endpoints
- `POST /v1/audio` – Accepts streamed PCM chunks from the device, opens a Realtime session with OpenAI, returns synthesized audio and transcript IDs.
- `GET /healthz` – Readiness probe for local monitoring.
