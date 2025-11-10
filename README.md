# Voice Assistant Proxy

Python-based proxy server for the [voice-assistant-device](https://github.com/pcomans/voice-assistant-device) firmware, bridging ESP32-S3 hardware with OpenAI's Realtime API for natural voice conversations.

## What is this?

This proxy server acts as a secure intermediary between ESP32-based voice assistant hardware and OpenAI's Realtime API. It handles the complex WebSocket protocol, maintains persistent connections, and streams audio responses back to devices in real-time.

**Key Benefits:**
- **Security**: Keeps OpenAI API keys off embedded devices
- **Simplicity**: Devices use simple HTTP/PCM instead of complex WebSocket/JSON protocols
- **Flexibility**: Easy to test, debug, and modify without reflashing hardware
- **Control**: Centralized authentication, rate limiting, and monitoring

## Hardware Required

This proxy is designed to work with the [voice-assistant-device](https://github.com/pcomans/voice-assistant-device) ESP32-S3 firmware.

## Prerequisites

### Python 3.13+

This project requires Python 3.13 or higher. Check your Python version:

```bash
python3 --version
```

Install Python 3.13 if needed:
- **macOS**: `brew install python@3.13`
- **Linux**: Use your package manager or [pyenv](https://github.com/pyenv/pyenv)
- **Windows**: Download from [python.org](https://www.python.org/downloads/)

### Poetry (Python Package Manager)

Install Poetry for dependency management:

**macOS:**
```bash
brew install poetry
```

**Linux/Windows:**
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Or see [Poetry installation guide](https://python-poetry.org/docs/#installation) for other methods.

### OpenAI API Key

You'll need an OpenAI API key with access to the Realtime API. Get one at [platform.openai.com](https://platform.openai.com/api-keys).

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/pcomans/voice-assistant-proxy.git
cd voice-assistant-proxy
```

### 2. Install dependencies

```bash
poetry install
```

This creates a virtual environment and installs all required packages.

## Configuration

### 1. Create environment file

Copy the example environment file:

```bash
cp .env.example .env
```

### 2. Configure secrets

Edit `.env` and set your credentials:

```bash
# OpenAI API key (required)
OPENAI_API_KEY=sk-your-openai-api-key-here

# Shared secret for device authentication (required)
# Must match PROXY_DEFAULT_TOKEN in the device firmware
ASSISTANT_SHARED_SECRET=498b1b65-26a3-49e8-a55e-46a0b47365e2
```

**Default Token:**
The device firmware comes with a hardcoded token: `498b1b65-26a3-49e8-a55e-46a0b47365e2`

**For first-time setup:** Use this default token value (shown above)

**For enhanced security:**
- Generate a new token (UUID or any secure random string)
- Update `ASSISTANT_SHARED_SECRET` in the proxy's `.env` file
- Update `PROXY_DEFAULT_TOKEN` in `main/proxy_client.c` on the device
- Rebuild and reflash the device firmware

**Security Notes:**
- Never commit `.env` to version control (it's gitignored)
- Consider using a unique token for production deployments
- The device sends this token in the `X-Assistant-Token` HTTP header

### 3. Find your local IP address

Devices need to connect to your proxy via your local network IP:

**macOS/Linux:**
```bash
# Get local IP
ifconfig | grep "inet " | grep -v 127.0.0.1
```

**Windows:**
```bash
ipconfig
```

Look for your local IP (usually `192.168.x.x` or `10.x.x.x`). You'll configure this IP in the device firmware.

## Running the Proxy

### Start the server

```bash
poetry run uvicorn smart_assistant_proxy.app:app --host 0.0.0.0 --port 8000
```

**Options:**
- `--host 0.0.0.0`: Listen on all network interfaces (allows device connections)
- `--port 8000`: Port to listen on (default: 8000)
- `--reload`: Auto-reload on code changes (useful during development)

**Example with auto-reload:**
```bash
poetry run uvicorn smart_assistant_proxy.app:app --host 0.0.0.0 --port 8000 --reload
```

### Verify it's running

Test the health endpoint:

```bash
curl http://localhost:8000/healthz
```

Should return:
```json
{"status": "ok"}
```

From another device on your network:
```bash
curl http://YOUR_LOCAL_IP:8000/healthz
```

### Run in background

To run the proxy in the background and redirect logs to a file:

```bash
poetry run uvicorn smart_assistant_proxy.app:app --host 0.0.0.0 --port 8000 > /tmp/proxy.log 2>&1 &
```

View logs:
```bash
tail -f /tmp/proxy.log
```

## View Logs

Logs are output to stdout by default. You can:

**View real-time logs:**
```bash
poetry run uvicorn smart_assistant_proxy.app:app --host 0.0.0.0 --port 8000
```

**Save logs to file:**
```bash
poetry run uvicorn smart_assistant_proxy.app:app --host 0.0.0.0 --port 8000 2>&1 | tee proxy.log
```

**Set log level:**
```bash
# Set to DEBUG for verbose logging
LOG_LEVEL=DEBUG poetry run uvicorn smart_assistant_proxy.app:app --host 0.0.0.0 --port 8000
```

**Filter logs:**
```bash
# Show only errors
poetry run uvicorn smart_assistant_proxy.app:app --host 0.0.0.0 --port 8000 2>&1 | grep ERROR
```

## API Reference

### `POST /v1/audio`

Accepts streaming audio chunks from devices, forwards to OpenAI Realtime API, and streams audio responses back.

**Headers:**
- `X-Assistant-Token`: Shared secret for authentication (required)

**Request Body:**
```json
{
  "session_id": "unique-session-id",
  "chunk_index": 0,
  "pcm_base64": "base64-encoded-pcm-audio",
  "is_final": false
}
```

**Response:**
- Non-final chunks: `{"status": "partial"}`
- Final chunk: Binary PCM audio stream (Content-Type: `application/octet-stream`)

**Audio Format:**
- Input: 16kHz, 16-bit PCM, mono (from device)
- Output: 24kHz, 16-bit PCM, mono (to device)

### `GET /healthz`

Health check endpoint for monitoring.

**Response:**
```json
{"status": "ok"}
```

## Project Structure

```
smart_assistant_proxy/
├── app.py          # FastAPI application and endpoints
├── realtime.py     # OpenAI Realtime API WebSocket client
├── config.py       # Configuration and environment variables
└── __init__.py     # Package initialization

tests/              # Test suite
├── test_app.py     # API endpoint tests
└── test_realtime.py # Realtime client tests

.env                # Environment configuration (gitignored)
.env.example        # Example environment template
pyproject.toml      # Python dependencies and project metadata
```

## Development

### Run tests

```bash
poetry run pytest
```

**With coverage:**
```bash
poetry run pytest --cov=smart_assistant_proxy --cov-report=html
```

View coverage report:
```bash
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

### Code style

The project uses standard Python conventions. Future additions may include:
- Black for formatting
- Ruff for linting
- mypy for type checking

## Troubleshooting

### Connection refused
- Verify proxy is running: `curl http://localhost:8000/healthz`
- Check firewall settings allow port 8000
- Ensure using `--host 0.0.0.0` (not `127.0.0.1`)

### Authentication failed (401)
- Verify `ASSISTANT_SHARED_SECRET` in `.env` matches `PROXY_DEFAULT_TOKEN` in device firmware
- Default token should be: `498b1b65-26a3-49e8-a55e-46a0b47365e2`
- Check device sends `X-Assistant-Token` header in requests
- Review proxy logs for "invalid token" messages

### OpenAI API errors
- Verify `OPENAI_API_KEY` is valid
- Check OpenAI account has Realtime API access
- Review proxy logs for detailed error messages

### Device can't connect to proxy
- Verify device and computer are on same network
- Check local IP address: `ifconfig` (macOS/Linux) or `ipconfig` (Windows)
- Test from device network: `curl http://YOUR_LOCAL_IP:8000/healthz`
- Disable VPN if active (can block local connections)

### Audio quality issues
- Check network latency between device and proxy
- Verify sufficient bandwidth for streaming audio
- Review proxy logs for buffer underruns or dropped frames

## Production Deployment

For production use, consider:

- **Process manager**: Use systemd, supervisor, or PM2 to keep proxy running
- **Reverse proxy**: Put behind nginx or caddy for HTTPS/SSL
- **Monitoring**: Add Prometheus metrics or logging to monitoring service
- **Rate limiting**: Implement per-device rate limits to control costs
- **Multiple instances**: Run multiple proxies behind a load balancer

**Example systemd service:**
```ini
[Unit]
Description=Voice Assistant Proxy
After=network.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/voice-assistant-proxy
Environment="PATH=/path/to/poetry/bin:$PATH"
ExecStart=/usr/local/bin/poetry run uvicorn smart_assistant_proxy.app:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/)
- WebSocket support via [websockets](https://websockets.readthedocs.io/)
- Audio processing with [NumPy](https://numpy.org/)
- OpenAI Realtime API integration
