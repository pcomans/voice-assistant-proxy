# Test Data

This directory contains audio files used for testing the streaming proxy.

## Files

- `Zebra.m4a` - Original audio recording
- `test_audio_16khz.wav` - Converted to 16kHz mono PCM16 for testing (3.45 seconds)

## Usage

These files are used by:
- pytest integration tests (`tests/test_integration.py`)
- Manual testing script (`scripts/test_streaming_proxy.py`)

## Regenerating test_audio_16khz.wav

If you need to regenerate the test audio from the original:

```bash
ffmpeg -i test_data/Zebra.m4a -ar 16000 -ac 1 -f wav test_data/test_audio_16khz.wav
```

## Note

- `Zebra.m4a` is committed to git as the source material
- `test_audio_16khz.wav` is gitignored (generated from Zebra.m4a)
- If you clone the repository fresh, regenerate the WAV file using the command above
