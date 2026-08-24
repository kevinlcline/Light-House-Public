# Kokoro ONNX model files (not committed — ~340MB)

Download into this directory:

```bash
cd models/kokoro
curl -L -O https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
curl -L -O https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
```

Then set in `.env`:

```
TTS_ENABLED=true
KOKORO_MODEL_PATH=./models/kokoro
```

On local env, TTS auto-enables when both files are present unless `TTS_ENABLED` is set explicitly.
