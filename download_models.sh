#!/bin/bash
# Downloads the model files that are too big for GitHub. Run once from this folder:  ./download_models.sh
# (The face sensor, the face recogniser and the depth AI download themselves the first time they are needed.)
set -e
cd "$(dirname "$0")"
KOKORO="https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
for f in kokoro-v1.0.onnx voices-v1.0.bin; do
  if [ -f "$f" ]; then echo "✔ $f is already here"; else echo "⬇️  $f"; curl -L --fail -o "$f.part" "$KOKORO/$f" && mv "$f.part" "$f"; fi
done
echo "Done. Put your Gemini API key in cloud_key.txt (one line) – it is never committed (see .gitignore)."
