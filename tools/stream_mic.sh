#!/bin/bash
# Live transcription from the default microphone, word by word.
# Ctrl+C stops the recording; the last words are still written before exiting.
#
# Usage: ./tools/stream_mic.sh [model.gguf] [extra voxtral-stream options]

cd "$(dirname "$0")/.."
MODEL="${1:-models/voxtral/Q4_K_M.gguf}"
shift 2>/dev/null

# Only arecord dies on Ctrl+C; voxtral-stream ignores it, sees the end of the
# audio and flushes the words still inside the model's delay.
arecord -q -f S16_LE -r 16000 -c 1 -t raw 2>/dev/null |
    (trap '' INT; exec ./build/voxtral-stream --model "$MODEL" "$@" 2> >(grep -v "ggml_vulkan\|radv" >&2))
