#!/bin/bash
# Records from the default microphone until Enter, then transcribes it with
# the realtime model on the GPU and prints the text and the timings.
#
# Usage: ./tools/test_mic.sh [model.gguf]

set -e
cd "$(dirname "$0")/.."
MODEL="${1:-models/voxtral/Q4_K_M.gguf}"
WAV="$(mktemp --suffix=.wav)"
trap 'rm -f "$WAV"' EXIT

echo "Gravando... aperte Enter para parar."
pw-record --rate 16000 --channels 1 --format s16 "$WAV" &
REC=$!
read -r
kill "$REC"
wait "$REC" 2>/dev/null || true

SECONDS_OF_AUDIO=$(python3 -c "import wave; w = wave.open('$WAV'); print(f'{w.getnframes() / w.getframerate():.1f}')")
echo "Transcrevendo ${SECONDS_OF_AUDIO}s de áudio..."
START=$(date +%s.%N)
TEXT=$(./build/voxtral --model "$MODEL" --audio "$WAV" --gpu vulkan 2>/dev/null | head -1)
END=$(date +%s.%N)
echo
echo "Texto:$TEXT"
python3 -c "print(f'Tempo: {$END - $START:.2f}s (inclui carregar o modelo, cerca de 1,5s)')"
