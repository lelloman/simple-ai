#!/usr/bin/env bash
# Run on halo1 as lelloman after staging the image, downloader and harness.
set -euo pipefail
cd /home/lelloman/halogen-benchmark
exec > >(tee -a experiment.log) 2>&1
date -Is
image=ghcr.io/peonist-ai/halogen-flash-server:0.16.2
checkpoint=Qwen3.8-Flash-Next-Uncensored.IQ4_XS.gguf
server=halogen-uncensored-benchmark
cache_mode=${HALOGEN_BENCH_CACHE:-1}
run_id=${HALOGEN_BENCH_RUN_ID:-uncensored}
cleanup() {
  podman logs "$server" > halogen-server.log 2>&1 || true
  podman stop --time 30 "$server" || true
  systemctl --user start simple-ai-runner
}
trap cleanup EXIT
until test -f "models/$checkpoint"; do
  if test "$(podman inspect --format '{{.State.Running}}' halogen-download-uncensored-iq4xs)" != true; then
    podman logs halogen-download-uncensored-iq4xs
    echo 'Download exited before the checkpoint became available.'
    exit 1
  fi
  sleep 20
done
test -f models/qwen38-flash-next-mtp.hgn
systemctl --user stop simple-ai-runner
free -h
uname -r
podman run -d --replace --name "$server" \
  -p 127.0.0.1:8731:8731 --device /dev/kfd --device /dev/dri \
  --group-add keep-groups --ipc=host --ulimit memlock=-1:-1 \
  --security-opt label=disable \
  -v "$PWD/models:/models:ro" \
  -e "HALOGEN_CHECKPOINT=/models/$checkpoint" \
  -e HALOGEN_TOKENIZER=/models/tokenizer \
  -e HALOGEN_MTP_HEAD=/models/qwen38-flash-next-mtp.hgn \
  -e HALOGEN_KV_SLOTS=4 -e HALOGEN_CTX=262144 \
  -e HALOGEN_KV_POOL_POSITIONS=262144 -e HALOGEN_MAX_TOK=16384 \
  -e "HALOGEN_PROMPT_CACHE=$cache_mode" "$image"
ready=0
for ((i=0; i<180; i++)); do
  if curl -fsS http://127.0.0.1:8731/v1/models > served-models.json; then
    ready=1
    break
  fi
  if test "$(podman inspect --format '{{.State.Running}}' "$server")" != true; then
    podman logs "$server"
    exit 1
  fi
  sleep 5
done
test "$ready" = 1
model=$(python3 -c 'import json; print(json.load(open("served-models.json"))["data"][0]["id"])')
python3 benchmark.py --url http://127.0.0.1:8731 --model "$model" \
  --out "results-halogen-$run_id-smoke" --rows 8 --output-tokens 64 --repeats 1 --concurrency 1
python3 benchmark.py --url http://127.0.0.1:8731 --model "$model" \
  --out "results-halogen-$run_id" --rows 256,1024,2048 \
  --output-tokens 128 --repeats 1 --concurrency 4
echo 'BENCHMARK COMPLETE'
date -Is
