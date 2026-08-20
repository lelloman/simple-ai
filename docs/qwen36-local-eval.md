# Qwen3.6 Local Evaluation

This note captures the local Qwen3.6 rollout for SimpleAI.

## Candidates

| Canonical ID | Upstream model | Shape | SimpleAI class | Runtime note |
|--------------|----------------|-------|------------------------|-----------------------|
| `qwen3.6-27b` | `Qwen/Qwen3.6-27B` | dense, 27B language model with vision encoder | `big` | Use as the capable dense class model. |
| `qwen3.6-35b-a3b` | `Qwen/Qwen3.6-35B-A3B` | MoE, 35B total with ~3B active | `fast` | Use as the fast chat model; MXFP4 fits on RTX 24GB and Halo machines. |

Both official repositories are Apache-2.0 and expose `image-text-to-text` cards. The upstream cards report BF16 safetensors model sizes of 28B params for `Qwen3.6-27B` and 36B params for `Qwen3.6-35B-A3B`, so raw BF16 is not a 24GB target.

## Recommended Mapping

1. Put `qwen3.6-27b` in `class:big`.
2. Put `qwen3.6-35b-a3b` in `class:fast`.
3. Keep `gpt-oss:120b` out of `class:big`; direct requests can still use runner aliases if needed.

## Backend Config

Add the canonical names to the backend model class list:

```toml
[models]
fast = [
    "qwen3.6-35b-a3b",
]
big = [
    "qwen3.6-27b",
]
```

## Runner Config: llama.cpp

Use aliases so users request stable canonical names while the runner uses the discovered GGUF file IDs.

```toml
[engines.llama_cpp]
enabled = true
model_dir = "/models"
server_binary = "/usr/bin/llama-server"
gpu_layers = 99
context_size = 8192
max_servers = 1
max_vram_gb = 24.0
startup_timeout_secs = 300
extra_args = ["--flash-attn", "on"]

[engines.llama_cpp.model_memory_gb]
# Replace these estimates after checking actual GGUF file sizes and runtime logs.
"Qwen3.6-27B-Q4_K_M" = 18.0
"Qwen3.6-35B-A3B-Q4_K_M" = 23.0

[aliases.mappings]
"qwen3.6-27b" = "Qwen3.6-27B-Q4_K_M"
"qwen3.6-35b-a3b" = "Qwen3.6-35B-A3B-Q4_K_M"
```

If the downloaded GGUF is sharded, SimpleAI uses the first shard name without `-00001-of-NNNNN` as the local model ID.

## Test Commands

After the backend and runner are connected, verify model discovery:

```bash
curl -H "Authorization: Bearer $SIMPLEAI_TEST_TOKEN" \
  https://ai.lelloman.com/v1/models
```

Smoke-test routing through the existing scripts:

```bash
python tests/test_model_routing.py \
  --token-binary ./get-token \
  --model qwen3.6-27b \
  --timeout 300

python tests/test_model_routing.py \
  --token-binary ./get-token \
  --model class:big \
  --timeout 300
```

Then run a low-concurrency workload test:

```bash
python tests/test_workload_routing.py \
  --token-binary ./get-token \
  --model qwen3.6-27b \
  --requests 3 \
  --workers 1 \
  --timeout 300
```

Repeat the same commands for `qwen3.6-35b-a3b` only after 27B has a known-good baseline.

## Measurements To Record

Record these for each quantization and context size:

| Field | Notes |
|-------|-------|
| GGUF repo and filename | Include quantization and shard count. |
| `context_size` | Start at 8192; increase only after stable runs. |
| `gpu_layers` | Keep at 99 for full offload attempts; lower if needed. |
| startup time | Runner logs show llama-server readiness time. |
| first-token latency | Use chat timing from the API response or audit logs. |
| decode tokens/sec | Compare same prompt across candidates. |
| peak VRAM | Use `nvidia-smi` during load and generation. |
| failure mode | OOM, unsupported arch, slow CPU offload, repetition, bad tool calls. |

## Decision Rule

Use `qwen3.6-27b` as the default `class:big` model and `qwen3.6-35b-a3b` as the default `class:fast` model. Keep other large models out of the class lists unless they beat the Qwen3.6 pair on quality, latency, or reliability.
