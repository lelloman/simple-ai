# GLM-4.5V on Strix Halo — 2026-09-06

GLM-4.5V Q4_K_M runs on either Halo independently with the vision projector
enabled, at approximately 20 generated tokens per second.
This is a short performance and functionality smoke test, not a model-quality
evaluation or a production rollout.

## Model and runtime

- Model: [`ggml-org/GLM-4.5V-GGUF`](https://huggingface.co/ggml-org/GLM-4.5V-GGUF/tree/401d1a7a3b6a28971d7b0884eb05c50445d8852e).
- Revision: `401d1a7a3b6a28971d7b0884eb05c50445d8852e`.
- Weights: `GLM-4.5V-Q4_K_M-00001-of-00002.gguf` and
  `GLM-4.5V-Q4_K_M-00002-of-00002.gguf`, 63,633,581,632 bytes combined.
- Vision projector: `mmproj-GLM-4.5V-Q8_0.gguf`, 994,457,504 bytes.
- Hardware: Ryzen AI Max+ 395, Radeon 8060S, Vulkan-reported 128 GiB unified
  memory. Linux reports approximately 62 GiB system RAM with 64 GiB reserved
  for the GPU.
- Existing container: `localhost/strix-qwen38-mtp:463c0f6`, runtime commit
  `463c0f6`, RADV GFX1151 Vulkan backend. The container logs a ROCm detection
  error because only `/dev/dri` is exposed; inference explicitly uses Vulkan0.
- Full GPU offload, Flash Attention on, Q8_0 K/V cache, batch and microbatch
  512, 16 CPU threads. Server: 8,192 context, one slot, thinking disabled.
- No speculative decoding. The baseline above used isolated servers; the
  subsequent simple-ai integration is documented below.

## Results

| Measurement | halo1 | halo2 |
|---|---:|---:|
| llama-bench pp512, tokens/s, mean ± sample SD | 277.79 ± 13.45 | 272.99 ± 14.01 |
| llama-bench tg128, tokens/s, mean ± sample SD | 20.04 ± 0.03 | 19.94 ± 0.004 |
| Chat generation, tokens/s, median of 3 runs | 20.09 | 19.91 |
| Chat time to first token, seconds, median | 1.02 | 1.02 |
| Image request total time, seconds | 3.57 | 3.68 |
| Separate OCR request total time, seconds | 3.34 | 3.61 |

`llama-bench` runs each test three times. pp512 and tg128 are separate
synthetic tests with the benchmark's context allocation, not an 8K occupied
context. The LAN model copy was paused during halo1's benchmark and had
finished before halo2's tests. halo1's chat tests ran with the copy active.
Chat prompts contain 32–36 tokens, with 256 output
tokens per measured run; all three reached the output limit. They demonstrate
generation, not correctness of complete coding answers. The warmup is excluded.
Streaming throughput is `(completion_tokens - 1) / (end - first_token_time)`;
raw server timings are also included in the saved results.

The 640×400 image fixture contains `HALO TEST 7319`, `Total: EUR 42.70`, and a
red square, blue circle, and green triangle. On both machines, the combined request
correctly identified all shapes and colors but omitted both text lines.
A separate OCR request transcribed both lines exactly. This is a partial
instruction-following failure in the combined request, despite working OCR.

Both machines' post-test GPU memory counters were 65,810,157,568 bytes VRAM
(61.29 GiB) and approximately 62 MB GTT. These are whole-device snapshots, not
peak measurements or model-only allocations. Larger contexts and concurrency
have not been tested.

## Reproduction and artifacts

The experiment lives at `/home/lelloman/glm45v-test` on each host; its model
files are in `models/` and raw results are in `results/`. Download the two
Q4_K_M shards and Q8_0 projector from the pinned revision above into `models/`.
Copy the scripts and fixture in this directory to that experiment directory.

```bash
bash /home/lelloman/glm45v-test/start.sh
# Wait until curl -f http://127.0.0.1:18045/health succeeds.
cd /home/lelloman/glm45v-test
python3 evaluate.py
podman stop glm45v-test
bash bench.sh
```

The temporary servers were stopped after testing. They listen only on
`127.0.0.1:18045` when started. Stop the server before running
the standalone benchmark, and avoid concurrent inference or file transfers
during timing comparisons. The downloaded weights are retained for reuse.

Raw summary and benchmark JSON are saved in the per-host subdirectories here.
`ocr.json` contains the separate OCR request and response, including the image
data. `vision.png` is a deterministic synthetic fixture, generated with Pillow
using DejaVu Sans 32-point text and solid colored shapes.

## Simple-ai integration

Deployed on 2026-09-06 to both Halo runners and the homelab gateway.
Use `"model": "glm-4.5v"` with `/v1/chat/completions`, with either a plain
text message or OpenAI-style `text` and `image_url` content parts. Streaming
and non-streaming requests are supported.

Each runner's model catalog contains symlinks to both model shards. The
projector stays outside the catalog, and the container mounts its directory
read-only. The runner alias maps `glm-4.5v` to `GLM-4.5V-Q4_K_M`.

```toml
[engines.llama_cpp.models."GLM-4.5V-Q4_K_M"]
context_size = 8192
reasoning = false
fit = false
parallel = 1
default_max_tokens = 4096
extra_args = ["--mmproj", "/home/lelloman/glm45v-test/models/mmproj-GLM-4.5V-Q8_0.gguf", "-ctk", "q8_0", "-ctv", "q8_0", "-b", "512", "-ub", "512"]

[aliases.mappings]
"glm-4.5v" = "GLM-4.5V-Q4_K_M"
```

The runner now supports typed per-model `context_size`, used consistently
for server startup and reported context metadata. It takes precedence over
the engine-wide limit, which takes precedence over GGUF metadata. The build
also includes the existing working-tree image-forwarding support, allowing
image content only for llama.cpp profiles with a configured projector.

The live server config at
`/home/lelloman/homelab-data/simple-ai/config.toml` adds `glm-4.5v` to
`models.big`, after the existing entries. Other models and defaults remain
unchanged. The public model listing advertises `glm-4.5v` from the registered
runners. Large-model classification does not guarantee that generic
`class:big` requests choose this model; use the explicit alias for GLM vision.

Validation:

- `cargo test -p inference-runner`: 77 unit tests and 4 API tests passed.
- Release build succeeded; deployed runner SHA256:
  `4f3a8b3af824115b333e0516cb61ef5a20e172b60b8b452d2843cc80a5bd7714`.
- Both runners passed text and image/OCR requests in both streaming modes,
  using the public alias through the runner API (8 successful requests).
- Server `/v1/models` lists `glm-4.5v`.
- Authenticated gateway inference is pending a simple-ai device login.
- Changed Rust files pass rustfmt. Workspace-wide formatting reports
  pre-existing differences in `backend/src/main.rs` and
  `inference-runner/src/engine/registry.rs`.

Raw runner checks are in each host's `runner-integration.json`. Repeat with
`check-api.py --output results/runner-integration.json` on the host. For the
gateway, provide `--url`, `--image`, and `--token-file`; the token file is an
OAuth response JSON containing `access_token` and must remain private.

Rollback copies were retained as `config.toml.before-glm45v-20260906` and
`simple-ai-runner.before-glm45v-20260906` under `/home/lelloman` on each Halo,
plus `config.toml.before-glm45v-20260906` alongside the live server config.
The integration uses the normal runner-managed model lifecycle; the old
standalone test servers remain stopped.
