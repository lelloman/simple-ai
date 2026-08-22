# Qwen3.8 27B uncensored on one RTX 3090

This overlay pins the upstream 3090 optimization stack and the uncensored
W4A16 vision/MTP checkpoint used by Simple-AI.

- Upstream commit: `a75ee4be40098e9d0b239cf4550ee12c4ac49338`
- Model revision: `c2f5341e7c9a1c55e6d61cfc8e6d0ca897bd7443`
- Model: `twolven/Qwen3.8-27B-abliterated-AWQ-MTP`
- Required host driver: NVIDIA 580 or newer
- Required free disk for a fresh preparation: 120 GiB

This checkpoint is used because it retains the vision processor and MTP
weights while publishing the model in the compressed-tensors AWQ W4A16 layout
expected by the optimization pipeline. "Abliterated" changes model behavior;
it does not change the serving, quantization, vision, tool-call, or reasoning
protocols used by Simple-AI.

Run `prepare-host.sh /home/lelloman/qwen38-serving`. It creates no API key if
one already exists, binds vLLM to loopback through the Compose override, and
never downloads the checkpoint-specific base-model fast variant.

The shortest deployable path is the `base` profile: text-only, 49,152-token
context, MTP-4 speculation, and no checkpoint-specific calibration. On the
project RTX 3090 it measured 126.7 tok/s median steady-state decode and about
123.9 tok/s median end-to-end across ten 512-token greedy requests, with 100 ms
median TTFT. Text, reasoning, and tool-call checks passed. Run `accept-base.sh`
to reproduce those gates. Expect roughly three minutes for a cold server start
and under a minute for the benchmark after the model has been prepared.

The expensive uncensored GPTQ calibration is an optional later phase via
`calibrate-fast.sh`; do not select the `fast` or `long` profiles until
`accept-fast.sh` passes.

After calibration, `accept-fast.sh` runs the upstream verifier, checkpoint
smokes for text/reasoning/tools/vision, and two full single-user benchmark
passes. It refuses promotion unless the warmed C1 greedy decode rate is at
least 100 tok/s.

On the current `rtx.homelab` Ubuntu 24.04 host, run
`sudo ./install-driver.sh` and reboot before calibration. The script installs
the repository-provided 580 driver but deliberately preserves the working Snap
Docker NVIDIA runtime configuration.

The shortest staged host sequence is:

```bash
ssh rtx.homelab
sudo /home/lelloman/simple-ai-qwen38-deploy/install-driver.sh
sudo reboot

# After the machine returns:
/home/lelloman/simple-ai-qwen38-deploy/prepare-host.sh /home/lelloman/qwen38-serving
/home/lelloman/simple-ai-qwen38-deploy/accept-base.sh /home/lelloman/qwen38-serving
```

Configure Simple AI's `qwen38-base` engine model as shown in
`inference-runner/config.example.toml`. Only after `accept-base.sh` writes a
PASS summary should the staged runner and backend binaries/configuration be
promoted. The existing llama.cpp service is left untouched until that cutover.

To evaluate the optional calibrated profile later:

```bash
/home/lelloman/simple-ai-qwen38-deploy/calibrate-fast.sh /home/lelloman/qwen38-serving
/home/lelloman/simple-ai-qwen38-deploy/accept-fast.sh /home/lelloman/qwen38-serving
```
