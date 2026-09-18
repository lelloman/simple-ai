# Multilingual information extraction

`POST /v1/extractions` serves GLiNER2.5 through the authenticated gateway and managed inference runners. It supports entities, label classification, relations, and structured records in the same request. Use `class:information_extraction` for routed requests or `fastino/gliner2.5-multi-v1` with permission to select a specific model. This endpoint is separate from NLI `/v1/classifications`; scores have different semantics.

```bash
curl "$SIMPLE_AI_URL/v1/extractions" \
  -H "Authorization: Bearer $SIMPLE_AI_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "class:information_extraction",
    "input": ["Giulia Rossi lavora per Acme a Milano."],
    "schema": {
      "entities": {"person":"Named people", "organization":"Named companies", "location":"Named places"},
      "classifications": [{"task":"topic", "labels":["employment","travel","shopping"]}],
      "relations": ["works_for"]
    },
    "threshold": 0.5
  }'
```

`input` accepts one string or an array. The response contains `object: "list"`, resolved `model`, pinned `revision`, input-indexed `data` with a `result` object per text, `inference_ms`, and `usage` (`input_count`, `input_characters`). Entity and relation mentions include text, confidence, start and end. Offsets count Unicode code points, with an exclusive end, relative to the original input. Confidence is not a calibrated correctness probability. Single-label classification chooses a label; `threshold` does not make it abstain. Multi-label classification uses the threshold.

For records, add a structure with a declared anchor and fields:

```json
{
  "structures": {
    "purchase": {
      "anchor": "buyer",
      "fields": [
        {"name":"buyer", "dtype":"str", "cardinality":"required_one"},
        {"name":"item", "dtype":"str"},
        {"name":"price", "dtype":"str"}
      ]
    }
  }
}
```

Records use natural anchored decoding. Fields accept `str` (default) or `list`, an optional description, and optional cardinality (`optional_one`, `required_one`, `zero_or_more`, `one_or_more`). Task/structure names must be unique and cannot use the reserved output names `entities` and `relation_extraction`. Entity, relation and classification labels accept lists or label-to-description maps. Unknown request/schema options are rejected.

## Documents and limits

- At most 16 inputs; 200,000 Unicode characters total.
- Short mode: at most 1,500 characters per input; excessive input is rejected.
- Set `long_text: true` for up to 100,000 characters per input. Processing uses overlapping word chunks, merges duplicate mentions, and returns global offsets. Context does not span the whole document: relation endpoints and mentions must fit within a chunk.
- `chunk_size`: 256 by default, allowed 64–512. `chunk_overlap`: 64 by default, smaller than chunk size; long-mode chunks must advance by at least 32 words.
- At most 32 labels per group, 16 fields per record, 64 labels/fields across the schema, and 8,192 serialized schema bytes. Label names have a 128-character limit; descriptions 512 characters.
- `threshold`: 0–1, default 0.5. `overlap`: `flat` (default), `nested`, `allow`, `longest`.
- `splitter`: `whitespace` (default) or `char`. Evaluate character splitting for languages without whitespace boundaries.

These limits bound service work; they are not promises about quality or the underlying model context. Large documents have longer latency. The default provider request timeout is 180 seconds. Providers batch inputs within a request and serialize HTTP inference; overload returns 429. Model unload waits for in-flight provider requests, including when the caller disconnects.

## Runner setup

On each runner, copy `scripts/setup-extraction.sh` and `scripts/extraction-requirements.txt` together and run:

```bash
bash setup-extraction.sh /home/lelloman/.simple-ai-extraction cpu
```

Use `cu130` instead of `cpu` to install CUDA-capable PyTorch on a compatible NVIDIA host. The CUDA wheel also supports CPU execution. The installer creates an isolated environment and downloads the pinned model revision. `scripts/deploy-fleet.sh` copies the provider with the runner deployment; dependency installation is a separate prerequisite.

```toml
[engines.extraction]
enabled = true
command = ["/home/lelloman/.simple-ai-extraction/venv/bin/python", "/home/lelloman/simple-ai-extraction-provider.py"]
model_id = "fastino/gliner2.5-multi-v1"
revision = "235cf92d6d4318da9bfca0d08975c8fa7250d13b"
model_path = "/home/lelloman/.simple-ai-extraction/model"
device = "cpu"
num_threads = 8
batch_size = 4
startup_timeout_secs = 600
request_timeout_secs = 180
```

Each configured engine advertises `information_extraction` automatically and loads on demand. CPU mode can coexist with GPU chat models. To use CUDA, set `device = "cuda"` and assign `extraction = "cuda:0"` in `[engine_resources]` alongside the other CUDA engines. This participates in the runner's exclusive GPU ownership policy and may unload an existing large chat model. Halo runners initially use CPU; ROCm is not qualified by this rollout.

Gateway configuration:

```toml
[models]
information_extraction = ["fastino/gliner2.5-multi-v1"]
```

Deploy the gateway before updated runners: older gateways cannot decode the new capability enum. Runner configs without extraction remain compatible because it defaults to disabled. For rollback, disable extraction on runners before rolling back the gateway.

## Validation and measurement

- `cargo test --workspace --all-features`
- `python3 -m unittest discover -s tests -p test_extraction_provider.py`
- `scripts/check-extraction-local.py` exercises the actual gateway, runner and checkpoint using isolated configuration, temporary SQLite, and mock OIDC. See its arguments for the prepared model/Python and FastText paths.
- `scripts/benchmark-extraction.py` measures identical CPU/CUDA workloads; [RTX results](evals/gliner25/README.md).

The standalone playground remains separate at `/home/lelloman/gliner25-demo` on the development machine. Its demo request shape and limits are independent of this API contract.
