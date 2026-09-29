"""Register Qwen3.5's LM head with vLLM's existing embedding-LoRA wrapper.

vLLM 0.27.1 supports LogitsProcessorWithLoRA, but Qwen3_5ForCausalLMBase
omits embedding_modules, so loading JEV's lm_head adapter otherwise fails.
This changes module discovery only, not inference math or model weights.
"""
from pathlib import Path
import vllm

if vllm.__version__ != "0.27.1":
    raise RuntimeError("Re-audit the JEV LM-head registration for this vLLM version")
path = Path(vllm.__file__).parent / "model_executor/models/qwen3_5.py"
text = path.read_text()
start = text.index("class Qwen3_5ForCausalLMBase(")
pos = text.index("    packed_modules_mapping = {", start)
text = text[:pos] + '    embedding_modules = {"lm_head": "output_embeddings"}\n\n' + text[pos:]
path.write_text(text)
