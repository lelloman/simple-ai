"""Backport standalone Qwen3.5 text serving to the isolated AMD runtime.

Uses existing vLLM 0.19.1 kernels and hybrid-cache helpers, the text-only
configuration hook from vLLM 0.27.1, and JEV LM-head adapter registration.
Does not change checkpoint files or inference kernels. Version-gated.
"""
from pathlib import Path
import vllm

if vllm.__version__ != "0.19.1+rocm7.13.0rc2":
    raise RuntimeError("Re-audit the JEV LM-head registration for this vLLM version")
path = Path(vllm.__file__).parent / "model_executor/models/qwen3_5.py"
text = path.read_text()
# Allow a stopped benchmark container to restart with its compiled kernels.
if '    embedding_modules = {"lm_head": "output_embeddings"}' in text:
    root = Path(vllm.__file__).parent
    assert '    is_hybrid = True' in text
    assert '"Qwen3_5ForCausalLM":' in (root / "model_executor/models/registry.py").read_text()
    assert 'class Qwen3_5ForCausalLMConfig(' in (root / "model_executor/models/config.py").read_text()
    raise SystemExit(0)
start = text.index("class Qwen3_5ForCausalLMBase(")
pos = text.index("    packed_modules_mapping = {", start)
text = text[:pos] + '    embedding_modules = {"lm_head": "output_embeddings"}\n\n' + text[pos:]
path.write_text(text)

# This AMD release has the text implementation but only registers the
# multimodal architecture. Register the actual text-only model explicitly.
registry = Path(vllm.__file__).parent / "model_executor/models/registry.py"
source = registry.read_text()
marker = "_TEXT_GENERATION_MODELS = {"
assert marker in source
assert '"Qwen3_5ForCausalLM":' not in source
source = source.replace(marker, marker + '\n    "Qwen3_5ForCausalLM": ("qwen3_5", "Qwen3_5ForCausalLM"),', 1)
registry.write_text(source)

# The text class was previously only an inner module, so the multimodal
# wrapper owns hybrid-cache metadata. Reuse those exact class methods for
# standalone text serving; no kernel or weight changes.
source = path.read_text()
wrapper_start = source.index("class Qwen3_5ForConditionalGeneration(")
methods_start = source.index("    @classmethod\n    def get_mamba_state_dtype_from_config", wrapper_start)
methods_end = source.index("\n\n########################################################", methods_start)
methods = source[methods_start:methods_end]
base_start = source.index("class Qwen3_5ForCausalLMBase(")
base_end = source.index("\n\nclass Qwen3_5ForCausalLM(", base_start)
source = source[:base_end] + '\n\n' + methods + source[base_end:]
pos = source.index('    embedding_modules = ', base_start)
source = source[:pos] + '    is_hybrid = True\n\n' + source[pos:]
path.write_text(source)

# Backport the text-only config hook from the working vLLM 0.27.1 runtime:
# preserve JEV's float32 recurrent state and remove inherited multimodal
# position fields. The snapshot on disk remains byte-for-byte unchanged.
config_path = Path(vllm.__file__).parent / "model_executor/models/config.py"
source = config_path.read_text()
marker = "MODELS_CONFIG_MAP: dict[str, type[VerifyAndUpdateConfig]] = {"
assert marker in source
hook = '''class Qwen3_5ForCausalLMConfig(Qwen3_5ForConditionalGenerationConfig):
    @staticmethod
    def verify_and_update_config(vllm_config: "VllmConfig") -> None:
        Qwen3_5ForConditionalGenerationConfig.verify_and_update_config(vllm_config)
        hf_text_config = vllm_config.model_config.hf_text_config
        rope_parameters = getattr(hf_text_config, "rope_parameters", None)
        if rope_parameters is not None:
            rope_parameters.pop("mrope_section", None)
            rope_parameters.pop("mrope_interleaved", None)


'''
source = source.replace(marker, hook + marker + '\n    "Qwen3_5ForCausalLM": Qwen3_5ForCausalLMConfig,', 1)
config_path.write_text(source)

