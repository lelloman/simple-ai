# Build only on RTX, using the same local vLLM 0.29.0 base as the benchmark.
FROM simple-ai-qwen38-cyber:7759002
ENV PATH="/app/venv/bin:${PATH}" \
    HF_HOME=/w/summary-quant-cache \
    HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
    HF_HUB_OFFLINE=1
RUN /app/venv/bin/pip install --no-cache-dir 'llmcompressor==0.14.0' 'compressed-tensors==0.20.1'
COPY scripts/simple_ai_summary_provider.py /opt/summary/simple_ai_summary_provider.py
COPY scripts/summary-quantization/quant.py /opt/summary/quant.py
ENTRYPOINT ["/app/venv/bin/python", "/opt/summary/quant.py"]
