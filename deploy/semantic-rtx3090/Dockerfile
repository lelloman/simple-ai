# Host runtime pinned by prepare.sh before building this small derived layer.
FROM qwen38-27b-3090:latest
COPY enable-head-lora.py /tmp/enable-head-lora.py
RUN /app/venv/bin/python /tmp/enable-head-lora.py
