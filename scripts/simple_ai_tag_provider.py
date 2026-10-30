#!/usr/bin/env python3
"""Run the standard async vLLM OpenAI server with the qualified summary stack."""
import argparse
import os
import runpy
import sys

from simple_ai_summary_provider import check_checkpoint


def server_args(model, path, port):
    return ['vllm.entrypoints.openai.api_server', '--model', path,
            '--served-model-name', model, '--host', '127.0.0.1', '--port', str(port),
            '--dtype', 'bfloat16', '--max-model-len', '2048',
            '--max-num-batched-tokens', '16384', '--max-num-seqs', '32',
            '--gpu-memory-utilization', '0.92', '--enable-prefix-caching',
            '--distributed-executor-backend', 'uni']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--port', required=True, type=int)
    args = parser.parse_args()
    os.environ.update(HF_HOME=os.path.join(os.path.dirname(os.path.abspath(args.model_path)), 'cache'),
                      HF_HUB_OFFLINE='1', HF_HUB_DISABLE_IMPLICIT_TOKEN='1',
                      VLLM_USE_FLASHINFER_SAMPLER='0', VLLM_WORKER_MULTIPROC_METHOD='spawn')
    check_checkpoint(args.model_path)
    sys.argv = server_args(args.model, args.model_path, args.port)
    # AsyncLLM owns an engine-core child; the runner manages the entire process group.
    runpy.run_module('vllm.entrypoints.openai.api_server', run_name='__main__')


if __name__ == '__main__':
    main()
