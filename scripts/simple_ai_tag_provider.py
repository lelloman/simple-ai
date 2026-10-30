#!/usr/bin/env python3
"""Run the standard async vLLM OpenAI server for the configured local tag checkpoint."""
import argparse
import os
import runpy
import sys


def server_args(model, path, port, vllm_args=None):
    return ['vllm.entrypoints.openai.api_server', '--model', path,
            '--served-model-name', model, '--host', '127.0.0.1', '--port', str(port)] + (vllm_args if vllm_args is not None else [
            '--dtype', 'bfloat16', '--max-model-len', '2048',
            '--max-num-batched-tokens', '16384', '--max-num-seqs', '32',
            '--gpu-memory-utilization', '0.92', '--enable-prefix-caching',
            '--distributed-executor-backend', 'uni'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--port', required=True, type=int)
    parser.add_argument('vllm_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    os.environ.update(HF_HOME=os.path.join(os.path.dirname(os.path.abspath(args.model_path)), 'cache'),
                      HF_HUB_OFFLINE='1', HF_HUB_DISABLE_IMPLICIT_TOKEN='1',
                      VLLM_USE_FLASHINFER_SAMPLER='0', VLLM_WORKER_MULTIPROC_METHOD='spawn')
    # A separator marks an explicit replacement, including an empty list.
    tuning = args.vllm_args[1:] if args.vllm_args[:1] == ['--'] else None
    sys.argv = server_args(args.model, args.model_path, args.port, tuning)
    # AsyncLLM owns an engine-core child; the runner manages the entire process group.
    runpy.run_module('vllm.entrypoints.openai.api_server', run_name='__main__')


if __name__ == '__main__':
    main()
