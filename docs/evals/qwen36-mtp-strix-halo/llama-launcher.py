#!/usr/bin/env python3
"""Select the validated Qwen3.6 runtime; preserve Podman for other models.

The runner supplies its existing Podman arguments followed by --server and
llama-server arguments. This process is replaced so runner unload still owns
the entire inference process group.
"""
import os
import pathlib
import sys

MODEL = 'Qwen3.6-35B-A3B-MXFP4_MOE.gguf'
RUNTIME = '/home/lelloman/flash-next/runtime/vulkan/llama-server'


def launch_command(args):
    if '--server' in args:
        server_args = args[args.index('--server')+1:]
        for i, arg in enumerate(server_args[:-1]):
            if arg in ('-m', '--model') and pathlib.Path(server_args[i+1]).name == MODEL:
                return [RUNTIME, *server_args]
    return ['podman', *args]


if __name__ == '__main__':
    command = launch_command(sys.argv[1:])
    os.execvp(command[0], command)
