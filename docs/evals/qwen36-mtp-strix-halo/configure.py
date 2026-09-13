#!/usr/bin/env python3
"""Stage or apply the measured Halo configuration, retaining a rollback copy."""
import argparse
import datetime
import json
import pathlib
import re
import shutil
import tomllib

parser = argparse.ArgumentParser()
parser.add_argument('--settings', required=True, type=pathlib.Path)
parser.add_argument('--apply', action='store_true')
args = parser.parse_args()
settings = json.loads(args.settings.read_text())
config = pathlib.Path('/home/lelloman/config.toml')
old = config.read_text()
parsed = tomllib.loads(old)
engine = parsed['engines']['llama_cpp']
wrapper = '/home/lelloman/simple-ai-llama-launcher.py'
assert engine['server_binary'] in ('podman', wrapper), 'Unexpected existing launcher'
model = 'Qwen3.6-35B-A3B-MXFP4_MOE'
profile = engine['models'][model]
assert not profile.get('extra_args'), 'Review existing per-model arguments before replacing them'
draft = f'/home/lelloman/flash-next/rocmfpx/drafts/Qwen3.6-35B-A3B-MTP-{settings["draft_quant"]}.gguf'
assert pathlib.Path(draft).is_file(), 'Missing draft weights'
assert pathlib.Path(wrapper).is_file(), 'Missing launcher'
assert pathlib.Path('/home/lelloman/flash-next/runtime/vulkan/llama-server').is_file(), 'Missing portable runtime'
extra = ['-md', draft, '--spec-type', 'draft-mtp', '--spec-draft-n-max', str(settings['draft']),
         '--spec-draft-ngl', 'all', '--spec-draft-device', 'Vulkan0',
         '--spec-draft-p-min', str(settings['pmin']), '--spec-draft-p-split', '0.1',
         '--spec-draft-type-k', settings['kv'], '--spec-draft-type-v', settings['kv'],
         '-ctk', settings['kv'], '-ctv', settings['kv'],
         '-b', str(settings['batch']), '-ub', str(settings['batch']), '-t', str(settings['threads'])]
header = '[engines.llama_cpp.models."'+model+'"]'
start = old.index(header)
end_match = re.search(r'\n\[', old[start+len(header):])
end = start+len(header)+end_match.start() if end_match else len(old)
section = old[start:end].rstrip()
# This fast model advertises only the "none" effort: make its default match.
if 'default_effort' not in section and 'supported_efforts = ["none"]' in section:
    section = section.replace('supported_efforts = ["none"]', 'supported_efforts = ["none"], default_effort = "none"')
section += '\nfit = false\nparallel = 1\nextra_args = '+json.dumps(extra)+'\n'
new = old[:start]+section+old[end:]
new = new.replace('server_binary = "podman"', 'server_binary = '+json.dumps(wrapper), 1)
updated = tomllib.loads(new)
assert updated['engines']['llama_cpp']['server_args'] == engine['server_args']
for name, before in engine['models'].items():
    if name != model:
        assert updated['engines']['llama_cpp']['models'][name] == before
candidate = config.with_name('config.toml.qwen36-mtp-candidate')
candidate.write_text(new)
shutil.copymode(config, candidate)
print(json.dumps({'candidate':str(candidate),'profile':updated['engines']['llama_cpp']['models'][model]},indent=2))
if args.apply:
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    backup = config.with_name('config.toml.before-qwen36-mtp-'+stamp)
    shutil.copy2(config, backup)
    candidate.replace(config)
    print('Applied; rollback config: '+str(backup))
