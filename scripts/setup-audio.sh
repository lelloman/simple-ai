#!/usr/bin/env bash
# Run on each audio embedding runner host. Builds the environment used by
# simple-ai-audio-provider.py (MusicFM MSD and MIT AST AudioSet):
#   <install_dir>/.venv_musicfm             Python 3.12 venv
#   <install_dir>/third_party/musicfm       MusicFM source at a pinned commit
#   <install_dir>/third_party/musicfm/data  MSD checkpoint and normalisation stats
# The default location matches the runner configs' --musicfm-root default.
# Safe to rerun: completed steps are skipped, downloads are checksum-verified.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
install_dir="${1:-$HOME/rustentia_musicfm}"
musicfm_repo="https://github.com/minzwon/musicfm"
musicfm_commit="b83ebedb401bcef639b26b05c0c8bee1dc2dfe71"
weights_url="https://huggingface.co/minzwon/MusicFM/resolve/main"
declare -A weights_sha256=(
  [msd_stats.json]="c36c61ab10ca4d2e7fdfefc3fcc15205316bec276a06a47baa3641a62c546f22"
  [pretrained_msd.pt]="218b483a0256ddef736267425fabb166fd97008983696bb9270def464b47bded"
)

musicfm_dir="$install_dir/third_party/musicfm"
mkdir -p "$install_dir/third_party"
if [ ! -d "$musicfm_dir/.git" ]; then
  git clone "$musicfm_repo" "$musicfm_dir"
fi
git -C "$musicfm_dir" fetch --quiet origin "$musicfm_commit" || true
git -C "$musicfm_dir" checkout --quiet "$musicfm_commit"

mkdir -p "$musicfm_dir/data"
for name in "${!weights_sha256[@]}"; do
  target="$musicfm_dir/data/$name"
  expected="${weights_sha256[$name]}"
  if [ -s "$target" ] && echo "$expected  $target" | sha256sum --check --status; then
    continue
  fi
  curl -fL --retry 3 -o "$target.part" "$weights_url/$name"
  echo "$expected  $target.part" | sha256sum --check --quiet
  mv -f "$target.part" "$target"
done

venv="$install_dir/.venv_musicfm"
if [ ! -x "$venv/bin/python" ]; then
  python3.12 -m venv "$venv"
fi
"$venv/bin/pip" install --quiet --upgrade pip
"$venv/bin/pip" install --quiet -r "$script_dir/audio-requirements.txt"
"$venv/bin/python" -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__, 'cuda', torch.cuda.is_available())"
echo "Audio embedding environment ready: $install_dir"
