#!/bin/sh
# macOS launcher: runs the editable source in the local .venv (see README-mac notes in commit history).
root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
python="$root/.venv/bin/python"
if [ ! -x "$python" ]; then
    echo "Missing $python. Create it with: uv venv .venv --python 3.10 && uv pip install --python .venv/bin/python torch==2.7.1 torchvision==0.22.1 numpy==1.23.5 -r requirements-mac.txt"
    exit 1
fi
cd -- "$root" || exit 1
export PATH="/opt/homebrew/bin:$PATH" PYTORCH_ENABLE_MPS_FALLBACK=1
exec "$python" -X utf8 app.py "$@"
