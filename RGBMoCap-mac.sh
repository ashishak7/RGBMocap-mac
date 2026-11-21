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
# The app targets Blender 4.5 LTS; prefer it, otherwise use whatever is installed.
if [ -z "${RGBMOCAP_BLENDER:-}" ]; then
    for app in "$root"/runtime_character/Blender*.app /Applications/Blender*4.5*.app "$HOME"/Applications/Blender*4.5*.app /Applications/Blender.app; do
        [ -x "$app/Contents/MacOS/Blender" ] && export RGBMOCAP_BLENDER="$app/Contents/MacOS/Blender" && break
    done
fi
exec "$python" -X utf8 app.py "$@"
