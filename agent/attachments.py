"""Pictures dropped into a local launcher's window.

A terminal types a dropped file in as its path: Trinidad Head and Terminal escape
spaces with backslashes, other apps wrap the path in quotes, a browser may hand
over a file:// URL. find_images() pulls every path that names a real picture
out of the typed line and returns the words that are left, so the model gets
the picture itself instead of a path it can't open.
"""
import os
import re
import subprocess
import tempfile
from urllib.parse import unquote

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif")
# Formats the model's image loader can't read are converted to PNG first.
CONVERT_EXTS = (".tif", ".tiff", ".heic", ".heif")

_TOKEN = re.compile(r"'[^']*'|\"[^\"]*\"|(?:\\.|\S)+")


def _unquote(token):
    if len(token) >= 2 and token[0] == token[-1] and token[0] in "'\"":
        path = token[1:-1]
    else:
        path = re.sub(r"\\(.)", r"\1", token)
    if path.startswith("file://"):
        path = unquote(path[len("file://"):])
    return os.path.expanduser(path)


def _loadable(path):
    if not path.lower().endswith(CONVERT_EXTS):
        return path
    out = os.path.join(tempfile.gettempdir(), "local-ai-drops",
                       os.path.splitext(os.path.basename(path))[0] + ".png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    r = subprocess.run(["sips", "-s", "format", "png", path, "--out", out],
                       capture_output=True)
    return out if r.returncode == 0 and os.path.exists(out) else None


def find_images(text):
    """(text without the picture paths, [picture paths the model can load])."""
    images, kept, last = [], [], 0
    for m in _TOKEN.finditer(text):
        path = _unquote(m.group(0))
        if not (path.lower().endswith(IMAGE_EXTS) and os.path.isfile(path)):
            continue
        loadable = _loadable(path)
        if loadable is None:
            continue
        images.append(loadable)
        kept.append(text[last:m.start()])
        last = m.end()
    if not images:
        return text, []
    kept.append(text[last:])
    rest = re.sub(r"[ \t]{2,}", " ", "".join(kept)).strip()
    return rest, images


def has_vision(model_path):
    """True when the model's config carries a vision tower."""
    import glob
    import json

    paths = []
    if os.path.isdir(os.path.expanduser(model_path)):
        paths.append(os.path.join(os.path.expanduser(model_path), "config.json"))
    repo = "models--" + model_path.replace("/", "--")
    paths += glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/{repo}/snapshots/*/config.json"))
    for p in paths:
        try:
            with open(p) as f:
                if "vision_config" in json.load(f):
                    return True
        except (OSError, ValueError):
            continue
    return False
