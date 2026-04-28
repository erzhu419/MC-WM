"""Generate MC-WM overview figures via gpt-image-2 on ruoli.dev.

Exact structural copy of /home/erzhu419/mine_code/Asumption Agent/paper/generate_intuition_figs.py
(which was verified working at 17:32 on 2026-04-24).
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time
from pathlib import Path
import urllib.request
import urllib.error


API_KEY = "sk-bW8P9HtMGAy3DjDyEZK8vtOxOVAyUSCRhAuTkGcTZt28AdME"
API_URL = "https://ruoli.dev/v1/images/generations"
OUT_DIR = Path(__file__).parent.parent  # figures/
MODEL = "gpt-image-2"

HERE = Path(__file__).parent

FIGURES = [
    {
        "name": "overview_claude",
        "size": "1024x1024",
        "prompt": (HERE / "claude_prompt.txt").read_text(),
    },
    {
        "name": "overview_gpt55",
        "size": "1024x1024",
        "prompt": (HERE / "gpt55_prompt.txt").read_text(),
    },
    {
        "name": "residual_bellman_claude",
        "size": "1024x1024",  # gateway only supports 1024x1024 / 2048x2048
        "prompt": (HERE / "residual_bellman_claude_prompt.txt").read_text(),
    },
    {
        "name": "residual_bellman_gpt55",
        "size": "1024x1024",
        "prompt": (HERE / "residual_bellman_gpt55_prompt.txt").read_text(),
    },
    {
        "name": "architecture_v2_claude",
        "size": "1024x1024",
        "prompt": (HERE / "architecture_v2_claude_prompt.txt").read_text(),
    },
    {
        "name": "rahd_overview_claude",
        "size": "1024x1024",
        "prompt": (HERE / "rahd_overview_claude_prompt.txt").read_text(),
    },
    {
        "name": "gap_taxonomy_v2_claude",
        "size": "1024x1024",
        "prompt": (HERE / "gap_taxonomy_v2_claude_prompt.txt").read_text(),
    },
    {
        "name": "architecture_v2_gpt55",
        "size": "1024x1024",
        "prompt": (HERE / "architecture_v2_gpt55_prompt.txt").read_text(),
    },
    {
        "name": "rahd_overview_gpt55",
        "size": "1024x1024",
        "prompt": (HERE / "rahd_overview_gpt55_prompt.txt").read_text(),
    },
    {
        "name": "gap_taxonomy_v2_gpt55",
        "size": "1024x1024",
        "prompt": (HERE / "gap_taxonomy_v2_gpt55_prompt.txt").read_text(),
    },
]


def generate_one(fig: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{fig['name']}.png"
    payload = {
        "model": MODEL,
        "prompt": fig["prompt"],
        "size": fig.get("size", "1024x1024"),
        "n": 1,
    }
    req = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} on {fig['name']}:\n{err_body}") from e
    data = json.loads(body)
    if "data" not in data or not data["data"]:
        raise RuntimeError(f"unexpected response for {fig['name']}:\n{body[:500]}")
    item = data["data"][0]
    if item.get("b64_json"):
        out_path.write_bytes(base64.b64decode(item["b64_json"]))
    elif item.get("url"):
        with urllib.request.urlopen(item["url"], timeout=120) as img_resp:
            out_path.write_bytes(img_resp.read())
    else:
        raise RuntimeError(f"no b64_json or url in response: {item!r}")
    elapsed = time.time() - t0
    print(f"  [{elapsed:5.1f}s] {out_path.name}  "
          f"({out_path.stat().st_size/1024:.0f} KB)", flush=True)
    return out_path


def main():
    which = sys.argv[1:] if len(sys.argv) > 1 else None
    for fig in FIGURES:
        if which and fig["name"] not in which:
            continue
        print(f"Generating {fig['name']} ({fig.get('size', '1024x1024')})...", flush=True)
        try:
            generate_one(fig, OUT_DIR)
        except Exception as e:
            print(f"  FAILED: {e}", flush=True)


if __name__ == "__main__":
    main()
