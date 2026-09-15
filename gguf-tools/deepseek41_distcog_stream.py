#!/usr/bin/env python3
"""Stream the gated distcog V4.1 checkpoint shard-by-shard into a DS4 GGUF.

Requires Hugging Face access to distributedcognition/DeepSeek-V4.1-Flash-abliterated.
Only shards containing the two abliterated tensor families are downloaded. A
weight shard plus any separate scale shard is kept only for the duration of the
patch, then removed. The destination should be an APFS clonefile copy of the
stock DS4 V4.1 Q2/Q4 GGUF.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

from huggingface_hub import HfApi, hf_hub_download

from deepseek41_patch_distcog import SOURCE_PATTERNS, patch as patch_shards
from deepseek41_quantize import scale_name

DEFAULT_REPO = "distributedcognition/DeepSeek-V4.1-Flash-abliterated"
META_FILES = ("model.safetensors.index.json", "config.json", "abliteration_config.json")


def wanted_source(name):
    return any(pattern.match(name) for pattern, _, _ in SOURCE_PATTERNS)


def download(repo, filename, local_dir):
    return hf_hub_download(repo, filename, repo_type="model", local_dir=local_dir)


def main(args):
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    work = Path(args.work_dir).resolve()
    meta = work / "meta"
    shards = work / "shards"
    meta.mkdir(parents=True, exist_ok=True)
    shards.mkdir(parents=True, exist_ok=True)

    try:
        for filename in META_FILES:
            download(args.repo, filename, str(meta))
    except Exception as error:
        text = str(error)
        if "403" in text or "gated" in text.lower() or "restricted" in text.lower():
            raise SystemExit(
                f"HF access is not approved for {args.repo}. Accept the repository access terms first; "
                "then rerun this command."
            ) from error
        raise

    index = json.loads((meta / "model.safetensors.index.json").read_text())
    weight_map = index.get("weight_map", {})
    wanted = sorted(name for name in weight_map if wanted_source(name))
    if not wanted:
        raise SystemExit("source index contains no distcog patch tensors")

    # Each modified FP8/I8 matrix needs its scale. The official V4.1 source
    # names the scale by replacing .weight with .scale. If a source happens to
    # be BF16/F16, a missing scale is harmless and the patcher will ignore it.
    batches = {}
    for name in wanted:
        ws = weight_map[name]
        ss = weight_map.get(scale_name(name))
        key = tuple(sorted({x for x in (ws, ss) if x}))
        batches.setdefault(key, []).append(name)

    print(f"repo={args.repo}")
    print(f"patch tensors={len(wanted)} shard-batches={len(batches)}")
    print(f"destination={args.gguf}")

    # Track downloaded shards to avoid fetching the same shard repeatedly when
    # the index puts several tensor families together.
    local = {}
    journal = args.journal or args.gguf + ".distcog-s3.json"
    for batch_index, (needed, names) in enumerate(sorted(batches.items()), 1):
        paths = []
        for shard in needed:
            path = local.get(shard)
            if path is None or not os.path.exists(path):
                print(f"[{batch_index}/{len(batches)}] download {shard}", flush=True)
                path = download(args.repo, shard, str(shards))
                local[shard] = path
            paths.append(path)

        class PatchArgs:
            gguf = args.gguf
            shard = paths
            journal = journal
            force = False
            quants_library = args.quants_library

        patch_shards(PatchArgs())

        # If no future batch needs a downloaded shard, release it immediately.
        future = set()
        for future_needed in list(sorted(batches))[batch_index:]:
            future.update(future_needed)
        if not args.keep_shards:
            for shard in list(local):
                if shard not in future:
                    path = local.pop(shard)
                    try:
                        os.remove(path)
                        print(f"released {shard}", flush=True)
                    except FileNotFoundError:
                        pass

    doc = json.loads(Path(journal).read_text()) if os.path.exists(journal) else {"patched": {}}
    patched = doc.get("patched", {})
    expected_targets = 2 * 40
    if len(patched) != expected_targets:
        raise SystemExit(f"patch incomplete: journal has {len(patched)}/{expected_targets} target tensors")
    print(f"PASS: patched {len(patched)}/{expected_targets} distcog target tensors")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gguf", required=True)
    p.add_argument("--repo", default=DEFAULT_REPO)
    p.add_argument("--work-dir", default="dist-v41-stream")
    p.add_argument("--journal")
    p.add_argument("--keep-shards", action="store_true")
    suffix = "dylib" if sys.platform == "darwin" else "so"
    p.add_argument("--quants-library", default=str(Path(__file__).with_name(f"libds4quants.{suffix}")))
    main(p.parse_args())
