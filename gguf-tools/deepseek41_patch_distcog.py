#!/usr/bin/env python3
"""Patch a DwarfStar DeepSeek V4.1 Q2/Q4 GGUF with distcog scale-3 weights.

The distributedcognition checkpoint differs from stock only in two tensor
families: layers.N.attn.wo_b.weight and layers.N.ffn.shared_experts.w2.weight.
DwarfStar stores both families as Q8_0 in its V4.1 GGUF recipes, so the routed
experts and 188+ GiB disk-only Engram tables can be reused byte-for-byte.

This tool intentionally patches only those two families. It is designed for a
clonefile/APFS CoW copy of the stock GGUF, allowing conversion with very little
additional disk space. Source shards may be supplied one at a time; every
matching tensor found in a shard is requantized to Q8_0 and written in-place.
A JSON journal prevents accidental double/mismatched application.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys

from deepseek41_metadata import GGUF_ALIGNMENT
from deepseek41_quantize import NativeQuantizer, scale_name
from glm53_manifest import load_safetensors_header
from glm53_quantize import (
    QTYPE_Q8_0, align, qtype_nbytes, read_exact, read_u32, read_u64,
    read_gguf_string, skip_gguf_value,
)

SOURCE_PATTERNS = (
    (re.compile(r"^layers\.(\d+)\.attn\.wo_b\.weight$"),
     lambda layer: f"blk.{layer}.attn_output_b.weight", "attention_wo_b"),
    (re.compile(r"^layers\.(\d+)\.ffn\.shared_experts\.w2\.weight$"),
     lambda layer: f"blk.{layer}.ffn_down_shexp.weight", "shared_w2"),
)


def fail(message):
    raise ValueError(message)


def sha256_file(path, chunk=8 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        while True:
            data = fp.read(chunk)
            if not data:
                return h.hexdigest()
            h.update(data)


def gguf_layout(path):
    with open(path, "rb") as fp:
        if read_exact(fp, 4, "GGUF magic") != b"GGUF":
            fail(f"{path}: not GGUF")
        version = read_u32(fp, "GGUF version")
        if version != 3:
            fail(f"{path}: expected GGUF v3, got {version}")
        tensor_count = read_u64(fp, "GGUF tensor count")
        metadata_count = read_u64(fp, "GGUF metadata count")
        metadata = {}
        for _ in range(metadata_count):
            key = read_gguf_string(fp, "GGUF metadata key")
            kind = read_u32(fp, "GGUF metadata type")
            if key in {"general.architecture", "deepseek41.quantization", "deepseek41.calibration"}:
                if kind != 8:  # GGUF_STRING
                    fail(f"{key}: expected string metadata")
                metadata[key] = read_gguf_string(fp, key)
            else:
                skip_gguf_value(fp, kind)
        tensors = {}
        for _ in range(tensor_count):
            name = read_gguf_string(fp, "tensor name")
            rank = read_u32(fp, f"{name} rank")
            shape = tuple(read_u64(fp, f"{name} dim") for _ in range(rank))
            qtype = read_u32(fp, f"{name} qtype")
            offset = read_u64(fp, f"{name} offset")
            if name in tensors:
                fail(f"duplicate GGUF tensor {name}")
            tensors[name] = {"shape": shape, "qtype": qtype, "offset": offset}
        data_start = align(fp.tell(), GGUF_ALIGNMENT)
    if metadata.get("general.architecture") != "deepseek41":
        fail(f"{path}: expected deepseek41 architecture, got {metadata.get('general.architecture')!r}")
    return data_start, tensors, metadata


class ShardDB:
    """Minimal SourceDB-compatible view over one or more safetensors shards."""
    def __init__(self, paths):
        self.tensors = {}
        self._fds = {}
        for path in paths:
            header = load_safetensors_header(path)
            for name, info in header.items():
                if name in self.tensors:
                    fail(f"duplicate source tensor {name} in {path}")
                self.tensors[name] = dict(info, shard=path)

    def info(self, name):
        if name not in self.tensors:
            fail(f"source tensor not available: {name}")
        return self.tensors[name]

    def read(self, name):
        info = self.info(name)
        path = info["shard"]
        fd = self._fds.get(path)
        if fd is None:
            fd = os.open(path, os.O_RDONLY)
            self._fds[path] = fd
        data = os.pread(fd, info["nbytes"], info["offset"])
        if len(data) != info["nbytes"]:
            fail(f"short source read: {name}")
        return data

    def close(self):
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()


def matching_sources(db):
    found = []
    for source in sorted(db.tensors):
        for pattern, dest_fn, role in SOURCE_PATTERNS:
            match = pattern.match(source)
            if match:
                found.append((source, dest_fn(int(match.group(1))), role))
                break
    return found


def required_scale(db, source):
    info = db.info(source)
    if info["dtype"] in ("I8", "F8_E4M3"):
        return scale_name(source)
    return None


def load_journal(path, gguf):
    if not os.path.exists(path):
        return {"version": 1, "gguf": os.path.realpath(gguf), "patched": {}}
    with open(path) as fp:
        doc = json.load(fp)
    if doc.get("version") != 1 or doc.get("gguf") != os.path.realpath(gguf):
        fail("journal belongs to another GGUF")
    if not isinstance(doc.get("patched"), dict):
        fail("invalid journal")
    return doc


def save_journal(path, doc):
    tmp = path + ".tmp"
    with open(tmp, "w") as fp:
        json.dump(doc, fp, indent=2, sort_keys=True)
        fp.write("\n")
        fp.flush()
        os.fsync(fp.fileno())
    os.replace(tmp, path)


def patch(args):
    data_start, targets, metadata = gguf_layout(args.gguf)
    db = ShardDB(args.shard)
    q = NativeQuantizer(args.quants_library)
    journal_path = args.journal or args.gguf + ".distcog-s3.json"
    journal = load_journal(journal_path, args.gguf)
    matched = matching_sources(db)
    if not matched:
        print("no distcog patch tensors in supplied shard(s)")
        return 0

    # Require quantization scales to be available before modifying the GGUF.
    ready = []
    for source, dest, role in matched:
        scale = required_scale(db, source)
        if scale and scale not in db.tensors:
            print(f"defer {source}: missing {scale}")
            continue
        ready.append((source, dest, role))
    if not ready:
        return 0

    with open(args.gguf, "r+b") as out:
        for source, dest, role in ready:
            if dest not in targets:
                fail(f"target GGUF is missing {dest}")
            target = targets[dest]
            if target["qtype"] != QTYPE_Q8_0:
                fail(f"{dest}: expected Q8_0, got qtype {target['qtype']}")
            source_info = db.info(source)
            expected_shape = tuple(reversed(source_info["shape"]))
            if target["shape"] != expected_shape:
                fail(f"{dest}: shape {target['shape']} != reversed source {expected_shape}")
            expected_bytes = qtype_nbytes(QTYPE_Q8_0, target["shape"])

            source_identity = {
                "source": source,
                "source_dtype": source_info["dtype"],
                "source_shape": source_info["shape"],
                "source_shard": os.path.basename(source_info["shard"]),
                "source_shard_size": os.path.getsize(source_info["shard"]),
                "role": role,
            }
            previous = journal["patched"].get(dest)
            if previous and previous.get("source") == source and not args.force:
                print(f"skip already patched {dest}")
                continue

            values = q.to_f32(db, source)
            payload = q.encode(values, QTYPE_Q8_0)
            if len(payload) != expected_bytes:
                fail(f"{dest}: encoded {len(payload)} bytes, expected {expected_bytes}")
            digest = hashlib.sha256(payload).hexdigest()
            absolute = data_start + target["offset"]
            out.seek(absolute)
            out.write(payload)
            out.flush()
            os.fsync(out.fileno())

            source_identity.update({
                "target": dest,
                "target_offset": absolute,
                "target_bytes": expected_bytes,
                "payload_sha256": digest,
            })
            journal["patched"][dest] = source_identity
            journal["metadata"] = metadata
            save_journal(journal_path, journal)
            print(f"patched {dest} <- {source} ({expected_bytes / (1<<20):.2f} MiB) {digest[:12]}")
    return len(ready)


def audit(args):
    data_start, tensors, metadata = gguf_layout(args.gguf)
    wanted = []
    for layer in range(48):
        for name in (f"blk.{layer}.attn_output_b.weight", f"blk.{layer}.ffn_down_shexp.weight"):
            if name in tensors:
                t = tensors[name]
                wanted.append((name, t))
    bad = [(n,t) for n,t in wanted if t["qtype"] != QTYPE_Q8_0]
    print(json.dumps({
        "gguf": args.gguf,
        "data_start": data_start,
        "metadata": metadata,
        "candidate_targets": len(wanted),
        "q8_targets": len(wanted) - len(bad),
        "bad_targets": [n for n,_ in bad],
        "logical_size": os.path.getsize(args.gguf),
    }, indent=2, sort_keys=True))
    return 1 if bad else 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gguf", required=True)
    p.add_argument("--audit", action="store_true")
    p.add_argument("--shard", action="append", default=[])
    p.add_argument("--journal")
    p.add_argument("--force", action="store_true")
    suffix = "dylib" if sys.platform == "darwin" else "so"
    p.add_argument("--quants-library", default=str(Path(__file__).with_name(f"libds4quants.{suffix}")))
    args = p.parse_args()
    try:
        if args.audit:
            raise SystemExit(audit(args))
        if not args.shard:
            fail("at least one --shard is required unless --audit is used")
        patch(args)
    except (OSError, ValueError) as error:
        sys.exit(f"deepseek41-patch-distcog: {error}")
