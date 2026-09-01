#!/usr/bin/env python3
import argparse, os, pathlib, subprocess, sys
from transformers import AutoTokenizer

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
SRC = REPO / "qwen_native"
DEFAULT_BIN = pathlib.Path("/tmp/qn_generate_ids")

def build(binary: pathlib.Path):
    sources = [
        SRC / "tools/qn_generate_ids.m", SRC / "qn_runtime.m", SRC / "qn_manifest.m",
        SRC / "qn_model_io.m", SRC / "qn_gdn_layer.m", SRC / "qn_ple_layer.m",
        SRC / "qn_qwen4_layer.m", SRC / "qn_qwen4_model.m",
    ]
    newest = max(p.stat().st_mtime for p in sources)
    if binary.exists() and binary.stat().st_mtime >= newest:
        return
    cmd = ["clang", "-O3", "-ffast-math", "-fobjc-arc", f"-I{REPO}"] + [str(p) for p in sources] + ["-framework", "Foundation", "-framework", "Metal", "-framework", "MetalPerformanceShaders", "-o", str(binary)]
    subprocess.run(cmd, check=True)

def main():
    ap = argparse.ArgumentParser(description="Text wrapper for the experimental Qwen4 native runtime")
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--manifest", default=str(REPO / "phase0/qwen38fn_manifest.json"))
    ap.add_argument("--binary", default=str(DEFAULT_BIN))
    ap.add_argument("--production", action="store_true", help="use fail-closed stable runtime prepare and verified prefill backends")
    args = ap.parse_args()
    model = pathlib.Path(args.model).expanduser().resolve()
    binary = pathlib.Path(args.binary).expanduser().resolve()
    build(binary)
    tok = AutoTokenizer.from_pretrained(str(model), trust_remote_code=True)
    ids = tok.encode(args.prompt, add_special_tokens=False)
    if not ids:
        raise SystemExit("prompt tokenized to zero tokens")
    cmd = [str(binary), str(model), args.manifest, str(model / "ngram_table.bin"), str(args.max_tokens)] + [str(i) for i in ids]
    env = os.environ.copy()
    if args.production:
        env["QN_RUNTIME_MODE"] = "stable"
    cp = subprocess.run(cmd, text=True, capture_output=True, env=env)
    if cp.stderr:
        print(cp.stderr, file=sys.stderr, end="")
    if cp.returncode:
        raise SystemExit(cp.returncode)
    line = next((x for x in cp.stdout.splitlines() if x.startswith("TOKENS ")), None)
    if not line:
        raise SystemExit("native runner did not return TOKENS")
    out_ids = [int(x) for x in line.split()[1:]]
    print(f"prompt_tokens={len(ids)} ids={ids}")
    print(f"generated_ids={out_ids}")
    print(tok.decode(out_ids, skip_special_tokens=False), end="")

if __name__ == "__main__":
    main()
