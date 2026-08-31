#!/usr/bin/env python3
"""Build a metadata-only manifest for Qwen4-Exp safetensors shards.

Reads only config/index JSON plus each safetensors JSON header. Tensor payloads are
never read, so this is safe to run without materializing model weights in RAM.
"""
from __future__ import annotations
import argparse, json, os, struct
from pathlib import Path


def read_header(path: Path):
    with path.open('rb') as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError(f'{path}: truncated safetensors header length')
        hlen = struct.unpack('<Q', raw)[0]
        header = json.loads(f.read(hlen))
    return hlen, header


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model_dir', type=Path)
    ap.add_argument('-o', '--output', type=Path, required=True)
    args = ap.parse_args()
    model = args.model_dir.resolve()
    cfg = json.load((model / 'config.json').open())
    text = cfg.get('text_config', cfg)
    index = json.load((model / 'model.safetensors.index.json').open())
    weight_map = index['weight_map']
    tensors = {}
    shards = {}
    for shard_name in sorted(set(weight_map.values())):
        p = model / shard_name
        hlen, header = read_header(p)
        data_base = 8 + hlen
        st = p.stat()
        shards[shard_name] = {
            'file_size': st.st_size,
            'header_bytes': hlen,
            'data_base': data_base,
        }
        for name, owner in weight_map.items():
            if owner != shard_name:
                continue
            ent = header.get(name)
            if ent is None:
                raise ValueError(f'{name}: missing from {shard_name} header')
            lo, hi = ent['data_offsets']
            tensors[name] = {
                'shard': shard_name,
                'dtype': ent['dtype'],
                'shape': ent['shape'],
                'payload_offset': lo,
                'payload_bytes': hi - lo,
                'file_offset': data_base + lo,
                'file_end': data_base + hi,
            }
    missing = set(weight_map) - set(tensors)
    if missing:
        raise ValueError(f'missing {len(missing)} indexed tensors')
    layer_types = text.get('layer_types', [])
    result = {
        'format': 'qwen-native-safetensors-manifest-v1',
        'model_dir': str(model),
        'architecture': cfg.get('architectures'),
        'model_type': cfg.get('model_type'),
        'text_model_type': text.get('model_type'),
        'shape': {
            'hidden_size': text.get('hidden_size'),
            'num_hidden_layers': text.get('num_hidden_layers'),
            'num_attention_heads': text.get('num_attention_heads'),
            'num_key_value_heads': text.get('num_key_value_heads'),
            'head_dim': text.get('head_dim'),
            'num_experts': text.get('num_experts'),
            'num_experts_per_tok': text.get('num_experts_per_tok'),
            'moe_intermediate_size': text.get('moe_intermediate_size'),
            'shared_expert_intermediate_size': text.get('shared_expert_intermediate_size'),
            'max_position_embeddings': text.get('max_position_embeddings'),
        },
        'layer_types': layer_types,
        'quantization': cfg.get('quantization_config') or cfg.get('quantization'),
        'shards': shards,
        'tensor_count': len(tensors),
        'tensors': tensors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w') as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write('\n')
    print(f'wrote {args.output}: {len(shards)} shards, {len(tensors)} tensors')

if __name__ == '__main__':
    main()
