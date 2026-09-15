#!/usr/bin/env python3
"""Patch a DS4 DeepSeek V4.1 GGUF from distcog using HTTP byte ranges only."""
import argparse, json, os, struct, sys
from pathlib import Path
import httpx
from huggingface_hub import get_token, hf_hub_download, hf_hub_url
from deepseek41_patch_distcog import patch as patch_shards
from deepseek41_quantize import scale_name
DEFAULT_REPO = "distributedcognition/DeepSeek-V4.1-Flash-abliterated"
def die(msg): raise SystemExit(msg)
def remote_header(client, repo, filename, token):
    h={"Authorization":f"Bearer {token}","Range":"bytes=0-7"}
    r=client.get(hf_hub_url(repo,filename),headers=h); r.raise_for_status()
    if r.status_code!=206 or len(r.content)!=8: raise RuntimeError(f"{filename}: no header range")
    n=struct.unpack('<Q',r.content)[0]; resolved=str(r.url)
    r=client.get(resolved,headers={"Range":f"bytes=8-{7+n}"}); r.raise_for_status()
    if r.status_code!=206 or len(r.content)!=n: raise RuntimeError(f"{filename}: short header")
    return n,json.loads(r.content),resolved
def tensor_bytes(client,resolved,header_len,entry,label):
    start,end=entry['data_offsets']; a=8+header_len+start; b=8+header_len+end-1
    r=client.get(resolved,headers={"Range":f"bytes={a}-{b}"}); r.raise_for_status()
    expected=end-start
    if r.status_code!=206 or len(r.content)!=expected: raise RuntimeError(f"{label}: short range {len(r.content)} != {expected}")
    return r.content
def write_mini(path,tensors):
    cursor=0; doc={}; payloads=[]
    for name,entry,payload in tensors:
        doc[name]={"dtype":entry['dtype'],"shape":entry['shape'],"data_offsets":[cursor,cursor+len(payload)]}
        cursor+=len(payload); payloads.append(payload)
    raw=json.dumps(doc,separators=(',',':'),sort_keys=True).encode(); raw+=b' '*((8-len(raw)%8)%8)
    with open(path,'wb') as f:
        f.write(struct.pack('<Q',len(raw))); f.write(raw)
        for p in payloads: f.write(p)
        f.flush(); os.fsync(f.fileno())
def load_patched(journal):
    try:
        d=json.loads(Path(journal).read_text()); p=d.get('patched',{}); return set(p) if isinstance(p,dict) else set()
    except FileNotFoundError: return set()
def main(args):
    token=get_token()
    if not token: die('No Hugging Face token configured')
    work=Path(args.work_dir).resolve(); work.mkdir(parents=True,exist_ok=True); meta=work/'meta'; meta.mkdir(exist_ok=True)
    index_path=hf_hub_download(args.repo,'model.safetensors.index.json',local_dir=str(meta))
    wm=json.loads(Path(index_path).read_text())['weight_map']; journal=args.journal or args.gguf+'.distcog-s3.json'
    layers=[]
    for layer in range(40):
        pairs=[(f'layers.{layer}.attn.wo_b.weight',f'blk.{layer}.attn_output_b.weight'),(f'layers.{layer}.ffn.shared_experts.w2.weight',f'blk.{layer}.ffn_down_shexp.weight')]
        for s,_ in pairs:
            if s not in wm or scale_name(s) not in wm: die(f'index missing {s} or scale')
        layers.append(pairs)
    timeout=httpx.Timeout(connect=30,read=180,write=30,pool=30)
    downloaded=0
    with httpx.Client(follow_redirects=True,timeout=timeout,limits=httpx.Limits(max_connections=8,max_keepalive_connections=4)) as c:
        for layer,pairs in enumerate(layers):
            patched=load_patched(journal); todo=[(s,d) for s,d in pairs if d not in patched]
            if not todo:
                print(f'[{layer+1}/40] layer {layer}: already patched',flush=True); continue
            by={}
            for s,_ in todo:
                for name in (s,scale_name(s)): by.setdefault(wm[name],[]).append(name)
            minis=[]
            try:
                for shard,names in sorted(by.items()):
                    hn,header,resolved=remote_header(c,args.repo,shard,token); selected=[]; seen=set()
                    for name in names:
                        if name in seen: continue
                        seen.add(name); entry=header.get(name)
                        if not isinstance(entry,dict): raise RuntimeError(f'{shard}: missing {name}')
                        payload=tensor_bytes(c,resolved,hn,entry,name); downloaded+=len(payload); selected.append((name,entry,payload))
                    mini=work/f'layer-{layer:02d}-{Path(shard).stem}-mini.safetensors'; write_mini(mini,selected); minis.append(str(mini))
                    print(f'[{layer+1}/40] fetched {shard}: {sum(len(x[2]) for x in selected)/(1<<20):.2f} MiB',flush=True)
                class A:
                    gguf=args.gguf; shard=minis; journal_path=journal; force=False; quants_library=args.quants_library
                # patcher expects attribute named journal
                setattr(A,'journal',journal)
                patch_shards(A())
            finally:
                if not args.keep_minis:
                    for p in minis:
                        try: os.remove(p)
                        except FileNotFoundError: pass
    patched=load_patched(journal); expected={d for pairs in layers for _,d in pairs}; missing=sorted(expected-patched)
    if missing: die(f'patch incomplete: {len(patched & expected)}/80; first missing {missing[0]}')
    print(f'PASS: patched 80/80; downloaded {downloaded/(1<<30):.3f} GiB tensor payload',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--gguf',required=True); p.add_argument('--repo',default=DEFAULT_REPO); p.add_argument('--work-dir',default='distcog-range'); p.add_argument('--journal'); p.add_argument('--keep-minis',action='store_true'); p.add_argument('--quants-library',default=str(Path(__file__).with_name('libds4quants.dylib'))); main(p.parse_args())
