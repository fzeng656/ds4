#!/usr/bin/env python3
import os, pathlib, subprocess, tempfile, time
import numpy as np

ROOT=pathlib.Path(__file__).resolve().parent
DS4=ROOT.parents[1]/'ds4'
MODEL=pathlib.Path('/Users/mikuru/models/GLM-5.3-Flash-Official-Q2-ds4/GLM-5.3-Flash-Q2.gguf')
HIDDEN=4096
LAYER=22

def read_prompts(p):
    return [x.strip() for x in pathlib.Path(p).read_text().splitlines()
            if x.strip() and not x.lstrip().startswith('#')]

def capture(prompt, idx, kind):
    with tempfile.TemporaryDirectory(prefix=f'glm53-l22-{kind}-{idx:02d}-',
                                     dir='/Users/mikuru/.openclaw/tmp') as td:
        td=pathlib.Path(td)
        pf=td/'prompt.txt'; pf.write_text(prompt)
        prefix=td/'cap'
        env=os.environ.copy()
        env['DS4_METAL_GRAPH_DUMP_PREFIX']=str(prefix)
        env['DS4_METAL_GRAPH_DUMP_NAME']='glm53_attn_hc_collapsed'
        env['DS4_METAL_GRAPH_DUMP_LAYER']=str(LAYER)
        cmd=[str(DS4),'-m',str(MODEL),'--ctx','512','--prompt-file',str(pf),
             '-n','1','--system','You are a helpful assistant.','--nothink']
        cp=None
        for attempt in range(40):
            cp=subprocess.run(cmd,env=env,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,
                              text=True,timeout=180)
            if cp.returncode == 0:
                break
            if "another ds4 process is already running" in cp.stderr:
                time.sleep(0.75)
                continue
            raise RuntimeError(cp.stderr[-4000:])
        if cp is None or cp.returncode:
            raise RuntimeError("could not acquire ds4 process lock after retries")
        files=list(td.glob(f'cap_glm53_attn_hc_collapsed-{LAYER}_pos*.bin'))
        if len(files)!=1:
            raise RuntimeError(f'{kind} {idx}: expected 1 dump, got {files}')
        arr=np.fromfile(files[0],dtype=np.float32)
        if arr.size % HIDDEN:
            raise RuntimeError(f'bad size {arr.size}')
        return arr.reshape(-1,HIDDEN)[-1].copy()

lock=ROOT/'.extract-l22.lock'
try:
    lock.mkdir()
except FileExistsError:
    raise SystemExit('another layer22 extractor is already running')
harm=read_prompts(ROOT/'refusal.txt')
ctrl=read_prompts(ROOT/'control.txt')
if len(harm)!=len(ctrl): raise SystemExit('pair count mismatch')
H=[]; C=[]
for i,(h,c) in enumerate(zip(harm,ctrl),1):
    print(f'pair {i}/{len(harm)}',flush=True)
    H.append(capture(h,i,'harm'))
    C.append(capture(c,i,'ctrl'))
H=np.stack(H); C=np.stack(C)
np.save(ROOT/'l22-harm-activations.npy',H)
np.save(ROOT/'l22-control-activations.npy',C)
raw=H.mean(0)-C.mean(0)
pooled=np.concatenate([H,C],axis=0)
importance=np.mean(np.abs(pooled),axis=0)
mask_n=max(1,round(HIDDEN*0.005))
massive=np.argpartition(importance,-mask_n)[-mask_n:]
masked=raw.copy(); masked[massive]=0
for name,v in [('raw',raw),('mask05',masked)]:
    n=np.linalg.norm(v)
    if not np.isfinite(n) or n<1e-12: raise RuntimeError(name+' invalid norm')
    v=(v/n).astype(np.float32)
    v.tofile(ROOT/f'refusal-l22-{name}.f32')
    for a,b in [(3,28),(3,44)]:
        bank=np.zeros((45,HIDDEN),dtype=np.float32)
        bank[a:b+1]=v
        bank.tofile(ROOT/f'refusal-l22-{name}-L{a:02d}-{b:02d}.f32')
print('mask_n',mask_n)
print('massive_idx',','.join(map(str,sorted(massive.tolist()))))
print('raw_norm_before',float(np.linalg.norm(raw)))
print('masked_norm_before',float(np.linalg.norm(masked)))
print('cos_raw_masked',float(np.dot(raw,masked)/(np.linalg.norm(raw)*np.linalg.norm(masked))))
lock.rmdir()
