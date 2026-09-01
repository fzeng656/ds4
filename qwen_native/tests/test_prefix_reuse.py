#!/usr/bin/env python3
import argparse, os, pathlib, subprocess, sys

def readline(p):
    line=p.stdout.readline()
    if not line: raise RuntimeError(f"worker EOF rc={p.poll()}")
    return line.strip()

def gen(p, ids, n):
    p.stdin.write("GEN %d 0 %d %s\n" % (n,len(ids)," ".join(map(str,ids))))
    p.stdin.flush(); begin=readline(p).split()
    if begin[0] != 'BEGIN': raise RuntimeError('expected BEGIN: '+' '.join(begin))
    out=[]
    while True:
        q=readline(p).split()
        if q[0]=='TOK': out.append(int(q[1]))
        elif q[0]=='END': break
        else: raise RuntimeError('unexpected: '+' '.join(q))
    return out, int(begin[3]) if len(begin)>3 else 0

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--worker',required=True); ap.add_argument('--model',required=True); ap.add_argument('--manifest',required=True); a=ap.parse_args()
    ng=str(pathlib.Path(a.model)/'ngram_table.bin'); env=os.environ.copy(); env['QN_RUNTIME_MODE']='stable'
    p=subprocess.Popen([a.worker,a.model,a.manifest,ng],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,bufsize=1,env=env)
    try:
        ready=readline(p); assert ready.startswith('READY stable '),ready
        base=[760,6511,314,9338,369]
        first,_=gen(p,base,2)
        extended=base+first+[760]
        hot,rh=gen(p,extended,1)
        p.stdin.write('RESET\n');p.stdin.flush(); assert readline(p)=='RESET'
        cold,rc=gen(p,extended,1)
        assert hot==cold,(hot,cold)
        assert rh==len(base)+len(first),(rh,len(base)+len(first))
        assert rc==0,rc
        print(f'prefix reuse ok reused={rh} hot={hot} cold={cold}')
        p.stdin.write('QUIT\n');p.stdin.flush(); assert readline(p)=='BYE'
    finally:
        if p.poll() is None: p.kill()
        err=p.stderr.read()
        if err: sys.stderr.write(err)
if __name__=='__main__': main()
