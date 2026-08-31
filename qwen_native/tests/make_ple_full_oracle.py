import sys, numpy as np, json, struct, pathlib, mlx.core as mx
D=pathlib.Path(sys.argv[1] if len(sys.argv)>1 else '/Users/mikuru/qwen38fn-official-pipeline/output/Qwen3.8-Flash-Next-OurAblit-Mixed-MLX-Serve')
A={};A.update(mx.load(str(D/'model-00005.safetensors')));A.update(mx.load(str(D/'model-00006.safetensors')))
p='language_model.model.layers.1.ple.'; pe=p+'ple_embedding.'
rng=np.random.default_rng(310831)
h=(rng.standard_normal(10240)*0.07).astype(np.float32)
cs=(rng.standard_normal((10240,9))*0.025).astype(np.float32)
hist=np.array([12345,67890,24680],np.int64)
mult=np.array(A[pe+'layer_multipliers']).astype(np.int64); sizes=np.array(A[pe+'ngram_heads_vocab_sizes']).astype(np.int64); offs=np.array(A[pe+'ngram_heads_offsets']).astype(np.int64)
ids=[]
for hd in range(16):
 m=np.int64(hist[-1]*mult[0]) ^ np.int64(hist[-2]*mult[1])
 if hd>=8:m ^= np.int64(hist[-3]*mult[2])
 ids.append(int(m%sizes[hd]+offs[hd]))
ids=np.array(ids,np.int64)
# table rows q4/group32
F=D/'ngram_table.bin'
with open(F,'rb') as f:
 hl=struct.unpack('<Q',f.read(8))[0];hdr=json.loads(f.read(hl));base=8+hl;wo=hdr['weight']['data_offsets'][0];so=hdr['scales']['data_offsets'][0];bo=hdr['biases']['data_offsets'][0]
 rows=[]
 for rid in ids:
  f.seek(base+wo+int(rid)*80);w=np.frombuffer(f.read(80),'<u4')
  f.seek(base+so+int(rid)*10);su=np.frombuffer(f.read(10),'<u2');sc=(su.astype(np.uint32)<<16).view(np.float32)
  f.seek(base+bo+int(rid)*10);bu=np.frombuffer(f.read(10),'<u2');bi=(bu.astype(np.uint32)<<16).view(np.float32)
  q=np.empty(160,np.float32)
  for j in range(160): q[j]=((int(w[j//8])>>(4*(j&7)))&15)*sc[j//32]+bi[j//32]
  rows.append(q)
emb=np.concatenate(rows)
def qmm(base,x):return np.array(mx.quantized_matmul(mx.array(x),A[base+'.weight'],A[base+'.scales'],A[base+'.biases'],group_size=64,bits=8),np.float32)
def bf(n):return np.array(A[n].astype(mx.float32),np.float32)
def gnorm(x,w):
 z=x.reshape(4,2560); z=z/np.sqrt(np.mean(z*z,-1,keepdims=True)+1e-6); return (z*w.reshape(4,2560)).reshape(-1)
key=gnorm(qmm(p+'key_proj',emb),bf(p+'norm_key.weight'))
val=qmm(p+'value_proj',emb)
query=gnorm(h,bf(p+'norm_query.weight'))
g=(key.reshape(4,2560)*query.reshape(4,2560)).sum(-1)/np.sqrt(2560)
g=np.sign(g)*np.sqrt(np.maximum(np.abs(g),1e-6))
gv=(1/(1+np.exp(-g)))[:,None]*val[None,:]
gvf=gv.astype(np.float32).reshape(-1)
gvn=gnorm(gvf,bf(p+'norm_conv.weight')).reshape(10240)
cw=bf(p+'conv1d.weight').reshape(10240,4)
co=cs[:,0]*cw[:,0]+cs[:,3]*cw[:,1]+cs[:,6]*cw[:,2]+gvn*cw[:,3]
co=co/(1+np.exp(-co))
cs2=np.concatenate([cs[:,1:],gvn[:,None]],axis=1).astype(np.float32)
out=(gvf+co).astype(np.float32)
P=pathlib.Path('/tmp/qn_ple_full');P.mkdir(exist_ok=True)
for n,x in [('hidden',h),('conv_state',cs),('hist',hist),('ids',ids),('embed',emb),('key',key),('value',val),('query',query),('gate',g.astype(np.float32)),('gated',gvf),('conv_norm',gvn),('conv_state_out',cs2),('out',out)]:np.asarray(x).tofile(P/(n+'.bin'))
print('ids',ids.tolist());print('gate',g.tolist());print('out first',out[:6]);print('conv_w_absmax',float(np.abs(cw).max()))
