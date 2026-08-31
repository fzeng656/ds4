import mlx.core as mx, numpy as np, json, pathlib, math, sys, time
if len(sys.argv)!=5:
 print(f'usage: {sys.argv[0]} layer model_dir manifest out_dir',file=sys.stderr); raise SystemExit(2)
L=int(sys.argv[1]); ROOT=pathlib.Path(sys.argv[2]); MAN=json.load(open(sys.argv[3]))['tensors']; OUT=pathlib.Path(sys.argv[4])
pre=f'language_model.model.layers.{L}.'
def meta(s): return MAN[pre+s]
shards=sorted({meta('attn_hyper_connection.hc_norm.weight')['shard'],meta('mlp.switch_mlp.gate_proj.weight')['shard'],meta('self_attn.q_proj.weight')['shard']})
loaded={s:mx.load(str(ROOT/s)) for s in shards}
def A(s):
 k=pre+s; return loaded[MAN[k]['shard']][k]
def qmm(base,z,bits=8,expert=None):
 w=A(base+'.weight'); sc=A(base+'.scales'); bi=A(base+'.biases')
 if expert is not None: w=w[expert]; sc=sc[expert]; bi=bi[expert]
 return mx.quantized_matmul(mx.array(z),w,sc,bi,group_size=64,bits=bits)
def f32(x): return np.array(x.astype(mx.float32),dtype=np.float32)
def norm_np(z,w,group=None):
 z=np.asarray(z,np.float32)
 if group:
  zz=z.reshape(-1,group); zz=zz/np.sqrt(np.mean(zz*zz,-1,keepdims=True)+1e-6); z=zz.reshape(-1)
 else: z=z/np.sqrt(np.mean(z*z,-1,keepdims=True)+1e-6)
 return z*f32(w)
def rope(z,pos):
 z=np.array(z,np.float32,copy=True); inv=1.0/(10000000.0**(np.arange(32,dtype=np.float32)*2/64)); ang=pos*inv;c=np.cos(ang);s=np.sin(ang);aa=z[...,:32].copy();bb=z[...,32:64].copy();z[...,:32]=aa*c-bb*s;z[...,32:64]=bb*c+aa*s;return z
rng=np.random.default_rng(20260831+L); pos=2560; T=pos+1; B=T//4
h=(rng.standard_normal(10240)*0.07).astype(np.float32)
rawhist=(rng.standard_normal((pos,128))*0.12).astype(np.float32)
Khist=(rng.standard_normal((pos,2,256))*0.05).astype(np.float32)
Vhist=(rng.standard_normal((pos,2,256))*0.05).astype(np.float32)
# attention hyper
xn=norm_np(h,A('attn_hyper_connection.hc_norm.weight'),2560)
d=f32(qmm('attn_hyper_connection.input_mix_weight_down',xn))/4; d=d/(1+np.exp(-d)); up=f32(qmm('attn_hyper_connection.input_mix_weight_up',d)); mw=1/(1+np.exp(-up)); mixed=(xn.reshape(4,2560)*mw.reshape(4,2560)).mean(0)
iw=f32(A('attn_hyper_connection.block_inject_weight.weight')); ainj=2/(1+np.exp(-(iw@xn)/4))
# indexer
iraw=f32(qmm('self_attn.indexer.index_qk_proj',mixed)); qidx=iraw[:512].reshape(4,128); qidx=norm_np(qidx,A('self_attn.indexer.q_layernorm.weight')); qidx=rope(qidx,pos)
kw=f32(A('self_attn.indexer.k_layernorm.weight')); pk=np.empty((B,128),np.float32)
for b in range(B):
 z=rawhist[b*4:b*4+4].mean(0,dtype=np.float32); z=z/np.sqrt(np.mean(z*z)+1e-6)*kw; pk[b]=rope(z,b*4)
scores=np.maximum(qidx@pk.T,0).sum(0)/math.sqrt(128); selb=np.argpartition(scores,-512)[-512:]; ids=np.concatenate([np.concatenate([np.arange(int(b)*4,int(b)*4+4,dtype=np.int32) for b in selb]),np.array([pos],np.int32)])
# current qkv
qr=f32(qmm('self_attn.q_proj',mixed)).reshape(24,512); q=norm_np(qr[:,:256],A('self_attn.q_norm.weight')); q=rope(q,pos); gate=qr[:,256:].reshape(-1)
kcur=norm_np(f32(qmm('self_attn.k_proj',mixed)).reshape(2,256),A('self_attn.k_norm.weight')); kcur=rope(kcur,pos); vcur=f32(qmm('self_attn.v_proj',mixed)).reshape(2,256)
K=np.concatenate([Khist,kcur[None]],0); V=np.concatenate([Vhist,vcur[None]],0)
ao=np.empty((24,256),np.float32)
for hh in range(24):
 kh=hh//12; sc=K[ids,kh]@q[hh]/16; sc-=sc.max(); p=np.exp(sc);p/=p.sum();ao[hh]=p@V[ids,kh]
ao=ao.reshape(-1)/(1+np.exp(-gate)); aout=f32(qmm('self_attn.o_proj',ao)); ah=h.reshape(4,2560)+ainj[:,None]*aout[None,:]; ah=ah.reshape(-1).astype(np.float32)
# mlp hyper
mxn=norm_np(ah,A('mlp_hyper_connection.hc_norm.weight'),2560); md=f32(qmm('mlp_hyper_connection.input_mix_weight_down',mxn))/4;md=md/(1+np.exp(-md));mu=f32(qmm('mlp_hyper_connection.input_mix_weight_up',md));mmw=1/(1+np.exp(-mu));mmix=(mxn.reshape(4,2560)*mmw.reshape(4,2560)).mean(0);miw=f32(A('mlp_hyper_connection.block_inject_weight.weight'));minj=2/(1+np.exp(-(miw@mxn)/4))
# router
logits=f32(qmm('mlp.gate',mmix)); vmax=logits.max(); probs=np.exp(logits-vmax);probs/=probs.sum(); rid=np.argpartition(probs,-10)[-10:];rid=rid[np.argsort(probs[rid])[::-1]];rw=probs[rid]
routed=np.zeros(2560,np.float32)
for e,w in zip(rid,rw):
 g=f32(qmm('mlp.switch_mlp.gate_proj',mmix,4,int(e)));u=f32(qmm('mlp.switch_mlp.up_proj',mmix,4,int(e)));hh=(g/(1+np.exp(-g)))*u;dd=f32(qmm('mlp.switch_mlp.down_proj',hh,4,int(e)));routed+=dd*w
sg=f32(qmm('mlp.shared_expert.gate_proj',mmix));su=f32(qmm('mlp.shared_expert.up_proj',mmix));sh=(sg/(1+np.exp(-sg)))*su;shared=f32(qmm('mlp.shared_expert.down_proj',sh));sgate=1/(1+np.exp(-(f32(A('mlp.shared_expert_gate.weight')).reshape(-1)@mmix)));moe=routed+shared*sgate
final=(ah.reshape(4,2560)+minj[:,None]*moe[None,:]).reshape(-1).astype(np.float32)
out=OUT;out.mkdir(parents=True,exist_ok=True)
for n,z in [('hyper',h),('rawk',rawhist),('K',Khist),('V',Vhist),('final',final),('route_ids',rid.astype(np.int32)),('route_weights',rw.astype(np.float32))]: np.asarray(z).tofile(out/(n+'.bin'))
print('L',L,'shards',shards,'route',rid.tolist(),'final0',float(final[0]))
