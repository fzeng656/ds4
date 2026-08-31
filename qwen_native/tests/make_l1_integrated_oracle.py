# Compose the independently validated PLE boundary with a full L1 GDN oracle.
import sys, numpy as np, pathlib, json, mlx.core as mx
D=pathlib.Path(sys.argv[1] if len(sys.argv)>1 else '/Users/mikuru/qwen38fn-official-pipeline/output/Qwen3.8-Flash-Next-OurAblit-Mixed-MLX-Serve')
M=json.load(open('/Users/mikuru/qwen-native-runtime/phase0/qwen38fn_manifest.json'))['tensors'];L=1;p='language_model.model.layers.1.'
# Reuse exact corrected PLE oracle artifacts.
P0=pathlib.Path('/tmp/qn_ple_full'); pre=np.fromfile(P0/'hidden.bin',np.float32); ple=np.fromfile(P0/'out.bin',np.float32); post=(pre+ple).astype(np.float32)
rng=np.random.default_rng(910831); conv=(rng.standard_normal((10240,4))*.03).astype(np.float32); S=(rng.standard_normal((48,128,128))*.005).astype(np.float32)
shards={v['shard'] for n,v in M.items() if n.startswith(p) and '.ple.' not in n};A={}
for sh in shards:A.update(mx.load(str(D/sh)))
def qmm(base,x,bits=8):return np.array(mx.quantized_matmul(mx.array(x),A[base+'.weight'],A[base+'.scales'],A[base+'.biases'],group_size=64,bits=bits),np.float32)
def qmme(base,e,x):return np.array(mx.quantized_matmul(mx.array(x),A[base+'.weight'][e],A[base+'.scales'][e],A[base+'.biases'][e],group_size=64,bits=4),np.float32)
def bf(n):return np.array(A[n].astype(mx.float32),np.float32)
def hyper(base,h):
 z=h.reshape(4,2560);z=z/np.sqrt(np.mean(z*z,-1,keepdims=True)+1e-6)*bf(base+'hc_norm.weight').reshape(4,2560);zf=z.reshape(-1);d=qmm(base+'input_mix_weight_down',zf)/4;d=d/(1+np.exp(-d));mw=1/(1+np.exp(-qmm(base+'input_mix_weight_up',d)));mixed=(zf.reshape(4,2560)*mw.reshape(4,2560)).mean(0);iw=bf(base+'block_inject_weight.weight').reshape(4,10240);inj=2/(1+np.exp(-(iw@zf)/4));return mixed,inj
mixed,inj=hyper(p+'attn_hyper_connection.',post);qkv=qmm(p+'linear_attn.in_proj_qkv',mixed);z=qmm(p+'linear_attn.in_proj_z',mixed).reshape(48,128);aa=qmm(p+'linear_attn.in_proj_a',mixed);bb=qmm(p+'linear_attn.in_proj_b',mixed);cw=bf(p+'linear_attn.conv1d.weight').reshape(10240,4);cat=np.concatenate([conv,qkv[:,None]],1);cv=(cat[:,1:5]*cw).sum(1);cv=cv/(1+np.exp(-cv));conv2=cat[:,-4:]
q=np.repeat(cv[:2048].reshape(16,128),3,0);k=np.repeat(cv[2048:4096].reshape(16,128),3,0);v=cv[4096:].reshape(48,128);q=q/np.sqrt(np.sum(q*q,-1,keepdims=True)+1e-6);k=k/np.sqrt(np.sum(k*k,-1,keepdims=True)+1e-6);beta=1/(1+np.exp(-bb));g=-np.exp(bf(p+'linear_attn.A_log'))*np.logaddexp(0,aa+bf(p+'linear_attn.dt_bias'));S2=S*np.exp(g)[:,None,None];dv=(v-np.einsum('hkv,hk->hv',S2,k))*beta[:,None];S2+=k[:,:,None]*dv[:,None,:];core=np.einsum('hk,hkv->hv',q/np.sqrt(128),S2);core=core/np.sqrt(np.mean(core*core,-1,keepdims=True)+1e-6)*bf(p+'linear_attn.norm.weight')[None,:];core*=1/(1+np.exp(-z));att=qmm(p+'linear_attn.out_proj',core.reshape(-1));half=(post.reshape(4,2560)+inj[:,None]*att[None,:]).reshape(-1).astype(np.float32)
mm,inj2=hyper(p+'mlp_hyper_connection.',half);logits=qmm(p+'mlp.gate',mm);pr=np.exp(logits-logits.max());pr/=pr.sum();ids=np.argsort(-pr)[:10].astype(np.int32);routed=np.zeros(2560,np.float32)
for e in ids:
 ga=qmme(p+'mlp.switch_mlp.gate_proj',int(e),mm);up=qmme(p+'mlp.switch_mlp.up_proj',int(e),mm);hh=ga/(1+np.exp(-ga))*up;routed+=qmme(p+'mlp.switch_mlp.down_proj',int(e),hh)*pr[e]
ga=qmm(p+'mlp.shared_expert.gate_proj',mm);up=qmm(p+'mlp.shared_expert.up_proj',mm);hh=ga/(1+np.exp(-ga))*up;shared=qmm(p+'mlp.shared_expert.down_proj',hh);sg=1/(1+np.exp(-(bf(p+'mlp.shared_expert_gate.weight')@mm)));moe=routed+shared*sg;final=(half.reshape(4,2560)+inj2[:,None]*moe[None,:]).reshape(-1).astype(np.float32)
O=pathlib.Path('/tmp/qn_l1_integrated');O.mkdir(exist_ok=True)
for n,x in [('pre',pre),('ple',ple),('post',post),('gdn_conv',conv),('gdn_state',S),('gdn_conv_out',conv2),('gdn_state_out',S2),('ids',ids),('final',final)]:np.asarray(x).tofile(O/(n+'.bin'))
print('ids',ids.tolist(),'post0',float(post[0]),'final0',float(final[0]))
