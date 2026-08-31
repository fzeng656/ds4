import mlx.core as mx, numpy as np, pathlib, json, sys
L=int(sys.argv[1]); root=pathlib.Path(sys.argv[2]); M=json.load(open(sys.argv[3]))['tensors']; p=f'language_model.model.layers.{L}.'
# load all layer shards
shards={v['shard'] for n,v in M.items() if n.startswith(p)}; A={}
for sh in shards:A.update(mx.load(str(root/sh)))
def qmm(base,x,bits=8):return np.array(mx.quantized_matmul(mx.array(x),A[base+'.weight'],A[base+'.scales'],A[base+'.biases'],group_size=64,bits=bits),np.float32)
def bf(n):return np.array(A[n].astype(mx.float32),np.float32)
# start from existing attention-half oracle inputs/state/output
P=pathlib.Path(f'/tmp/qn_gdn_L{L:02d}'); h=np.fromfile(P/'hyper.bin',np.float32); conv=np.fromfile(P/'conv.bin',np.float32).reshape(10240,4); S=np.fromfile(P/'state.bin',np.float32).reshape(48,128,128)
# attention hyper + gdn
xn=h.reshape(4,2560);xn=xn/np.sqrt(np.mean(xn*xn,-1,keepdims=True)+1e-6)*bf(p+'attn_hyper_connection.hc_norm.weight').reshape(4,2560);xf=xn.reshape(-1);d=qmm(p+'attn_hyper_connection.input_mix_weight_down',xf)/4;d=d/(1+np.exp(-d));mw=1/(1+np.exp(-qmm(p+'attn_hyper_connection.input_mix_weight_up',d)));mixed=(xf.reshape(4,2560)*mw.reshape(4,2560)).mean(0);iw=bf(p+'attn_hyper_connection.block_inject_weight.weight').reshape(4,10240);inj=2/(1+np.exp(-(iw@xf)/4))
qkv=qmm(p+'linear_attn.in_proj_qkv',mixed);z=qmm(p+'linear_attn.in_proj_z',mixed).reshape(48,128);aa=qmm(p+'linear_attn.in_proj_a',mixed);bb=qmm(p+'linear_attn.in_proj_b',mixed);cw=bf(p+'linear_attn.conv1d.weight').reshape(10240,4);cat=np.concatenate([conv,qkv[:,None]],1);cv=(cat[:,1:5]*cw).sum(1);cv=cv/(1+np.exp(-cv));conv2=cat[:,-4:]
q=np.repeat(cv[:2048].reshape(16,128),3,0);k=np.repeat(cv[2048:4096].reshape(16,128),3,0);v=cv[4096:].reshape(48,128);q=q/np.sqrt(np.sum(q*q,-1,keepdims=True)+1e-6);k=k/np.sqrt(np.sum(k*k,-1,keepdims=True)+1e-6);beta=1/(1+np.exp(-bb));g=-np.exp(bf(p+'linear_attn.A_log'))*np.logaddexp(0,aa+bf(p+'linear_attn.dt_bias'));S2=S*np.exp(g)[:,None,None];dv=(v-np.einsum('hkv,hk->hv',S2,k))*beta[:,None];S2+=k[:,:,None]*dv[:,None,:];core=np.einsum('hk,hkv->hv',q/np.sqrt(128),S2);core=core/np.sqrt(np.mean(core*core,-1,keepdims=True)+1e-6)*bf(p+'linear_attn.norm.weight')[None,:];core*=1/(1+np.exp(-z));att=qmm(p+'linear_attn.out_proj',core.reshape(-1));att_h=(h.reshape(4,2560)+inj[:,None]*att[None,:]).astype(np.float32).reshape(-1)
# mlp hyper
xn=att_h.reshape(4,2560);xn=xn/np.sqrt(np.mean(xn*xn,-1,keepdims=True)+1e-6)*bf(p+'mlp_hyper_connection.hc_norm.weight').reshape(4,2560);xf=xn.reshape(-1);d=qmm(p+'mlp_hyper_connection.input_mix_weight_down',xf)/4;d=d/(1+np.exp(-d));mw=1/(1+np.exp(-qmm(p+'mlp_hyper_connection.input_mix_weight_up',d)));mh=(xf.reshape(4,2560)*mw.reshape(4,2560)).mean(0);iw=bf(p+'mlp_hyper_connection.block_inject_weight.weight').reshape(4,10240);minj=2/(1+np.exp(-(iw@xf)/4))
# router
logits=qmm(p+'mlp.gate',mh); mxv=logits.max(); probs=np.exp(logits-mxv);probs/=probs.sum();ids=np.argsort(-probs)[:10].astype(np.int32);weights=probs[ids].astype(np.float32)
routed=np.zeros(2560,np.float32)
for eid,w in zip(ids,weights):
 # slice expert directly then quantized matmul by selecting expert tensor arrays
 def eq(base,x):
  return np.array(mx.quantized_matmul(mx.array(x),A[base+'.weight'][eid],A[base+'.scales'][eid],A[base+'.biases'][eid],group_size=64,bits=4),np.float32)
 gg=eq(p+'mlp.switch_mlp.gate_proj',mh); uu=eq(p+'mlp.switch_mlp.up_proj',mh); hh=(gg/(1+np.exp(-gg)))*uu; routed += eq(p+'mlp.switch_mlp.down_proj',hh)*w
sg=qmm(p+'mlp.shared_expert.gate_proj',mh);su=qmm(p+'mlp.shared_expert.up_proj',mh);sh=(sg/(1+np.exp(-sg)))*su;shared=qmm(p+'mlp.shared_expert.down_proj',sh);shared*=1/(1+np.exp(-(bf(p+'mlp.shared_expert_gate.weight')@mh)))
moe=routed+shared
final=(att_h.reshape(4,2560)+minj[:,None]*moe[None,:]).astype(np.float32).reshape(-1)
for n,x in [('att_half',att_h),('full',final),('ids',ids),('weights',weights),('conv_state_out_full',conv2),('state_out_full',S2)]:np.asarray(x).tofile(P/(n+'.bin'))
print('L',L,'ids',ids.tolist(),'att0',float(att_h[0]),'full0',float(final[0]))
