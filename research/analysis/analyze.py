"""Post-hoc diagnosis only. No training or test-set model inference."""
import os,sys,json,csv,gc,math,time
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,average_precision_score,log_loss,f1_score
R=Path('/home/edo/hint-attention');O=R/'diagnosis';O.mkdir(exist_ok=True)
os.chdir(R/'runs/conditioned_s0/runtime')
sys.path[:0]=[str(R/'upstream'),str(R),str(R/'final_conditioned')]
from HINT.dataloader import csv_three_feature_2_dataloader,sentence2vec
from HINT.protocol_encode import split_protocol
from hint_attention import install_attention
from hint_conditioned import install_conditioned
from torch.nn import functional as F
torch.set_num_threads(2)
arms={'original':R/'runs/continuation_s0/original_s0','attention':R/'runs/continuation_s0/attention_s0','conditioned':R/'runs/conditioned_s0'}
def read(p):
 with p.open() as f:return list(csv.DictReader(f))
def stats(a):
 a=np.asarray(a,dtype=float)
 return {'n':len(a),'mean':float(a.mean()),'median':float(np.median(a)),'p10':float(np.quantile(a,.1)),'p90':float(np.quantile(a,.9)),'max':float(a.max())}
def metrics(y,p):return {'roc_auc':float(roc_auc_score(y,p)),'ap':float(average_precision_score(y,p)),'bce':float(log_loss(y,p)),'f1':float(f1_score(y,np.asarray(p)>=.5))}
out={'scope':'Post-hoc exploratory diagnosis; no tuning; ablations on validation only','prediction_comparisons':{},'parser':{},'attention':{}}
for split in ['valid','test']:
 rows=read(arms['original']/f'predictions_{split}.csv');ids=[r['nctid'] for r in rows];y=np.array([int(r['label']) for r in rows]);p=np.array([float(r['probability']) for r in rows]);out['prediction_comparisons'][split]={'original':metrics(y,p)}
 for arm in ['attention','conditioned']:
  rm={r['nctid']:r for r in read(arms[arm]/f'predictions_{split}.csv')};assert len(rm)==len(ids) and all(int(rm[i]['label'])==yi for i,yi in zip(ids,y));q=np.array([float(rm[i]['probability']) for i in ids]);d=q-p;cp=p>=.5;cq=q>=.5
  v={'metrics':metrics(y,q),'pearson':float(np.corrcoef(p,q)[0,1]),'mean_abs_probability_change':float(abs(d).mean()),'mean_probability_change':float(d.mean()),'max_abs_probability_change':float(abs(d).max()),'classification_flips':int(sum(cp!=cq)),'wrong_to_right':int(sum((cp!=y)&(cq==y))),'right_to_wrong':int(sum((cp==y)&(cq!=y))),'mean_change_positive':float(d[y==1].mean()),'mean_change_negative':float(d[y==0].mean()),'positive_predictions':int(sum(cq)),'original_positive_predictions':int(sum(cp))}
  if split=='test':
   rng=np.random.default_rng(20261011);pos=np.flatnonzero(y==1);neg=np.flatnonzero(y==0);deltas=[]
   for _ in range(1000):
    ix=np.r_[rng.choice(pos,len(pos),replace=True),rng.choice(neg,len(neg),replace=True)];deltas.append([roc_auc_score(y[ix],q[ix])-roc_auc_score(y[ix],p[ix]),average_precision_score(y[ix],q[ix])-average_precision_score(y[ix],p[ix])])
   v['paired_stratified_bootstrap_1000_95pct']={name:np.quantile(np.array(deltas)[:,k],[.025,.975]).tolist() for k,name in enumerate(['auc_delta','ap_delta'])}
  out['prediction_comparisons'][split][arm]=v
print('SAVED PREDICTIONS DONE',flush=True)
for split in ['train','valid','test']:
 rows=read(R/f'assets/phase_II_{split}.csv');counts=[];single=0;missing=0;total=0;empty=0
 for row in rows:
  groups=split_protocol(row['criteria']);single+=len(groups)==1;empty+=not groups[0];counts.append(len(groups[0])+(len(groups[1]) if len(groups)>1 else 0))
  for group in groups:
   for line in group:total+=1;missing+=line not in sentence2vec
 out['parser'][split]={'trials':len(rows),'unsplit_criteria_duplicated_into_both_branches':single,'empty_criteria':empty,'missing_cache_lines':missing,'total_parsed_lines':total,'line_count':stats(counts)}
print('PARSER DONE',flush=True)
valid_loader=csv_three_feature_2_dataloader(str(R/'assets/phase_II_valid.csv'),False,32)
@torch.no_grad()
def predict(model):
 ys=[];ps=[];ids=[]
 for n,y,sm,co,cr in valid_loader:
  p=torch.sigmoid(model(sm,co,cr).reshape(-1));ys+=y.tolist();ps+=p.tolist();ids+=n
 return np.array(ys),np.array(ps),ids
for arm in ['attention','conditioned']:
 print('MODEL',arm,flush=True)
 model=torch.load(R/'assets/phase_II.ckpt',map_location='cpu',weights_only=False)
 (install_attention if arm=='attention' else install_conditioned)(model)
 ck=torch.load(arms[arm]/'best_state.pt',map_location='cpu',weights_only=True);model.load_state_dict(ck['state_dict'],strict=True);model.to('cpu');model.adj=model.adj.cpu()
 for module in model.modules():
  if hasattr(module,'device'):module.device=torch.device('cpu')
 model.eval();enc=model.protocol_encoder;record={'inclusion':[],'exclusion':[]};cosines=[];changes=[];zero_outputs=0;pooledcos=[];querynorm=[];qchanges=[];weight_context_changes=[]
 saved_forward=enc.forward
 @torch.no_grad()
 def diagnosed(groups,*context):
  global zero_outputs
  output,weights=enc.forward_with_attention(groups,*context)
  means=torch.cat([torch.cat([a.mean(0,keepdim=True),b.mean(0,keepdim=True)],dim=1) for a,b in groups]);uniform=enc.f(enc.fc(means));cosines.extend(F.cosine_similarity(output,uniform).tolist());changes.extend(((output-uniform).norm(dim=1)/(uniform.norm(dim=1)+1e-8)).tolist());zero_outputs+=int((output.norm(dim=1)==0).sum())
  if context:
   c=context[0];norm=enc.context_norm(c);qi=enc.inclusion_query(norm);qe=enc.exclusion_query(norm);querynorm.extend(qi.norm(dim=1).tolist());qchanges.extend((qi-qi.roll(1,0)).norm(dim=1).tolist())
   _,swapped=enc.forward_with_attention(groups,c.roll(1,0))
  for row,(features,ws) in enumerate(zip(groups,weights)):
   for idx,key in enumerate(['inclusion','exclusion']):
    w=ws[key];v=features[idx];n=len(w);entropy=float(-(w*w.clamp_min(1e-30).log()).sum())
    record[key].append({'n':n,'entropy_normalized':entropy/math.log(n) if n>1 else 1.,'effective_fraction':float(1/(w.square().sum()*n)),'top_vs_uniform':float(w.max()*n),'l1_from_uniform':float(abs(w-1/n).sum()),'all_zero_input':bool((v==0).all())})
    pooled=(w[:,None]*v).sum(0);pooledcos.append(float(F.cosine_similarity(pooled[None],v.mean(0,keepdim=True))))
    if context:weight_context_changes.append(float(abs(w-swapped[row][key]).sum()))
  return output
 enc.forward=diagnosed
 y,p,ids=predict(model);saved={r['nctid']:float(r['probability']) for r in read(arms[arm]/'predictions_valid.csv')}
 report={'normal_validation':metrics(y,p),'max_abs_vs_saved_cuda_validation':float(max(abs(p-np.array([saved[i] for i in ids])))),'protocol_vs_uniform_cosine':stats(cosines),'protocol_relative_l2_change':stats(changes),'pooled_vector_vs_mean_cosine':stats(pooledcos),'zero_protocol_outputs':zero_outputs,'groups':{}}
 for key,values in record.items():
  mult=[v for v in values if v['n']>1];report['groups'][key]={'total':len(values),'single_sentence':sum(v['n']==1 for v in values),'all_zero_input':sum(v['all_zero_input'] for v in values),'multi_sentence':{k:stats([v[k] for v in mult]) for k in ['entropy_normalized','effective_fraction','top_vs_uniform','l1_from_uniform']}}
 # Inference-only counterfactual: retain this arm's learned downstream weights and restore uniform pooling.
 def uniform_forward(groups,*context):
  means=torch.cat([torch.cat([a.mean(0,keepdim=True),b.mean(0,keepdim=True)],dim=1) for a,b in groups]);return enc.f(enc.fc(means))
 enc.forward=uniform_forward
 _,pu,_=predict(model);report['uniform_pooling_validation']=metrics(y,pu);report['normal_vs_uniform_probability_mae']=float(abs(p-pu).mean())
 if arm=='conditioned':
  def swapped_forward(groups,context):return saved_forward(groups,context.roll(1,0))
  enc.forward=swapped_forward;_,ps,_=predict(model);report['rolled_context_validation']=metrics(y,ps);report['normal_vs_rolled_context_probability_mae']=float(abs(p-ps).mean());report['query_norm']=stats(querynorm);report['query_difference_rolled_context']=stats(qchanges);report['weight_l1_change_rolled_context']=stats(weight_context_changes)
 out['attention'][arm]=report
 (O/'diagnosis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({arm:report}),flush=True)
 del model,enc,ck;gc.collect()
(O/'diagnosis.json').write_text(json.dumps(out,indent=2)+'\n')
print('DONE',flush=True)
