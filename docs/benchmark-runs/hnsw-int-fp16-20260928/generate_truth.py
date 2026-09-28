import hashlib,json,pathlib,time
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
root=pathlib.Path(__file__).resolve().parent;data=root/'dataset';data.mkdir(exist_ok=True)
source=pathlib.Path('/home/zhenghaoz/vamana-retest-20260928/dataset')
for name in ['shuffle_train.parquet','test.parquet']:
 p=data/name
 if not p.exists():p.symlink_to(source/name)
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read_vectors(p):
 t=pq.read_table(p);ids=t['id'].to_numpy();emb=t['emb'].combine_chunks();vectors=emb.values.to_numpy().reshape(len(ids),768).astype(np.float64)
 assert np.isfinite(vectors).all()
 return ids,vectors
start=time.monotonic();ids,train=read_vectors(data/'shuffle_train.parquet');qids,queries=read_vectors(data/'test.parquet')
assert len(ids)==100000 and len(qids)==1000 and len(set(qids))==1000
order=np.argsort(ids);ids=ids[order];train=train[order];assert np.array_equal(ids,np.arange(100000))
tnorm=np.linalg.norm(train,axis=1);qnorm=np.linalg.norm(queries,axis=1);assert np.all(tnorm>0) and np.all(qnorm>0)
normalized=train/tnorm[:,None];qnormalized=queries/qnorm[:,None]
rates=[.999,.998,.995,.99,.98,.95,.9,.8,.5];thresholds=[int(100000*r) for r in rates]
results={t:np.empty((1000,100),dtype=np.int64) for t in thresholds}
def top100(scores,offset):
 cutoff=np.partition(scores,len(scores)-100)[len(scores)-100]
 candidates=np.flatnonzero(scores>=cutoff)
 return candidates[np.lexsort((candidates,-scores[candidates]))[:100]]+offset
for start_q in range(0,1000,32):
 scores=qnormalized[start_q:start_q+32] @ normalized[50000:].T
 for local,row in enumerate(scores):
  qi=start_q+local
  for threshold in thresholds:results[threshold][qi]=top100(row[threshold-50000:],threshold)
 print('queries',min(start_q+32,1000),flush=True)
# Independent full-coordinate reductions verify the BLAS-selected sets/ranking.
for qi in [0,137,999]:
 raw_scores=np.sum(train[50000:]*queries[qi],axis=1)/(tnorm[50000:]*qnorm[qi])
 for threshold in thresholds:
  expected=np.lexsort((ids[threshold:],-raw_scores[threshold-50000:]))[:100]+threshold
  assert np.array_equal(expected,results[threshold][qi]),(qi,threshold)
# Cross-check the FP32 source/ID interpretation against existing unfiltered truth.
official=pq.read_table(source/'neighbors.parquet').to_pylist();official_by={r['id']:r['neighbors_id'] for r in official}
checks=[]
for qi in [0,137,999]:
 score=np.sum(train*queries[qi],axis=1)/(tnorm*qnorm[qi]);exact=top100(score,0)
 reference=official_by[int(qids[qi])][:100];overlap=len(set(exact.tolist())&set(reference))/100
 assert overlap>=.99,(qi,overlap);checks.append({'query_id':int(qids[qi]),'unfiltered_top100_overlap':overlap})
outputs=[]
for rate,threshold in zip(rates,thresholds):
 pct=rate*100;name=f'neighbors_int_{int(pct)}p.parquet' if 1<=pct<=99 else f'neighbors_int_{pct:.1f}p.parquet'
 neighbors=results[threshold]
 assert np.all(neighbors>=threshold) and np.all(neighbors<100000)
 assert all(len(set(row))==100 for row in neighbors)
 table=pa.table({'id':pa.array(qids,type=pa.int64()),'neighbors_id':pa.array(neighbors.tolist(),type=pa.list_(pa.int64()))})
 path=data/name;assert not path.exists();pq.write_table(table,path,compression='zstd')
 outputs.append({'file':name,'sha256':sha(path),'filter_rate':rate,'filter_expression':f'id >= {threshold}','matching_documents':100000-threshold,'query_count':1000,'neighbors_per_query':100})
meta={'source':'locally generated exact integer-filter truth, not a published VectorDBBench artifact','metric':'cosine','method':'Exhaustive float64 cosine over original FP32 source vectors; descending similarity then ascending integer ID; no ANN/quantization used','numpy_version':np.__version__,'pyarrow_version':pa.__version__,'training_rows':100000,'query_rows':1000,'dimension':768,'source_sha256':{n:sha(data/n) for n in ['shuffle_train.parquet','test.parquet']},'ground_truth':outputs,'independent_full_reduction_query_indices':[0,137,999],'published_unfiltered_cross_checks':checks,'elapsed_seconds':time.monotonic()-start}
(root/'dataset-validation.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta,indent=2),flush=True)
