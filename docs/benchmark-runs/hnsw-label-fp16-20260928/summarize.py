import csv, json, pathlib, shutil
root=pathlib.Path(__file__).resolve().parent
repo=pathlib.Path('/home/zhenghaoz/xvec')
archive=repo/'docs/benchmark-runs/hnsw-label-fp16-20260928'
meta=json.loads((root/'metadata.json').read_text())
assert len(meta['runs'])==20 and all(r['exit_code']==0 for r in meta['runs'])
validation=json.loads((root/'dataset-validation.json').read_text())
resources={r['name']:r for r in meta['runs']}
rows=[]
for percentage in [0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5]:
 label=f'label_{percentage*100:g}p'
 for backend in ['xvec','zvec']:
  name=f'{backend}-fp16-label-{percentage*100:g}p';buildname=f'{backend}-fp16-build'
  report=json.loads((root/(name+'.json')).read_text())
  build=json.loads((root/(buildname+'.json')).read_text())['load']
  c=report['config'];s=report['serial'];q=report['concurrent'][0];r=resources[name]
  assert c['backend']==backend and c['index_type']=='hnsw' and c['quantize_type']=='fp16'
  assert c['label_percentage']==percentage and c['filter_expression']==f"labels = '{label}'"
  assert c['m']==50 and c['ef_construction']==500 and c['ef_search']==300 and c['k']==100
  assert s['queries']==1000 and q['concurrency']==8 and c['concurrency_duration']=='30s'
  assert c['serial_cooldown']=='3s' and not c['use_refiner'] and c['enable_mmap']
  assert build['rows']==100000 and 'load' not in report
  row={'machine':meta['machine'],'backend':backend,'backend_version':meta['xvec_revision'] if backend=='xvec' else 'v0.7.0',
   'harness_revision':meta['xvec_revision'],'case':report['case']['name'],'dataset':meta['dataset'],'dimension':768,'metric':'cosine',
   'index_type':'hnsw','quantize_type':'fp16','rotate':'false','use_refiner':'false','enable_mmap':'true',
   'm':50,'ef_construction':500,'ef_search':300,'k':100,'label_percentage':percentage,
   'matching_documents':validation['label_counts'][label],'filter_expression':c['filter_expression'],
   'batch_size':c['batch_size'],'max_docs_per_segment':c['max_docs_per_segment'],'optimize_concurrency':8,'query_concurrency':8,
   'concurrency_duration_sec':30,'serial_cooldown_sec':3,'warmup_queries':100,'payload_profile':'ids_only',
   'gomaxprocs':8,'gomemlimit':'24GiB','cpu_affinity':'0-7','go_version':report['system']['go_version'],
   'inserted_count':build['rows'],'insert_duration_sec':build['insert_duration_sec'],'optimize_duration_sec':build['optimize_duration_sec'],
   'load_duration_sec':build['load_duration_sec'],'build_peak_rss_mib':resources[buildname]['peak_rss_mib'],
   'serial_queries':s['queries'],'serial_qps':s['qps'],'recall_at_k_pct':s['recall']*100}
  for prefix,metrics in [('serial',s),('concurrent',q)]:
   for key in ['queries','qps','latency_avg_ms','latency_p95_ms','latency_p99_ms']:row[prefix+'_'+key]=metrics[key]
  for key in ['peak_rss_mib','wall_seconds','user_cpu_seconds','system_cpu_seconds']:row['query_'+key]=r[key]
  rows.append(row)
archive.mkdir(parents=True,exist_ok=True)
for r in meta['runs']:
 for suffix in ['.json','.resources.json','.log']:
  shutil.copy2(root/(r['name']+suffix),archive/(r['name']+suffix))
for name in ['build-info.txt','metadata.json','downloads.json','dataset-validation.json','native-validation.json','run_fp16.py','summarize.py','inspect_labels.go','check_native.py']:
 shutil.copy2(root/name,archive/name)
shutil.copy2(root/'native/zvec-libs-linux-x64.tar.gz.sha256',archive/'native-archive.sha256')
with (repo/'docs/benchmark-hnsw-label-filter.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader()
 for row in rows:writer.writerow({k:(f'{v:.6f}'.rstrip('0').rstrip('.') if isinstance(v,float) else v) for k,v in row.items()})
print('| Matching labels | xvec QPS | zvec QPS | zvec/xvec | xvec recall (%) | zvec recall (%) | xvec P99 (ms) | zvec P99 (ms) |')
print('| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |')
for x,z in zip(rows[::2],rows[1::2]):
 print(f"| {x['label_percentage']*100:g}% | {x['concurrent_qps']:.2f} | {z['concurrent_qps']:.2f} | {z['concurrent_qps']/x['concurrent_qps']:.2f}× | {x['recall_at_k_pct']:.3f} | {z['recall_at_k_pct']:.3f} | {x['concurrent_latency_p99_ms']:.3f} | {z['concurrent_latency_p99_ms']:.3f} |")
print('\n| Backend | Insert (s) | Optimize (s) | Total load (s) | Build peak RSS (MiB) |')
print('| --- | ---: | ---: | ---: | ---: |')
for row in rows[:2]:print(f"| {row['backend']} | {row['insert_duration_sec']:.2f} | {row['optimize_duration_sec']:.2f} | {row['load_duration_sec']:.2f} | {row['build_peak_rss_mib']:.2f} |")
