import csv,json,pathlib,shutil
root=pathlib.Path(__file__).resolve().parent;repo=pathlib.Path('/home/zhenghaoz/xvec');archive=repo/'docs/benchmark-runs/hnsw-label-50-opt-20260928'
meta=json.loads((root/'metadata.json').read_text());assert len(meta['runs'])==18 and all(r['exit_code']==0 for r in meta['runs'])
resources={r['name']:r for r in meta['runs']}
with (archive/'previous.csv').open() as f:
 reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
assert 'build_backend_version' in fields
old={(r['backend'],r['label_percentage']):dict(r) for r in rows}
for row in rows:
 backend=row['backend'];rate=float(row['label_percentage']);name=f'{backend}-fp16-label-{100*rate:g}p'
 report=json.loads((root/(name+'.json')).read_text());c=report['config'];s=report['serial'];q=report['concurrent'][0]
 assert c['backend']==backend and c['label_percentage']==rate and c['quantize_type']=='fp16' and 'load' not in report
 assert c['m']==50 and c['ef_construction']==500 and c['ef_search']==300 and c['k']==100
 assert s['queries']==1000 and q['concurrency']==8 and c['concurrency_duration']=='30s' and c['serial_cooldown']=='3s'
 assert c['filter_expression']==row['filter_expression'] and c['enable_mmap'] and not c['use_refiner']
 row['backend_version']=meta['backend_version'] if backend=='xvec' else 'v0.7.0'
 row['recall_at_k_pct']=s['recall']*100
 for prefix,metrics in [('serial',s),('concurrent',q)]:
  for key in ['queries','qps','latency_avg_ms','latency_p95_ms','latency_p99_ms']:row[prefix+'_'+key]=metrics[key]
 for key in ['peak_rss_mib','wall_seconds','user_cpu_seconds','system_cpu_seconds']:row['query_'+key]=resources[name][key]
 for suffix in ['.json','.log','.resources.json']:shutil.copy2(root/(name+suffix),archive/(name+suffix))
with (repo/'docs/benchmark-hnsw-label-filter.csv').open('w') as f:
 writer=csv.DictWriter(f,fields,lineterminator='\n');writer.writeheader()
 for row in rows:writer.writerow({k:(f'{v:.6f}'.rstrip('0').rstrip('.') if isinstance(v,float) else v) for k,v in row.items()})
for name in ['metadata.json','source.patch','run.py','update_csv.py','tests-full.log','tests-noasm.log','build-info.txt','profile-before-top.txt','profile-after-top.txt','before.cpu','step2.cpu','before-profile.json','before-profile.log','step2-profile.json','step2-profile.log','profile-command.json']:
 shutil.copy2(root/name,archive/name)
shutil.copy2(root/'profile_init.go',archive/'profile_init.go.txt')
print('| Matching labels | xvec before QPS | xvec after QPS | Speedup | zvec rerun QPS | xvec recall (%) | zvec recall (%) |')
print('| --- | ---: | ---: | ---: | ---: | ---: | ---: |')
for x,z in zip(rows[::2],rows[1::2]):
 before=old[('xvec',x['label_percentage'])]
 print(f"| {100*float(x['label_percentage']):g}% | {float(before['concurrent_qps']):.2f} | {x['concurrent_qps']:.2f} | {x['concurrent_qps']/float(before['concurrent_qps']):.2f}× | {z['concurrent_qps']:.2f} | {x['recall_at_k_pct']:.3f} | {z['recall_at_k_pct']:.3f} |")
 print('',end='')
for row in rows[::2]:
 before=old[('xvec',row['label_percentage'])]
 assert abs(row['recall_at_k_pct']-float(before['recall_at_k_pct']))<1e-6,(row['label_percentage'],row['recall_at_k_pct'],before['recall_at_k_pct'])
print('\nAll nine xvec recall values unchanged on the same persisted graph.')
