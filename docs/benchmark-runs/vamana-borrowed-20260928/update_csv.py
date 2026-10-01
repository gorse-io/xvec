import csv, json, pathlib, shutil
root=pathlib.Path(__file__).parent
repo=pathlib.Path('/home/zhenghaoz/xvec')
archive=repo/'docs/benchmark-runs/vamana-borrowed-20260928'
meta=json.loads((root/'metadata.json').read_text())
assert len(meta['runs'])==4 and all(r['exit_code']==0 for r in meta['runs'])
with (archive/'previous.csv').open() as f:
 reader=csv.DictReader(f); fields=reader.fieldnames; rows=list(reader)
for row in rows:
 if row['backend']!='xvec': continue
 kind=row['quantize_type']; name='xvec-'+kind
 report=json.loads((root/(name+'.json')).read_text()); resource=json.loads((root/(name+'.resources.json')).read_text())
 c=report['config']; load=report['load']; serial=report['serial']; conc=report['concurrent'][0]
 assert load['rows']==100000 and serial['queries']==1000 and conc['concurrency']==8
 assert len(report['concurrent'])==1 and report['case']['name']==row['case']
 assert report['system']['go_version']==row['go_version']
 assert c['backend']=='xvec' and c['index_type']=='vamana'
 assert c['quantize_type']==('none' if kind=='fp32' else kind)
 assert c['ef_search']==200 and c['concurrency_duration']=='30s' and c['serial_cooldown']=='3s'
 for key in ['use_refiner','enable_mmap','k','batch_size','max_docs_per_segment','optimize_concurrency','payload_profile']:
  assert str(c[key]).lower()==row[key],(key,c[key],row[key])
 row['backend_version']=meta['backend_version']
 row['inserted_count']=load['rows']
 for key in ['insert_duration_sec','optimize_duration_sec','load_duration_sec']: row[key]=load[key]
 row['insert_rows_per_sec']=load['rows_per_second']
 row['recall_at_k_pct']=serial['recall']*100
 for prefix,metrics in [('serial',serial),('concurrent',conc)]:
  for key in ['queries','qps','latency_avg_ms','latency_p95_ms','latency_p99_ms']: row[prefix+'_'+key]=metrics[key]
 for key in ['peak_rss_kib','peak_rss_mib','wall_seconds','user_cpu_seconds','system_cpu_seconds']: row[key]=resource[key]
 for key,value in row.items():
  if isinstance(value,float): row[key]=f'{value:.6f}'.rstrip('0').rstrip('.')
 for suffix in ['.json','.resources.json','.log']: shutil.copy2(root/(name+suffix),archive/(name+suffix))
with (repo/'docs/benchmark-vamana.csv').open('w') as f:
 writer=csv.DictWriter(f,fields,lineterminator='\n'); writer.writeheader(); writer.writerows(rows)
shutil.copy2(root/'metadata.json',archive/'metadata.json')
# Store the exact commands and wait4 implementation for reproducibility.
shutil.copy2(root/'run.py',archive/'run.py')
shutil.copy2(root/'update_csv.py',archive/'update_csv.py')
with (archive/'previous.csv').open() as f: old={r['quantize_type']:r for r in csv.DictReader(f) if r['backend']=='xvec'}
for row in rows:
 if row['backend']=='xvec':
  prior=old[row['quantize_type']]
  print(row['quantize_type'],'RSS MiB',prior['peak_rss_mib'],'->',row['peak_rss_mib'], 'change%',round((float(row['peak_rss_mib'])/float(prior['peak_rss_mib'])-1)*100,2),'recall',row['recall_at_k_pct'])
