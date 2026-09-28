import csv,hashlib,json,pathlib,subprocess
repo=pathlib.Path('/home/zhenghaoz/xvec')
root=pathlib.Path('/home/zhenghaoz/hnsw-filter-opt-20260928')
previous=pathlib.Path('/home/zhenghaoz/hnsw-label-20260928')
archive=repo/'docs/benchmark-runs/hnsw-label-opt-20260928'
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def close(a,b):assert abs(float(a)-float(b))<=1e-6,(a,b)
meta=json.loads((archive/'metadata.json').read_text())
with (repo/'docs/benchmark-hnsw-label-filter.csv').open() as f:
 reader=csv.DictReader(f);rows=list(reader);assert len(reader.fieldnames)==53
old={(r['backend'],r['label_percentage']):r for r in csv.DictReader((archive/'previous.csv').open())}
assert len(rows)==18 and len(meta['runs'])==18
assert len({(r['backend'],r['label_percentage']) for r in rows})==18
resources={r['name']:r for r in meta['runs']}
for row in rows:
 backend=row['backend'];rate=float(row['label_percentage']);name=f'{backend}-fp16-label-{rate*100:g}p'
 report=json.loads((archive/(name+'.json')).read_text());s=report['serial'];q=report['concurrent'][0];c=report['config'];r=resources[name]
 assert r['exit_code']==0 and s['queries']==1000 and q['concurrency']==8
 assert c['quantize_type']=='fp16' and c['label_percentage']==rate and c['backend']==backend
 assert c['m']==50 and c['ef_construction']==500 and c['ef_search']==300 and c['k']==100
 assert c['concurrency_duration']=='30s' and c['serial_cooldown']=='3s'
 assert c['enable_mmap'] and not c['use_refiner'] and c['filter_expression']==row['filter_expression']
 assert 'load' not in report
 assert row['build_backend_version']==old[(backend,row['label_percentage'])]['backend_version']
 assert row['backend_version']==(meta['backend_version'] if backend=='xvec' else 'v0.7.0')
 for field in ['inserted_count','insert_duration_sec','optimize_duration_sec','load_duration_sec','build_peak_rss_mib']:
  assert row[field]==old[(backend,row['label_percentage'])][field]
 close(row['recall_at_k_pct'],s['recall']*100)
 if backend=='xvec':close(row['recall_at_k_pct'],old[(backend,row['label_percentage'])]['recall_at_k_pct'])
 for prefix,metrics in [('serial',s),('concurrent',q)]:
  for field in ['queries','qps','latency_avg_ms','latency_p95_ms','latency_p99_ms']:close(row[prefix+'_'+field],metrics[field])
 for field in ['peak_rss_mib','wall_seconds','user_cpu_seconds','system_cpu_seconds']:close(row['query_'+field],r[field])
 for suffix in ['.json','.log','.resources.json']:assert (archive/(name+suffix)).read_bytes()==(root/(name+suffix)).read_bytes()
print('PASS: 18 unique successful runs; CSV settings, metrics, resources and archived reports match.')
print('PASS: All nine xvec recall values unchanged; original build metrics and versions preserved.')
patch=subprocess.check_output(['git','diff','HEAD','--',*meta['source_files']],cwd=repo)
for name in meta['source_files']:
 if subprocess.run(['git','ls-files','--error-unmatch',name],cwd=repo,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode:
  result=subprocess.run(['git','diff','--no-index','--','/dev/null',name],cwd=repo,stdout=subprocess.PIPE);assert result.returncode==1;patch+=result.stdout
assert patch==(archive/'source.patch').read_bytes()
assert hashlib.sha256(patch).hexdigest()==meta['source_patch_sha256']
assert sha(root/'vector-db-bench')==meta['binary_sha256']
assert sha(previous/'metadata.json')==meta['previous_metadata_sha256']
assert sha(previous/'native/linux_amd64/libzvec_c_api.so')==meta['native_library_sha256']
assert sha(previous/'native/zvec-libs-linux-x64.tar.gz')==meta['native_archive_sha256']
for filename,digest in meta['dataset_sha256'].items():assert sha(previous/'dataset'/filename)==digest
subprocess.run(['git','apply','--reverse','--check',str(archive/'source.patch')],cwd=repo,check=True)
subprocess.run(['git','diff','--check'],cwd=repo,check=True)
print('PASS: Current source matches measured patch; binary, native release and dataset hashes verified.')
print('PASS: Source patch reverse check and git diff --check.')
