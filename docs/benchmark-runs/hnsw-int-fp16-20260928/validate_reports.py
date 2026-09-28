import csv,hashlib,json,pathlib,subprocess
repo=pathlib.Path('/home/zhenghaoz/xvec');root=pathlib.Path(__file__).resolve().parent;archive=repo/'docs/benchmark-runs/hnsw-int-fp16-20260928'
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def close(a,b):assert abs(float(a)-float(b))<1e-6,(a,b)
meta=json.loads((archive/'metadata.json').read_text());truth=json.loads((archive/'dataset-validation.json').read_text())
rows=list(csv.DictReader((repo/'docs/benchmark-hnsw-int-filter.csv').open()));assert len(rows)==18 and len(meta['runs'])==20
assert all(r['exit_code']==0 for r in meta['runs']);resources={r['name']:r for r in meta['runs']}
assert len({(r['backend'],r['filter_rate']) for r in rows})==18
for row in rows:
 backend=row['backend'];percentage=float(row['matching_fraction']);rate=float(row['filter_rate']);name=f'{backend}-fp16-int-match-{percentage*100:g}p';buildname=f'{backend}-fp16-build'
 r=json.loads((archive/(name+'.json')).read_text());build=json.loads((archive/(buildname+'.json')).read_text())['load'];s=r['serial'];q=r['concurrent'][0];c=r['config'];t=next(t for t in truth['ground_truth'] if t['filter_rate']==rate)
 assert r['case']['name']=='NewIntFilterPerformanceCase' and c['local_int_ground_truth']
 assert c['filter_rate']==rate and row['filter_expression']==c['filter_expression']==t['filter_expression']
 assert int(row['matching_documents'])==t['matching_documents']==100000-int(100000*rate)
 close(percentage,t['matching_documents']/100000)
 assert c['backend']==backend and c['index_type']=='hnsw' and c['quantize_type']=='fp16'
 assert c['m']==50 and c['ef_construction']==500 and c['ef_search']==300 and c['k']==100
 assert c['enable_mmap'] and not c['use_refiner'] and c['concurrency_duration']=='30s' and c['serial_cooldown']=='3s'
 assert s['queries']==1000 and q['concurrency']==8 and build['rows']==100000 and 'load' not in r
 assert row['backend_version']==row['build_backend_version']==(meta['backend_version'] if backend=='xvec' else 'v0.7.0')
 assert row['harness_revision']==meta['backend_version'] and row['ground_truth_source']=='local_exact_fp64_cosine' and row['ground_truth_file']==t['file']
 close(row['recall_at_k_pct'],s['recall']*100)
 for prefix,m in [('serial',s),('concurrent',q)]:
  for k in ['queries','qps','latency_avg_ms','latency_p95_ms','latency_p99_ms']:close(row[prefix+'_'+k],m[k])
 for k in ['peak_rss_mib','wall_seconds','user_cpu_seconds','system_cpu_seconds']:close(row['query_'+k],resources[name][k])
 close(row['build_peak_rss_mib'],resources[buildname]['peak_rss_mib'])
 for k in ['insert_duration_sec','optimize_duration_sec','load_duration_sec']:close(row[k],build[k])
 for run in [name,buildname]:
  for suffix in ['.json','.log','.resources.json']:assert (root/(run+suffix)).read_bytes()==(archive/(run+suffix)).read_bytes()
print('PASS: 2 fresh builds and 18 successful query runs; all CSV parameters, metrics and resources match raw reports.')
for name,digest in meta['dataset_sha256'].items():assert sha(root/'dataset'/name)==digest
for t in truth['ground_truth']:assert sha(archive/'ground-truth'/t['file'])==t['sha256']
assert sha(archive/'dataset-validation.json')==meta['ground_truth_validation_sha256']
assert sha(root/'vector-db-bench')==meta['binary_sha256']
assert sha(pathlib.Path('/home/zhenghaoz/hnsw-label-20260928/native/linux_amd64/libzvec_c_api.so'))==meta['native_library_sha256']
patch=subprocess.check_output(['git','diff','HEAD','--',*meta['source_files']],cwd=repo)
for name in meta['source_files']:
 if subprocess.run(['git','ls-files','--error-unmatch',name],cwd=repo,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode:
  diff=subprocess.run(['git','diff','--no-index','--','/dev/null',name],cwd=repo,stdout=subprocess.PIPE);assert diff.returncode==1;patch+=diff.stdout
assert patch==(archive/'source.patch').read_bytes()
assert hashlib.sha256(patch).hexdigest()==meta['source_patch_sha256']
subprocess.run(['git','apply','--reverse','--check',str(archive/'source.patch')],cwd=repo,check=True)
subprocess.run(['git','diff','--check'],cwd=repo,check=True)
print('PASS: Local exact ground truth, input datasets, binary, native library and measured source hashes verified.')
print('PASS: Source patch reverse check and git diff --check.')
