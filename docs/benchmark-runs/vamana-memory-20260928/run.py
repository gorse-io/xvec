import hashlib, json, os, pathlib, subprocess, time
def file_hash(path):
 with path.open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()

root = pathlib.Path(__file__).parent
repo = pathlib.Path('/home/zhenghaoz/xvec')
source_files = ['collection.go','internal/core/algorithm/vamana_algorithm.go','internal/core/algorithm/vamana_quantized_searcher.go']
patch = subprocess.check_output(['git','diff','HEAD','--',*source_files], cwd=repo)
(root/'source.patch').write_bytes(patch)
revision = subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
patch_hash = hashlib.sha256(patch).hexdigest()
metadata = {
 'machine':'e2-standard-8', 'cpu':'AMD EPYC 7B12', 'cpu_affinity':'0-7',
 'gomaxprocs':8, 'gomemlimit':'24GiB', 'cgo_enabled':0,
 'base_commit': revision, 'source_patch_sha256': patch_hash,
 'backend_version': revision+'+vamana-memory.'+patch_hash[:12],
 'binary_sha256':file_hash(root/'vector-db-bench'),
 'dataset_sha256':{p.name:file_hash(p) for p in sorted((root/'dataset').glob('*.parquet'))},
 'runs':[]
}
(root/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
for precision in ['int4','int8','fp16','fp32']:
 name = 'xvec-'+precision
 args = ['taskset','-c','0-7',str(root/'vector-db-bench'),'xvec',
 '--path',str(root/(name+'.collection')),'--case-type','Performance768D100K',
 '--dataset-dir',str(root/'dataset'),'--skip-download','--index-type','vamana',
 '--ef-search','200','--k','100','--batch-size','100','--max-docs-per-segment','10000000',
 '--optimize-concurrency','8','--num-concurrency','8','--concurrency-duration','30s',
 '--serial-cooldown','3s','--payload-profile','ids_only','--enable-mmap=true',
 '--is-using-refiner=false','--output',str(root/(name+'.json'))]
 if precision != 'fp32': args += ['--quantize-type',precision]
 env = dict(os.environ,GOMAXPROCS='8',GOMEMLIMIT='24GiB')
 print('START',name,flush=True)
 start = time.monotonic()
 with (root/(name+'.log')).open('w') as log:
  proc = subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=repo)
  _, status, usage = os.wait4(proc.pid,0)
  proc.returncode = os.waitstatus_to_exitcode(status)
 elapsed = time.monotonic()-start
 metrics = {'precision':precision,'command':args,'exit_code':proc.returncode,
 'peak_rss_kib':usage.ru_maxrss,'peak_rss_mib':usage.ru_maxrss/1024,
 'wall_seconds':elapsed,'user_cpu_seconds':usage.ru_utime,'system_cpu_seconds':usage.ru_stime}
 (root/(name+'.resources.json')).write_text(json.dumps(metrics,indent=2)+'\n')
 metadata['runs'].append(metrics)
 (root/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
 print('FINISH',name,json.dumps(metrics),flush=True)
 if proc.returncode: raise SystemExit(proc.returncode)
 print((root/(name+'.log')).read_text()[-2000:],flush=True)
