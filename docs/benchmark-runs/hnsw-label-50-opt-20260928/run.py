import hashlib,json,os,pathlib,subprocess,time
root=pathlib.Path(__file__).resolve().parent
repo=pathlib.Path('/home/zhenghaoz/xvec')
previous=pathlib.Path('/home/zhenghaoz/hnsw-label-20260928')
old=json.loads((previous/'metadata.json').read_text())
previous_query=pathlib.Path('/home/zhenghaoz/hnsw-filter-opt-20260928')
source_files=['collection.go','collection_filter.go','internal/core/algorithm/quantized_flat_searcher.go','internal/core/algorithm/hnsw_quantized_searcher.go','internal/core/algorithm/hnsw_utility.go','internal/core/algorithm/hnsw_prefetch_amd64.go','internal/core/algorithm/hnsw_prefetch_amd64.s','internal/core/algorithm/hnsw_prefetch_generic.go']
patch=subprocess.check_output(['git','diff','HEAD','--',*source_files],cwd=repo)
for name in source_files:
 if subprocess.run(['git','ls-files','--error-unmatch',name],cwd=repo,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode:
  r=subprocess.run(['git','diff','--no-index','--','/dev/null',name],cwd=repo,stdout=subprocess.PIPE);assert r.returncode==1;patch+=r.stdout
(root/'source.patch').write_bytes(patch)
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
meta={k:v for k,v in old.items() if k!='runs'}
meta['source_files']=source_files;meta['source_patch_sha256']=hashlib.sha256(patch).hexdigest()
meta['backend_version']=meta['xvec_revision']+'+filter50.'+meta['source_patch_sha256'][:12]
meta['binary_sha256']=sha(root/'vector-db-bench')
meta['build_command']=['go','build','-o',str(root/'vector-db-bench'),'./cmd/vector-db-bench']
meta['method']='Query-only rerun of both backends on exactly the previous persisted FP16 HNSW collections; no graph rebuild. Build measurements remain the original runs. Same nine rates, 100 warmup queries, 30s concurrent, 3s cooldown, all 1000 serial queries. Measure 50% first, then the other eight rates in the original alternating backend order.'
meta['build_backend_versions']={'xvec':old['xvec_revision'],'zvec':'v0.7.0'}
meta['previous_metadata_sha256']=sha(previous_query/'metadata.json')
meta['runs']=[]
env=dict(os.environ,GOMAXPROCS='8',GOMEMLIMIT='24GiB',ZVEC_LIBRARY_PATH=str(previous/'native/linux_amd64/libzvec_c_api.so'))
query_runs=[r for r in old['runs'] if r['phase']=='query']
query_runs.sort(key=lambda r: 0 if r['name'].endswith('-50p') else 1)
for original in query_runs:
 if original['phase']!='query':continue
 name=original['name'];assert not (root/(name+'.json')).exists()
 args=original['command'][:];args[3]=str(root/'vector-db-bench')
 args[args.index('--output')+1]=str(root/(name+'.json'))
 args[args.index('--note')+1]='Same-graph 50%-focused filter/prefetch optimization rerun; '+meta['backend_version']
 print('START',name,flush=True);start=time.monotonic()
 with (root/(name+'.log')).open('w') as log:
  proc=subprocess.Popen(args,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
  _,status,usage=os.wait4(proc.pid,0);proc.returncode=os.waitstatus_to_exitcode(status)
 resource={**original,'command':args,'exit_code':proc.returncode,'peak_rss_kib':usage.ru_maxrss,'peak_rss_mib':usage.ru_maxrss/1024,
  'wall_seconds':time.monotonic()-start,'user_cpu_seconds':usage.ru_utime,'system_cpu_seconds':usage.ru_stime}
 (root/(name+'.resources.json')).write_text(json.dumps(resource,indent=2)+'\n')
 meta['runs'].append(resource);(root/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
 print('FINISH',name,'RSS MiB',resource['peak_rss_mib'],flush=True)
 print((root/(name+'.log')).read_text()[-1400:],flush=True)
 if proc.returncode:raise SystemExit(proc.returncode)
 report=json.loads((root/(name+'.json')).read_text());assert report['serial']['queries']==1000 and report['config']['quantize_type']=='fp16'
