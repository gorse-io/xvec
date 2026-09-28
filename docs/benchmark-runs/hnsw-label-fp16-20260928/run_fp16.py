import hashlib, json, os, pathlib, subprocess, time
root = pathlib.Path(__file__).resolve().parent
repo = pathlib.Path('/home/zhenghaoz/xvec')
native = root/'native/linux_amd64/libzvec_c_api.so'
binary = root/'vector-db-bench'
def sha(path):
 with path.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
meta = {
 'case':'LabelFilterPerformanceCase','dataset':'Small Cohere (768dim, 100K)',
 'xvec_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
 'zvec_go_version':'v0.7.0','zvec_native_release':'https://github.com/zvec-ai/zvec-go/releases/tag/v0.7.0',
 'binary_sha256':sha(binary),'native_library_sha256':sha(native),
 'native_archive_sha256':sha(root/'native/zvec-libs-linux-x64.tar.gz'),
 'dataset_sha256':{p.name:sha(p) for p in sorted((root/'dataset').glob('*.parquet'))},
 'machine':'e2-standard-8','cpu':'AMD EPYC 7B12','gomaxprocs':8,'gomemlimit':'24GiB','cpu_affinity':'0-7','cgo_enabled':0,
 'build_command':['go','build','-o',str(binary),'./cmd/vector-db-bench'],
 'method':'Build one fresh FP16 HNSW collection per backend, then reuse it across all nine label rates. Build and query resources are measured separately. Alternate backend order between rates. Single run per condition.',
 'runs':[]
}
assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=repo), 'Tracked sources must be clean'
env=dict(os.environ,GOMAXPROCS='8',GOMEMLIMIT='24GiB',ZVEC_LIBRARY_PATH=str(native))
def execute(backend, rate, phase):
 tag=f'{rate*100:g}p';name=f'{backend}-fp16-'+('build' if phase=='build' else f'label-{tag}')
 collection=root/f'{backend}-fp16.collection'
 if phase=='build': assert not collection.exists(), collection
 report=root/(name+'.json')
 assert not report.exists(), report
 args=['taskset','-c','0-7',str(binary),backend,'--path',str(collection),
 '--case-type','LabelFilterPerformanceCase','--dataset-with-size-type','Small Cohere (768dim, 100K)',
 '--label-percentage',str(rate),'--dataset-dir',str(root/'dataset'),'--skip-download',
 '--index-type','hnsw','--m','50','--ef-construction','500','--ef-search','300',
 '--quantize-type','fp16','--k','100','--batch-size','100','--max-docs-per-segment','10000000','--optimize-concurrency','8',
 '--num-concurrency','8','--concurrency-duration','30s','--serial-cooldown','3s','--warmup-queries','100',
 '--payload-profile','ids_only','--enable-mmap=true','--is-using-refiner=false','--seed','0',
 '--note',f'Cohere 100K HNSW label filter; FP16; {phase} resources measured separately', '--output',str(report)]
 if phase=='build': args+=['--skip-search-serial','--skip-search-concurrent']
 else: args+=['--skip-load','--skip-drop-old']
 print('START',name,flush=True)
 start=time.monotonic()
 with (root/(name+'.log')).open('w') as log:
  proc=subprocess.Popen(args,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
  _,status,usage=os.wait4(proc.pid,0);proc.returncode=os.waitstatus_to_exitcode(status)
 resource={'name':name,'backend':backend,'phase':phase,'label_percentage':rate,'command':args,'exit_code':proc.returncode,
 'peak_rss_kib':usage.ru_maxrss,'peak_rss_mib':usage.ru_maxrss/1024,'wall_seconds':time.monotonic()-start,
 'user_cpu_seconds':usage.ru_utime,'system_cpu_seconds':usage.ru_stime}
 (root/(name+'.resources.json')).write_text(json.dumps(resource,indent=2)+'\n')
 meta['runs'].append(resource);(root/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
 print('FINISH',name,'exit',proc.returncode,'RSS MiB',resource['peak_rss_mib'],flush=True)
 print((root/(name+'.log')).read_text()[-1800:],flush=True)
 if proc.returncode: raise SystemExit(proc.returncode)
 data=json.loads(report.read_text())
 if phase=='build': assert data['load']['rows']==100000
 else: assert data['serial']['queries']==1000 and len(data['concurrent'])==1 and data['concurrent'][0]['concurrency']==8
(root/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
for backend in ['xvec','zvec']: execute(backend,0.001,'build')
for i,rate in enumerate([0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5]):
 for backend in (['xvec','zvec'] if i%2==0 else ['zvec','xvec']): execute(backend,rate,'query')
