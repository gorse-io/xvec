import pathlib,subprocess,json,os
root=pathlib.Path(__file__).parent
meta=json.loads((root/'metadata.json').read_text())
base=meta['runs'][-1]['command']
for label,binary in [('before','/home/zhenghaoz/vamana-retest-20260928/vector-db-bench'),('after',str(root/'vector-db-bench'))]:
 args=base.copy();args[3]=binary
 args[args.index('--concurrency-duration')+1]='10s'
 args[args.index('--output')+1]=str(root/('query-'+label+'.json'))
 args+=['--skip-load','--skip-drop-old']
 print('START query',label,flush=True)
 with (root/('query-'+label+'.log')).open('w') as f:
  subprocess.run(args,env=dict(os.environ,GOMAXPROCS='8',GOMEMLIMIT='24GiB'),stdout=f,stderr=subprocess.STDOUT,check=True)
 print((root/('query-'+label+'.log')).read_text(),flush=True)
