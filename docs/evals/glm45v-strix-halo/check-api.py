import argparse, base64, json, pathlib, time, urllib.request, urllib.error
p=argparse.ArgumentParser()
p.add_argument('--url',default='http://127.0.0.1:8080')
p.add_argument('--token-file')
p.add_argument('--image',default='/home/lelloman/glm45v-test/vision.png')
p.add_argument('--output',required=True)
a=p.parse_args()
headers={'Content-Type':'application/json'}
if a.token_file:
    headers['Authorization']='Bearer '+json.loads(pathlib.Path(a.token_file).read_text())['access_token']
image='data:image/png;base64,'+base64.b64encode(pathlib.Path(a.image).read_bytes()).decode()
cases=[('text','Reply with exactly: GLM integration ready.'),('vision',[{'type':'text','text':'Transcribe all text visible in this image, including the heading and amount. Return only the text.'},{'type':'image_url','image_url':{'url':image}}])]
results=[]
for name,content in cases:
    for streaming in (False,True):
        body={'model':'glm-4.5v','messages':[{'role':'user','content':content}], 'temperature':0,'max_tokens':128,'stream':streaming}
        req=urllib.request.Request(a.url+'/v1/chat/completions',data=json.dumps(body).encode(),headers=headers)
        start=time.monotonic()
        try:
            with urllib.request.urlopen(req,timeout=600) as r:
                status=r.status
                if streaming:
                    chunks=[];answer=''
                    for line in r:
                        line=line.decode().strip()
                        if line.startswith('data: ') and line!='data: [DONE]':
                            chunk=json.loads(line[6:]);chunks.append(chunk)
                            for choice in chunk.get('choices',[]): answer+=choice.get('delta',{}).get('content') or ''
                    raw=chunks
                else:
                    raw=json.load(r);answer=raw['choices'][0]['message']['content']
        except urllib.error.HTTPError as e:
            raise RuntimeError(f'{name} stream={streaming}: HTTP {e.code}: {e.read().decode()}')
        expected=['GLM integration ready.'] if name=='text' else ['HALO TEST 7319','Total: EUR 42.70']
        result=dict(case=name,stream=streaming,status=status,elapsed_s=time.monotonic()-start,answer=answer,passed=all(x in answer for x in expected),response=raw)
        results.append(result)
        print(json.dumps({k:v for k,v in result.items() if k!='response'}),flush=True)
        pathlib.Path(a.output).write_text(json.dumps(results,indent=2))
assert all(r['passed'] for r in results),'Unexpected response content'
