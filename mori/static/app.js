'use strict';
const $=id=>document.getElementById(id);
const labels={retrieval_k:'Semantic candidates',jev_k:'Jev shortlist',final_passages:'Final passages (no reranker)',full_units:'Complete source units',summaries:'Other source summaries',min_relevance:'Minimum passage relevance',summary_min_relevance:'Minimum summary relevance',summary_candidates:'Summary candidate cap (0 = all)',rerank_k:'Final passages with reranker',rating_neighbors:'Neighbors shown to Jev',answer_neighbors:'Neighbors in answer packet'};
let defaults={},current=null,busy=false,polling=false,readingVersion=0,loadingRun=false,serverBusy=false;
let sources=[],selectedSources=new Set(),sourcesReady=false,selectionInitialized=false,selectAllSources=true,allowUploads=false;
const sourceParams=globalThis.location?new URLSearchParams(globalThis.location.search):null;
const requestedSources=sourceParams?.has('sources')?sourceParams.getAll('sources'):null;
function status(text){$('status').textContent=text;}
function renderAnswer(text){$('answer-view').replaceChildren();for(const block of text.split(/\n\s*\n/)){const heading=/^(#{1,3})\s+/.exec(block);const element=document.createElement(heading?'h3':'p');const value=heading?block.slice(heading[0].length):block;for(const part of value.split(/(\*\*[^*]+\*\*|\[[UMP]\d+\])/g)){if(/^\[[UMP]\d+\]$/.test(part)){const button=document.createElement('button');button.className='citation';button.textContent=part;button.addEventListener('click',()=>{showTab('evidence');const target=document.querySelector(`[data-citation="${part.slice(1,-1)}"]`);if(target){target.open=true;target.scrollIntoView({block:'center',behavior:'smooth'});}});element.append(button);}else if(part.startsWith('**')&&part.endsWith('**')){const strong=document.createElement('strong');strong.textContent=part.slice(2,-2);element.append(strong);}else element.append(document.createTextNode(part));}$('answer-view').append(element);}}
async function api(path,data){const r=await fetch(path,data?{method:'POST',headers:{'Content-Type':'application/json','X-Mori-Token':document.querySelector('meta[name=mori-token]').content},body:JSON.stringify(data)}:{});const v=await r.json();if(!r.ok)throw Error(v.error||'Request failed');return v;}
function settings(){const result={};for(const key of Object.keys(labels))result[key]=Number($(key).value);result.rerank_enabled=$('rerank_enabled').checked;result.exclude_full_unit_passages=$('exclude_full_unit_passages').checked;return result;}
function selectedAvailableSources(){return sources.filter(source=>selectedSources.has(source.id));}
function unavailableSources(){const available=new Set(sources.map(source=>source.id));return [...selectedSources].filter(id=>!available.has(id));}
function selectionError(){const missing=unavailableSources();return missing.length?`Unavailable sources: ${missing.map(id=>id||'(empty source ID)').join(', ')}. Open Sources and choose available sources or select all to continue.`:'';}
function updateControls(){
 const locked=busy||serverBusy||loadingRun||!sourcesReady,selected=selectedAvailableSources(),error=selectionError();
 $('composer').setAttribute('aria-busy',String(busy||serverBusy||loadingRun));
 $('source-picker').disabled=locked;
 for(const id of ['retrieve','answer','plan'])$(id).disabled=locked||!selected.length||Boolean(error);
 $('upload').disabled=locked||!allowUploads;
 $('sources-all').disabled=locked||!sources.length||(!error&&selected.length===sources.length);
 $('sources-none').disabled=locked||!selectedSources.size;
 $('source-error').textContent=error;$('source-error').hidden=!error;
 $('source-count').textContent=!sourcesReady?'Loading…':!sources.length?'No sources available':!selected.length?'None selected':selected.length===sources.length?`All ${sources.length} selected`:`${selected.length} of ${sources.length} selected`;
 $('source-summary').textContent=!sourcesReady?'Loading sources…':!sources.length?(allowUploads?'No sources available. Import a source to begin.':'No sources are available in this corpus.'):`${selected.length} of ${sources.length} sources selected · ${selected.reduce((sum,source)=>sum+source.units,0)} source units · ${selected.reduce((sum,source)=>sum+source.chunks,0)} passages${selected.length?'':'. Choose at least one source.'}`;
}
function working(value){busy=value;updateControls();}
function renderSources(){
 $('source-options').replaceChildren();
 for(const source of sources){
  const label=document.createElement('label'),input=document.createElement('input'),description=document.createElement('span'),name=document.createElement('span'),counts=document.createElement('small');
  label.className='source-option';input.type='checkbox';input.value=source.id;input.checked=selectedSources.has(source.id);
  name.textContent=source.label;counts.textContent=`${source.units} source units · ${source.chunks} passages`;description.append(name,counts);label.append(input,description);
  input.addEventListener('change',()=>{
   if(busy||serverBusy||loadingRun||!sourcesReady)return;
   // Changing the selection explicitly discards unavailable URL or historical IDs.
   selectedSources=new Set(selectedAvailableSources().map(row=>row.id));selectAllSources=false;
   if(input.checked)selectedSources.add(source.id);else selectedSources.delete(source.id);
   updateControls();
  });
  $('source-options').append(label);
 }
 updateControls();
}
function syncSources(next){
 if(!Array.isArray(next)){sourcesReady=false;updateControls();throw Error('The source list is unavailable. Reload before submitting a question.');}
 sources=next;
 if(!selectionInitialized&&requestedSources!==null){selectedSources=new Set(requestedSources);selectAllSources=false;}
 else if(selectAllSources)selectedSources=new Set(sources.map(source=>source.id));
 sourcesReady=true;selectionInitialized=true;renderSources();
}
function querySources(){
 if(!sourcesReady)throw Error('Sources are still loading.');
 if(busy||serverBusy||loadingRun)throw Error('Wait for the current operation to finish.');
 const error=selectionError();if(error)throw Error(error);
 const ids=selectedAvailableSources().map(source=>source.id);if(!ids.length)throw Error('Choose at least one source.');return ids;
}
for(const [id,all]of [['sources-all',true],['sources-none',false]])$(id).addEventListener('click',()=>{
 if(busy||serverBusy||loadingRun||!sourcesReady)return;
 selectAllSources=all;selectedSources=new Set(all?sources.map(source=>source.id):[]);renderSources();
});
async function refresh(){const s=await api('/api/state');serverBusy=s.job.status==='running';allowUploads=s.allow_uploads===true;$('add-sources').hidden=!allowUploads;syncSources(s.sources);$('stats').textContent=`${s.stats.units} source units · ${s.stats.chunks} passages`;$('providers').textContent=s.providers.embedding==='demo'?'DEMO FIXTURE · NOT SEMANTIC':`${s.providers.embedding} embeddings · ${s.providers.llm==='none'?'evidence handoff':s.providers.llm+' answers'}`;if(!Object.keys(defaults).length){defaults=s.settings;for(const [key,label]of Object.entries(labels)){const l=document.createElement('label');l.textContent=label;const input=document.createElement('input');input.type='number';input.id=key;input.min=key.includes('relevance')||['full_units','summaries','summary_candidates','rating_neighbors','answer_neighbors'].includes(key)?'0':'1';if(key.includes('relevance'))input.max='100';input.value=defaults[key];l.append(input);$('fields').append(l);}$('rerank_enabled').checked=defaults.rerank_enabled;$('exclude_full_unit_passages').checked=defaults.exclude_full_unit_passages;}
const runs=await api('/api/runs');$('history').replaceChildren();for(const run of runs){const b=document.createElement('button');b.textContent=run.question;b.disabled=run.status!=='complete';b.addEventListener('click',()=>openRun(run.id).catch(e=>status(e.message)));$('history').append(b);}return s;}
async function watch(){if(polling)return;polling=true;try{const s=await refresh();if(s.job.status==='running'){working(true);status('Working on your sources. Successful relevance decisions are saved as they finish.');setTimeout(()=>{polling=false;watch();},1500);return;}working(false);if(s.job.status==='failed')status(s.job.error);else if(s.job.status==='complete'){status('Complete.');if(s.job.result.run_id&&current?.context?.run_id!==s.job.result.run_id)await openRun(s.job.result.run_id,current!==null||requestedSources===null);}}catch(e){working(false);status(e.message);}polling=false;}
async function openRun(id,restoreSelection=true){
 const version=++readingVersion;loadingRun=true;updateControls();
 try{
 const value=await api('/api/runs/'+encodeURIComponent(id));if(version!==readingVersion)return;
 if(restoreSelection&&Array.isArray(value.context.sources)){selectAllSources=false;selectedSources=new Set(value.context.sources.map(source=>source.id));renderSources();}
 current=value;$('question').value=value.context.question;$('result-title').textContent=value.context.question;$('result').hidden=false;
 renderAnswer(value.answer||'Your evidence packet is ready. Use the Model handoff tab with your preferred LLM, or configure an answer provider.');
 $('prompt').textContent=value.prompt;$('counts').replaceChildren();
 for(const [key,label]of [['full_units','complete units'],['summaries','summaries'],['passages','passages']]){const span=document.createElement('span');span.textContent=`${value.context.counts[key]} ${label}`;$('counts').append(span);}
 $('notices').textContent=value.context.notices.join('. ');$('evidence-view').replaceChildren();
 for(const [group,key]of [['full_units','text'],['summaries','summary'],['passages','target']])for(const row of value.context[group]){
  const details=document.createElement('details'),summary=document.createElement('summary'),meta=document.createElement('p'),pre=document.createElement('pre');
  details.dataset.citation=row.citation;summary.textContent=`[${row.citation}] ${row.title||row.unit.title} · ${group==='summaries'?'summary only':group==='full_units'?'complete original':'original passage'}`;
  const identity=row.unit||row;meta.textContent=`${identity.locator} · Source: ${identity.source_id} · Relevance: ${row.rating.score.toFixed(2)}/100`;
  pre.textContent=key==='target'?row.target.text+row.neighbors.map(n=>'\n\nAdjacent context:\n'+n.text).join(''):row[key];
  details.append(summary,meta,pre);$('evidence-view').append(details);
 }
 showTab(value.answer?'answer':'evidence');
 }finally{if(version===readingVersion){loadingRun=false;updateControls();}}
}
function showTab(name){for(const key of ['answer','evidence','prompt']){$(key+'-tab').setAttribute('aria-selected',String(name===key));$(key+'-view').hidden=name!==key;}}
for(const name of ['answer','evidence','prompt']){$(name+'-tab').addEventListener('click',()=>showTab(name));$(name+'-tab').addEventListener('keydown',e=>{const names=['answer','evidence','prompt'];let next=names.indexOf(name);if(e.key==='ArrowRight')next=(next+1)%3;else if(e.key==='ArrowLeft')next=(next+2)%3;else return;e.preventDefault();showTab(names[next]);$(names[next]+'-tab').focus();});}
for(const [id,answer]of [['retrieve',false],['answer',true]])$(id).addEventListener('click',async()=>{if(busy||serverBusy||loadingRun)return;try{const sources=querySources();working(true);await api('/api/query',{question:$('question').value,settings:settings(),answer,sources});watch();}catch(e){working(false);status(e.message);}});
$('plan').addEventListener('click',async()=>{if(busy||serverBusy||loadingRun)return;try{const sources=querySources();working(true);const p=await api('/api/plan',{settings:settings(),sources});status(`Up to ${p.candidate_passages} passage ratings + ${p.summary_ratings} summary ratings. Up to ${p.rating_attempt_ceiling} attempts including retries; cached decisions are reused. Final packet: ${p.full_units} complete units, ${p.summaries} summaries, up to ${p.final_passage_cap} passages. No provider calls made by this budget check.`);}catch(e){status(e.message);}finally{working(false);}});
function fileData(file){return new Promise((resolve,reject)=>{const r=new FileReader();r.onload=()=>resolve(String(r.result).split(',')[1]);r.onerror=reject;r.readAsDataURL(file);});}
$('upload').addEventListener('click',async()=>{if(busy||serverBusy||loadingRun||!sourcesReady||!allowUploads)return;try{const files=Array.from($('files').files);if(files.length!==1)throw Error('Import one file at a time. A JSONL file can contain many units.');if(files[0].size>8*1024*1024)throw Error('Upload exceeds 8 MB. Split into meaningful units.');working(true);await api('/api/upload',{filename:files[0].name,data:await fileData(files[0]),split:$('split').value,summarize:$('summarize').checked});watch();}catch(e){working(false);status(e.message);}});
$('download').addEventListener('click',()=>{if(!current)return;const url=URL.createObjectURL(new Blob([JSON.stringify(current.context,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='mori-'+current.context.run_id+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
working(true);
watch();
