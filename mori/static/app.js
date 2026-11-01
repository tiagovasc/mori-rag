'use strict';
const $=id=>document.getElementById(id);
// English digit grouping without Intl: the first toLocaleString() call loads ICU data and delayed startup.
const formatNumber=n=>Number.isFinite(Number(n))?String(Math.trunc(Number(n))).replace(/\B(?=(\d{3})+(?!\d))/g,','):String(n);
const plural=(n,one,many=one+'s')=>`${formatNumber(n)} ${Number(n)===1?one:many}`;
const reduceMotion=()=>typeof matchMedia==='function'&&matchMedia('(prefers-reduced-motion: reduce)').matches;
const narrow=()=>typeof matchMedia==='function'&&matchMedia('(max-width: 900px)').matches;
const browserLayout=()=>typeof document.body?.getBoundingClientRect==='function';
// Per-browser conveniences only; storage may be unavailable, so every use is optional.
const storeKey=(key,fallback='')=>({get(){try{return globalThis.localStorage?.getItem(key)??fallback;}catch{return fallback;}},set(value){try{if(value)globalThis.localStorage?.setItem(key,value);else globalThis.localStorage?.removeItem(key);}catch{}}});
const draftStore=storeKey('mori.draft');
const markStore=storeKey('mori.mark-summaries');
const labels={retrieval_k:'Semantic candidates',jev_k:'Jev shortlist',final_passages:'Final passages (no reranker)',full_units:'Complete source units',summaries:'Other source summaries',min_relevance:'Minimum passage relevance',summary_min_relevance:'Minimum summary relevance',summary_candidates:'Summary candidate cap (0 = all)',rerank_k:'Final passages with reranker',rating_neighbors:'Neighbors shown to Jev',answer_neighbors:'Neighbors in answer packet'};
const MONTHS=['January','February','March','April','May','June','July','August','September','October','November','December'];
const WEEKDAYS=['Sun','Mon','Tue','Wed','Thu','Fri','Sat'];
const RAMP=['#e3e8e0','#d8e0d5','#c9d7c9','#b3c9b7','#98b9a1','#7ba58a','#5c8f72','#407859','#2a6246','#164e35'];
let defaults={},current=null,busy=false,polling=false,readingVersion=0,loadingRun=false,serverBusy=false;
let sources=[],selectedSources=new Set(),sourcesReady=false,selectionInitialized=false,selectAllSources=true,allowUploads=false;
let llmEnabled=false,modelOptions=[],selectedModel=null,modelCatalogKey=null;
let budgetOverhead=1024,maxRatingCalls=null,features=[];
let promptExtension=false;
let lastJob=null,jobKey=null,progressSuppressed=false,lastStateAt=0,localStartedAt=null,localFinishedAt=null,connectionLost=false,historyLoading=false;
const DRAFT_KEPT='Showing a saved inquiry. Your unsent question is still in the box.';
let workflowStepsKey=null,filledQuestion=null,followSubmission=false,lastSubmission=null,uploadSubmitted=false,historyKey=null,historyLoaded=false,sourcesKey=null,emptyCorpusOpened=false,workflowStateKey=null;
let screen='ask',routed=false,stopRequested=false,submitting=false,lastLive=null,liveTimer=null,liveBusy=false,liveKeys={};
let historyRuns=[],libraryFilter='all',libraryQuery='',libraryCollection='',collectionsKey=null,roomKey=null;
let answerBlocks=[],citeIndex=new Map(),citedOrder=[],readerCite=null,unitCache=new Map(),activeCite=null,droppedFile=null;
const sourceParams=globalThis.location?new URLSearchParams(globalThis.location.search):null;
const requestedSources=sourceParams?.has('sources')?sourceParams.getAll('sources'):null;
const explicitRoute=Boolean(globalThis.location?.hash&&globalThis.location.hash!=='#'&&globalThis.location.hash!=='#/');
let toastTurn=0;
function status(text,tone=''){
 // A new message restarts the toast's fade (CSS); errors stay until something replaces them.
 const box=$('status');if(box.textContent!==text){box.textContent=text;toastTurn=1-toastTurn;box.setAttribute('data-turn',String(toastTurn));}
 box.setAttribute('data-tone',text?tone:'');
}
function make(tag,className,text){const element=document.createElement(tag);if(className)element.className=className;if(text!==undefined)element.textContent=text;return element;}
function textIfChanged(element,text){if(element.textContent!==text)element.textContent=text;}
// Render keys live beside elements, not in the DOM, so unchanged polls skip rebuilding.
const renderKeys=new WeakMap();
function changedKey(element,key){if(renderKeys.get(element)===key)return false;renderKeys.set(element,key);return true;}
function validDate(value){const time=typeof value==='string'?Date.parse(value):NaN;return Number.isFinite(time)?time:null;}
function duration(milliseconds){const seconds=Math.max(0,Math.floor(milliseconds/1000));return seconds<60?`${seconds}s`:`${Math.floor(seconds/60)} min ${seconds%60} s`;}
const pad=n=>String(n).padStart(2,'0');
function clock(time){const d=new Date(time);return `${pad(d.getHours())}:${pad(d.getMinutes())}`;}
function longDate(time){const d=new Date(time);return `${d.getDate()} ${MONTHS[d.getMonth()]} ${d.getFullYear()} at ${clock(time)}`;}
function shortLabel(text){return String(text||'').split(':')[0].trim();}
function seg(level,levels=10,kind=''){
 // Jev's rubric as a bar: one step per level above the lowest.
 const bar=make('span','seg'),steps=Math.max(1,levels-1);if(kind)bar.dataset.kind=kind;bar.setAttribute('aria-hidden','true');
 for(let k=0;k<steps;k++){const step=make('i');if(level!==null&&k<level)step.dataset.on='true';bar.append(step);}
 return bar;
}
function levelFor(rating,levels=10){
 if(!rating||typeof rating!=='object')return null;
 const raw=Number.isFinite(rating.raw_score)?rating.raw_score:Number.isFinite(rating.score)?rating.score/100*(levels-1):null;
 return raw===null?null:Math.max(0,Math.min(levels-1,Math.floor(raw+0.5)));
}
function planMessage(p){
 const packet=p.evidence_mode==='context'?`whole passages that fit the ${formatNumber(p.context_window_tokens)}-token context budget, chosen from up to ${plural(p.final_passage_cap,'ranked passage')} (final usage is calculated after ranking; answer and overhead space are reserved)`:`up to ${plural(p.final_passage_cap,'passage')}`;
 const extension=p.expansion_model_calls?` Prompt extension adds ${plural(p.expansion_model_calls,'model call')} and up to ${plural(p.maximum_query_embeddings??1,'query embedding')}, also for evidence-only inquiries.`:'';
 return `Up to ${plural(p.candidate_passages,'passage rating')} and ${plural(p.summary_ratings,'summary rating')}, at most ${plural(p.rating_attempt_ceiling,'attempt')} including retries; cached decisions are reused. Final packet: ${plural(p.full_units,'complete unit')}, ${plural(p.summaries,'summary','summaries')} and ${packet}.${extension} No provider calls were made for this check.`;
}

/* ---------- screens and routes ---------- */
const SCREENS={ask:'screen-ask',progress:'workflow',inquiry:'result',library:'screen-library',settings:'screen-settings'};
const CRUMBS={ask:'New inquiry',progress:'Inquiry in progress',library:'Library',settings:'Settings'};
function show(name){
 if(!SCREENS[name])name='ask';
 const changed=screen!==name;screen=name;
 for(const [key,id] of Object.entries(SCREENS))$(id).hidden=key!==name;
 $('crumb').hidden=name==='inquiry';$('back').hidden=name!=='inquiry';$('views').hidden=name!=='inquiry';
 $('progress-actions').hidden=name!=='progress';$('result-actions').hidden=name!=='inquiry';
 if(name==='progress'&&lastJob?.kind==='upload')textIfChanged($('crumb'),'Import in progress');else textIfChanged($('crumb'),CRUMBS[name]||'');
 const page=name==='library'?(routeName()==='sources'?'nav-sources':'nav-library'):name==='settings'?'nav-settings':null;
 for(const id of ['nav-library','nav-sources','nav-settings']){if(id===page)$(id).setAttribute('aria-current','page');else $(id).removeAttribute('aria-current');}
 closeMenu();
 if(changed&&$('screens').scrollTo)$('screens').scrollTo({top:0});
 if(name==='progress')ensureLive();
 if(name==='inquiry')scheduleLayout();
}
function routeName(){return parseRoute(globalThis.location?.hash).name;}
function parseRoute(hash){
 let parts=[];
 try{parts=String(hash||'').replace(/^#\/?/,'').split('/').filter(Boolean).map(decodeURIComponent);}catch{parts=[];}
 if(!parts.length||parts[0]==='new')return {name:'ask'};
 if(parts[0]==='inquiry'&&parts[1])return {name:'inquiry',id:parts[1],cite:/^[UMP]\d{1,4}$/.test(parts[2]||'')?parts[2]:null};
 if(['library','sources','settings','progress'].includes(parts[0]))return {name:parts[0]};
 return {name:'ask'};
}
function navigate(hash){if(globalThis.location&&globalThis.location.hash!==hash)globalThis.location.hash=hash;}
async function route(){
 const r=parseRoute(globalThis.location?.hash);
 if(r.name==='inquiry'){
  if(current?.context?.run_id===r.id){show('inquiry');if(r.cite)openReader(r.cite,{updateHash:false});else if(readerCite)closeReader(false);}
  else await openRun(r.id,true,false,{cite:r.cite}).catch(e=>status(e.message,'error'));
  return;
 }
 if(r.name==='progress'){show(lastJob?'progress':'ask');return;}
 if(r.name==='sources'){show('library');const side=$('library-sources');side.classList?.remove('focus');void side.offsetWidth;side.classList?.add('focus');side.scrollIntoView?.({block:'start',behavior:reduceMotion()?'auto':'smooth'});return;}
 show(r.name);
 if(r.name==='ask'&&!busy&&!serverBusy)$('question').focus?.();
}
function openMenu(){$('app').dataset.menu='open';$('scrim').hidden=false;}
function closeMenu(){if($('app').dataset.menu){delete $('app').dataset.menu;$('scrim').hidden=true;}}

/* ---------- progress ---------- */
function progressSteps(progress){
 if(!progress||!Array.isArray(progress.steps))return [];
 return progress.steps.filter(step=>step&&typeof step.id==='string'&&typeof step.label==='string').map(step=>({
  ...step,status:['pending','running','complete','skipped','failed'].includes(step.status)?step.status:'pending',
  completed:Number.isSafeInteger(step.completed)&&step.completed>=0?step.completed:null,
  total:Number.isSafeInteger(step.total)&&step.total>=0?step.total:null,
  cached:Number.isSafeInteger(step.cached)&&step.cached>0?step.cached:0,
  message:typeof step.message==='string'?step.message:'',
 }));
}
function stepDetail(step){
 // Counts matter for rating stages; "1 / 1 completed" and "0 / 0" on skipped steps are noise.
 const counted=step.status!=='skipped'&&(step.id==='rate_passages'||step.id==='rate_summaries'||(step.total!==null&&step.total>1));
 const unit=step.id==='rate_passages'?'passages scored':step.id==='rate_summaries'?'summaries scored':'completed';
 const count=!counted?'':step.completed!==null&&step.total!==null?`${formatNumber(step.completed)} / ${formatNumber(step.total)} ${unit}`:step.completed!==null?`${formatNumber(step.completed)} ${unit}`:'';
 return [count,step.cached?`${formatNumber(step.cached)} reused from cache`:'',step.message].filter(Boolean).join('. ');
}
function updateProgressTime(){
 if(!lastJob||$('workflow').hidden&&screen!=='progress')return;
 const now=Date.now(),started=validDate(lastJob.progress?.started_at)??localStartedAt;
 const ended=validDate(lastJob.progress?.finished_at),finished=ended??validDate(lastJob.progress?.updated_at)??localFinishedAt??(lastStateAt||now);
 const running=lastJob.status==='running'&&ended===null;
 textIfChanged($('workflow-elapsed'),started===null?'':running?`Running for ${duration(now-started)}`:`Took ${duration(finished-started)}`);
 if(lastJob.status!=='running'){textIfChanged($('workflow-connection'),lastJob.status==='failed'?'Stopped after an error':lastJob.status==='stopped'?'Stopped on request':'Finished');return;}
 const age=lastStateAt?now-lastStateAt:0,updated=validDate(lastJob.progress?.updated_at);
 textIfChanged($('workflow-connection'),connectionLost?'Connection interrupted, retrying automatically':age>10000?'Waiting for a server update':lastStateAt?'Connected':'Waiting for the server');
 const stopping=progressSteps(lastJob.progress).some(step=>step.status==='failed');
 textIfChanged($('workflow-wait'),connectionLost?'The job may still be running. Your last confirmed progress is shown.':lastJob.stopping||stopRequested?'Stopping after the requests already in flight finish.':stopping?'Stopping after an error. Waiting for in-flight requests to finish.':updated&&now-updated>=10000?`No new progress reported for ${duration(now-updated)}. This step is still running.`:'');
}
function requestInfo(job){
 const request=job?.request&&typeof job.request==='object'?job.request:null;
 if(request)return request;
 if(lastSubmission&&job?.kind!=='upload')return {question:lastSubmission.question,answer:lastSubmission.answer,sources:lastSubmission.sources,settings:lastSubmission.settings};
 return null;
}
function depthName(settings){
 if(!settings||!Object.keys(defaults).length)return '';
 const match=Object.entries(presets()).find(([,values])=>Object.entries(values).every(([key,value])=>settings[key]===undefined||Number(settings[key])===value));
 return match?match[0]:'custom';
}
function runMeta(request,job){
 if(!request)return '';
 if(job.kind==='upload')return request.filename?'Extracting, summarizing where needed, and indexing the file.':'';
 const depth=depthName(request.settings),count=Array.isArray(request.sources)?request.sources.length:null;
 const scope=count===null?'':count===sources.length&&count>1?` across all ${count} collections`:count===1?` in ${sources.find(source=>source.id===request.sources[0])?.label||'1 collection'}`:` across ${count} collections`;
 const how=depth&&depth!=='custom'?`${depth[0].toUpperCase()+depth.slice(1)} depth`:'Custom depth';
 const model=request.settings?.llm_model||selectedModel;
 return `${how}${scope}. ${request.answer?(model?`Answer by ${model}.`:'Answer by the configured model.'):'Evidence only, no answer requested.'}`;
}
function renderProgress(job){
 if(!job||!['running','complete','failed','stopped'].includes(job.status)){
  // A state read that raced the submission must not replace its progress placeholder.
  if(submitting)return;
  lastJob=null;if(screen==='progress')show('ask');return;
 }
 const nextKey=typeof job.progress?.started_at==='string'?job.progress.started_at:null;
 if(nextKey&&nextKey!==jobKey){jobKey=nextKey;progressSuppressed=false;localFinishedAt=null;stopRequested=false;}
 if(job.status!=='running'&&localFinishedAt===null)localFinishedAt=Date.now();
 lastJob=job;
 const panel=$('workflow'),steps=progressSteps(job.progress),active=steps.find(step=>step.id===job.progress?.current_stage)||steps.find(step=>step.status==='running'||step.status==='failed');
 const stopped=job.status==='stopped',failed=job.status==='failed'||stopped||steps.some(step=>step.status==='failed'),terminal=job.status!=='running',percent=typeof job.progress?.percent==='number'&&Number.isFinite(job.progress.percent)?Math.max(0,Math.min(100,job.progress.percent)):job.status==='complete'?100:null;
 panel.dataset.state=stopped?'stopped':failed?'failed':job.status;panel.dataset.kind=job.kind==='upload'?'upload':'query';
 // Finished steps fold away; failures and imports keep their step list open.
 const stateKey=`${jobKey}|${failed?'failed':job.status}`;
 if(stateKey!==workflowStateKey){workflowStateKey=stateKey;$('workflow-details').open=job.status!=='complete'||failed;}
 const noun=job.kind==='upload'?'Import':'Inquiry',request=requestInfo(job);
 textIfChanged($('workflow-title'),job.status==='complete'?`${noun} complete`:failed?`${noun} stopped`:job.kind==='upload'?'Importing sources':'Building your inquiry');
 textIfChanged($('run-question'),job.kind==='upload'?(request?.filename?`Importing ${request.filename}`:'Importing a source'):(request?.question||'Your inquiry'));
 textIfChanged($('run-meta'),runMeta(request,job));
 const currentLabel=stopped?'Stopped on request':failed?'Stopped'+(active?`: ${active.label}`:''):job.status==='complete'?'All requested steps finished':active?active.label:'Preparing your request';
 textIfChanged($('workflow-current'),currentLabel);
 const finished=steps.filter(step=>step.status==='complete'||step.status==='skipped').length;
 textIfChanged($('workflow-summary'),steps.length?`${finished} of ${steps.length} steps finished${percent!==null?', '+Math.round(percent)+'%':''}`:terminal?(job.status==='complete'?'Complete':'Progress details unavailable'):'Waiting for step details');
 const bar=$('workflow-progress');
 if(percent===null)bar.removeAttribute('value');else bar.value=percent;
 bar.setAttribute('aria-valuetext',steps.length?`${finished} of ${steps.length} workflow steps finished. ${currentLabel}`:currentLabel);
 textIfChanged($('workflow-note'),steps.length?'Progress measures completed work, not time remaining.':job.status==='running'?'This server has not reported individual steps yet.':'');
 const names={pending:'Queued',running:'In progress',complete:'Done',skipped:'Skipped',failed:stopped?'Stopped':'Failed'};
 // Polls repeat unchanged steps; rebuilding them would reset text selection and screen-reader position.
 const stepsKey=JSON.stringify(steps.map(step=>[step.id,step.label,step.status,step.completed,step.total,step.cached,step.message]));
 if(stepsKey!==workflowStepsKey){workflowStepsKey=stepsKey;$('workflow-steps').replaceChildren();
 for(const step of steps){
  const item=document.createElement('li'),marker=document.createElement('span'),body=document.createElement('div'),title=document.createElement('span'),detail=document.createElement('small'),badge=document.createElement('span');
  item.dataset.state=step.status;if(step.status==='running')item.setAttribute('aria-current','step');marker.className='workflow-marker';marker.setAttribute('aria-hidden','true');marker.textContent=step.status==='complete'?'✓':step.status==='failed'?'!':'';
  title.className='workflow-step-title';title.textContent=step.label;detail.textContent=stepDetail(step);detail.hidden=!detail.textContent;body.append(title,detail);
  badge.className='workflow-badge';badge.textContent=names[step.status];item.append(marker,body,badge);$('workflow-steps').append(item);
 }}
 $('workflow-details').hidden=!steps.length;textIfChanged($('workflow-details-summary'),`${job.kind==='upload'?'Import':'Workflow'} steps (${steps.length})`);
 $('workflow-retry').hidden=!(failed&&terminal&&lastSubmission&&job.kind!=='upload');
 const canStop=job.status==='running'&&job.kind!=='upload'&&features.includes('cancel');
 $('stop').hidden=!canStop;$('stop').disabled=Boolean(job.stopping||stopRequested);textIfChanged($('stop-label'),job.stopping||stopRequested?'Stopping…':'Stop');
 const live=features.includes('live')&&job.kind!=='upload';
 $('live-fallback').hidden=live||job.kind==='upload';$('ratings').hidden=!live;$('strongest-panel').hidden=!live;
 renderStages(job,steps);
 const detail=active?stepDetail(active):'';
 textIfChanged($('workflow-announcement'),[currentLabel,detail].filter(Boolean).join('. '));
 if(terminal)$('workflow-wait').textContent='';
 updateProgressTime();
 if(live&&screen==='progress')ensureLive();
}
function stage(index){
 const list=$('stages');
 while(list.children.length<6){const item=make('li','stage'),node=make('span','node'),n=make('span','n'),l=make('span','l'),s=make('span','s');node.setAttribute('aria-hidden','true');item.append(node,n,l,s);list.append(item);}
 return list.children[index];
}
function setStage(index,state,number,small,label,sub,fill){
 const item=stage(index),[,n,l,s]=item.children;
 if(item.dataset.state!==state)item.dataset.state=state;
 if(changedKey(n,`${number}|${small}`)){n.replaceChildren(document.createTextNode(number));if(small)n.append(make('small','',small));}
 textIfChanged(l,label);textIfChanged(s,sub||'');s.hidden=!sub;
 if(item.style?.setProperty)item.style.setProperty('--fill',`${Math.round((fill||0)*100)}%`);
}
function renderStages(job,steps){
 const list=$('stages');list.hidden=job.kind==='upload';if(job.kind==='upload')return;
 const by=Object.fromEntries(steps.map(step=>[step.id,step])),request=requestInfo(job)||{},s=request.settings||{},live=lastLive&&lastLive.schema==='mori.live.v1'?lastLive:null;
 const state=(...ids)=>{const rows=ids.map(id=>by[id]).filter(Boolean);if(!rows.length)return 'todo';if(rows.some(row=>row.status==='failed'))return 'failed';if(rows.some(row=>row.status==='running'))return 'now';if(rows.every(row=>row.status==='complete'||row.status==='skipped'))return 'done';return rows.some(row=>row.status==='complete')?'now':'todo';};
 const chosen=Array.isArray(request.sources)?sources.filter(source=>request.sources.includes(source.id)):selectedAvailableSources();
 const scope=chosen.reduce((sum,source)=>sum+(Number(source.chunks)||0),0);
 const early=state('prepare','query_expansion','embed');
 setStage(0,early==='failed'?'failed':early==='done'?'done':'now',scope?formatNumber(scope):'All','',scope===1?'passage in scope':'passages in scope');
 const rp=by.rate_passages,rs=by.rate_summaries,matches=rp?.total??live?.passages?.total;
 setStage(1,state('retrieve'),formatNumber(matches??s.retrieval_k??''),'','closest matches found');
 const ratedState=state('rate_passages','rate_summaries'),done=rp?.completed??live?.passages?.rated??0,total=rp?.total??live?.passages?.total;
 const summaries=rs&&rs.status!=='skipped'&&rs.total?`and ${formatNumber(rs.completed??0)} of ${formatNumber(rs.total)} summaries`:'';
 setStage(2,ratedState,formatNumber(done),total?`of ${formatNumber(total)}`:'','rated by Jev',summaries,total?done/total:0);
 const shortlist=live?.shortlist??s.jev_k,shortState=ratedState==='done'?(by.rerank?.status==='running'?'now':'done'):ratedState==='failed'?'todo':'todo';
 setStage(3,state('rerank')==='failed'?'failed':shortState,formatNumber(shortlist??''),'',by.rerank&&by.rerank.status!=='skipped'&&by.rerank.status!=='pending'?'reranked for the shortlist':'kept for the shortlist');
 const evidence=live?.evidence,planned=(Number(s.full_units)||0)+(Number(s.summaries)||0)+(Number(s.rerank_enabled?s.rerank_k:s.final_passages)||0);
 const pieces=evidence?evidence.full_units+evidence.summaries+evidence.passages:planned;
 const breakdown=evidence?[plural(evidence.passages,'passage'),evidence.full_units?plural(evidence.full_units,'full source'):'',evidence.summaries?plural(evidence.summaries,'summary','summaries'):''].filter(Boolean).join(', '):s.evidence_mode==='context'?'as many as fit the context budget':'';
 setStage(4,state('assemble'),pieces?formatNumber(pieces):'',evidence||!pieces?'':'','pieces of evidence',breakdown);
 if(request.answer===false||(by.answer&&by.answer.status==='skipped'))setStage(5,'skipped','None','','evidence only, no answer requested');
 else{const answerState=state('answer','validate');setStage(5,answerState,'Answer','',by.validate?.status==='running'?'written, now checking every citation':answerState==='done'?'written and every citation checked':'written, then every citation checked');}
}

/* ---------- live ratings ---------- */
function ensureLive(){
 // Poll only a job the server has confirmed (or one final time after it ends), never the local placeholder.
 if(features.includes('live')&&!submitting&&liveTimer===null&&!liveBusy&&lastJob&&lastJob.kind!=='upload'&&(lastJob.status==='running'?serverBusy:!liveKeys.final))liveTimer=setTimeout(pollLive,0);
}
async function pollLive(){
 liveTimer=null;if(liveBusy||!features.includes('live'))return;liveBusy=true;let again=false;
 try{
  const value=await api('/api/live');
  if(submitting)return;
  if(value?.job){serverBusy=value.job.status==='running';renderProgress(value.job);}
  if(value?.live&&value.job?.progress?.started_at===jobKey){lastLive=value.live;renderLive(value.live,value.job);if(value.job.status!=='running')liveKeys.final=jobKey;}
  again=value?.job?.status==='running'&&screen==='progress';
 }catch{again=lastJob?.status==='running'&&screen==='progress';}
 finally{liveBusy=false;if(again)liveTimer=setTimeout(pollLive,600);}
}
function renderLive(live,job){
 const labelsList=Array.isArray(live.labels)&&live.labels.length>=2?live.labels:['Unrelated','Relevant'],L=labelsList.length;
 const levels=Array.isArray(live.passages?.levels)?live.passages.levels:[],counts=Array(L).fill(0);
 for(const level of levels)if(Number.isInteger(level)&&level>=0&&level<L)counts[level]++;
 const rated=counts.reduce((a,b)=>a+b,0),threshold=Math.ceil((L-1)*2/3),strong=counts.slice(threshold).reduce((a,b)=>a+b,0);
 const gridKey=`${jobKey}|${levels.length}|${rated}`;
 if(liveKeys.grid!==gridKey){liveKeys.grid=gridKey;drawGrid(levels,L);}
 $('grid').setAttribute('aria-label',levels.length?`${formatNumber(rated)} of ${plural(levels.length,'candidate')} rated. ${formatNumber(strong)} rate ${labelsList[threshold]} or better.`:'Ratings will appear as Jev rates candidates');
 const hist=$('hist');
 if(hist.children.length!==L||liveKeys.labels!==labelsList.join('|')){liveKeys.labels=labelsList.join('|');hist.replaceChildren();for(let level=L-1;level>=0;level--){const row=make('div','hist-row'),bar=make('b');row.append(make('span','',labelsList[level]),bar,make('em','','0'));row.dataset.level=String(level);if(bar.style)bar.style.background=RAMP[Math.round(level/(L-1)*9)];hist.append(row);}}
 const most=Math.max(1,...counts);
 for(const row of hist.children){const level=Number(row.dataset.level),bar=row.children[1];textIfChanged(row.children[2],formatNumber(counts[level]));row.dataset.empty=String(!counts[level]);if(bar.style)bar.style.width=`${Math.max(2,counts[level]/most*100)}%`;}
 const running=job?.status==='running',cap=requestInfo(job)?.settings,finalCap=cap?(cap.evidence_mode==='context'?null:cap.rerank_enabled?cap.rerank_k:cap.final_passages):null;
 const summary=$('rated-summary');
 const sentence=!levels.length?'':`${plural(strong,'passage')} rate ${labelsList[threshold]} or better${running?' so far':''}.`;
 const follow=!levels.length?'':finalCap?` The answer will draw on the best ${formatNumber(finalCap)}.`:' The answer will draw on as many of the best as fit the context budget.';
 const advice=running&&levels.length&&rated>=levels.length*.25&&strong<3?' If almost nothing reaches it, stop and rephrase rather than wait.':'';
 const key=sentence+follow+advice;
 if(changedKey(summary,key)){summary.replaceChildren();if(sentence){summary.append(make('b','',sentence),document.createTextNode(follow+advice));}}
 const strongest=Array.isArray(live.strongest)?live.strongest.slice(0,4):[],list=$('strongest');
 const strongKey=JSON.stringify(strongest.map(row=>[row.id,row.level,row.excerpt]));
 if(liveKeys.strongest!==strongKey){liveKeys.strongest=strongKey;list.replaceChildren();
  for(const row of strongest){const item=make('li'),top=make('div','top');top.append(make('span','level',row.label||labelsList[row.level]||''),seg(Number.isInteger(row.level)?row.level:null,L),make('span','src',row.locator||row.title||''));item.append(top,make('p','',row.excerpt||''));list.append(item);}}
 $('strongest-empty').hidden=strongest.length>0;
 textIfChanged($('strongest-live').lastChild||$('strongest-live'),running?'updates as Jev rates':'final ratings');
}
function drawGrid(levels,L){
 const canvas=$('grid');if(typeof canvas.getContext!=='function')return;
 const context=canvas.getContext('2d');if(!context)return;
 const n=levels.length,cols=n<=500?25:n<=1000?40:Math.ceil(Math.sqrt(n*1.6)),rows=Math.max(1,Math.ceil(n/cols));
 const width=Math.max(160,Math.min(409,canvas.parentElement?.clientWidth?canvas.parentElement.clientWidth-(narrow()?0:174):409));
 const gap=cols<=25?3.5:cols<=40?2.5:1.5,cell=Math.max(2,(width-(cols-1)*gap)/cols),height=rows*cell+(rows-1)*gap;
 const ratio=globalThis.devicePixelRatio||1;
 canvas.width=Math.round(width*ratio);canvas.height=Math.round(height*ratio);canvas.style.width=`${width}px`;canvas.style.height=`${height}px`;
 if($('hist').style)$('hist').style.minHeight=`${Math.max(180,height)}px`;
 context.setTransform(ratio,0,0,ratio,0,0);context.clearRect(0,0,width,height);
 const radius=Math.min(2.5,cell/4);
 for(let index=0;index<n;index++){
  const x=(index%cols)*(cell+gap),y=Math.floor(index/cols)*(cell+gap),level=levels[index];
  context.beginPath();
  if(context.roundRect)context.roundRect(x,y,cell,cell,radius);else context.rect(x,y,cell,cell);
  if(Number.isInteger(level)){context.fillStyle=RAMP[Math.round(level/Math.max(1,L-1)*9)];context.fill();}
  else{context.fillStyle='#f1f3ef';context.fill();context.strokeStyle='#e0e5de';context.lineWidth=1;context.stroke();}
 }
}
function beginProgress(kind){
 localStartedAt=Date.now();localFinishedAt=null;lastStateAt=0;jobKey=null;connectionLost=false;progressSuppressed=false;stopRequested=false;
 lastLive=null;liveKeys={};$('strongest').replaceChildren();$('hist').replaceChildren();$('rated-summary').replaceChildren();$('stages').replaceChildren();
 renderProgress({status:'running',kind});status(kind==='upload'?'Starting the import…':'');
 show('progress');navigate('#/progress');
}
if(typeof setInterval==='function')setInterval(updateProgressTime,1000);

/* ---------- answer rendering ---------- */
function showCitation(id){showTab('evidence');const target=document.querySelector(`[data-citation="${id}"]`);if(target){target.open=true;target.scrollIntoView({block:'center',behavior:reduceMotion()?'auto':'smooth'});}}
function appendInline(element,value,cites){
 for(const part of value.split(/(\*\*[^*]+\*\*|\[[UMP]\d+\])/g)){
  if(!part)continue;
  if(/^\[[UMP]\d+\]$/.test(part)){
   const id=part.slice(1,-1),button=document.createElement('button');button.type='button';button.className='citation';button.textContent=id;button.dataset.cite=id;button.dataset.kind=id[0];
   button.setAttribute('aria-label',`Show evidence ${id}`);
   button.addEventListener('click',()=>openReader(id));
   for(const name of ['mouseenter','focus'])button.addEventListener(name,()=>setActive(id,button));
   for(const name of ['mouseleave','blur'])button.addEventListener(name,()=>clearActive(id));
   element.append(button);cites?.push(id);
  }
  else if(part.length>4&&part.startsWith('**')&&part.endsWith('**')){const strong=document.createElement('strong');strong.textContent=part.slice(2,-2);element.append(strong);}
  else element.append(document.createTextNode(part));
 }
}
const listLine=/^\s*(?:[-*•]|(\d+)[.)])\s+(.*)$/;
function renderAnswer(text){
 // Headings, bullet and numbered lists, and single line breaks survive; all content is text, never HTML.
 const view=$('answer-view');view.replaceChildren();answerBlocks=[];let heading='';
 const block=(element,line,cites)=>{answerBlocks.push({element,cites,section:heading,line});};
 for(const chunk of text.replace(/\r\n?/g,'\n').split(/\n\s*\n/)){
  let lines=[],list=null;
  const flush=()=>{if(lines.length){const p=document.createElement('p'),cites=[];appendInline(p,lines.join('\n'),cites);view.append(p);block(p,lines.join(' '),cites);lines=[];}};
  for(const line of chunk.split('\n')){
   const head=/^(#{1,6})\s+(.*)$/.exec(line.trim()),item=listLine.exec(line);
   if(head){flush();list=null;const h=document.createElement(head[1].length<=3?'h3':'h4');appendInline(h,head[2]);view.append(h);heading=head[2].replace(/\*\*/g,'');}
   else if(item){flush();const ordered=item[1]!==undefined;if(!list||list.ordered!==ordered){list={ordered,element:document.createElement(ordered?'ol':'ul')};if(ordered&&item[1]!=='1')list.element.setAttribute('start',item[1]);view.append(list.element);}const li=document.createElement('li'),cites=[];appendInline(li,item[2],cites);list.element.append(li);block(li,item[2],cites);}
   else if(list&&/^\s+\S/.test(line)&&list.element.children.length){const li=list.element.children[list.element.children.length-1],entry=answerBlocks.find(row=>row.element===li);appendInline(li,' '+line.trim(),entry?.cites);if(entry)entry.line+=' '+line.trim();}
   else if(line.trim()){list=null;lines.push(line.trim());}
  }
  flush();
 }
}
function evidenceRows(context){
 const rows=[];
 for(const group of ['full_units','summaries','passages'])for(const row of Array.isArray(context?.[group])?context[group]:[])if(row&&typeof row.citation==='string')rows.push({group,row,kind:row.citation[0],identity:row.unit||row});
 return rows;
}
function kindName(kind){return kind==='U'?'Full source':kind==='M'?'Summary':'Passage';}
function excerptOf(entry,limit=240){
 const text=String(entry.kind==='M'?entry.row.summary||'':entry.kind==='U'?entry.row.text||'':entry.row.target?.text||'').replace(/\s+/g,' ').trim();
 return text.length>limit?text.slice(0,limit).replace(/\s+\S*$/,'')+'…':text;
}
function renderInsights(){
 const context=current?.context||{},answered=Boolean(current?.answer);
 citeIndex=new Map(evidenceRows(context).map(entry=>[entry.row.citation,entry]));
 citedOrder=[];const counts={};
 for(const entry of answerBlocks)for(const id of entry.cites){counts[id]=(counts[id]||0)+1;if(!citedOrder.includes(id)&&citeIndex.has(id))citedOrder.push(id);}
 const instances=answerBlocks.flatMap(entry=>entry.cites).filter(id=>citeIndex.has(id));
 const summaryOnly=answerBlocks.filter(entry=>entry.cites.length&&entry.cites.every(id=>id[0]==='M'));
 for(const entry of answerBlocks){if(entry.cites.length&&entry.cites.every(id=>id[0]==='M'))entry.element.dataset.sumonly='true';}
 const check=$('check');check.hidden=!answered||!instances.length;
 if(answered&&instances.length){
  const quoted=instances.filter(id=>id[0]!=='M').length,summarized=instances.length-quoted,text=$('check-text');
  text.replaceChildren(make('b','',instances.length===1?'The citation':`All ${formatNumber(instances.length)} citations`),document.createTextNode(` ${instances.length===1?'points':'point'} to evidence Mori supplied.`));
  const mix=$('check-mix');mix.replaceChildren();if(instances.length<=40)for(const id of [...instances].sort((a,b)=>(a[0]==='M')-(b[0]==='M'))){const square=make('i');square.dataset.kind=id[0];mix.append(square);}
  const keys=$('check-keys');keys.replaceChildren();
  const key=(kind,label)=>{const span=make('span'),square=make('i');square.dataset.kind=kind;span.append(square,document.createTextNode(label));keys.append(span);};
  if(quoted)key('P',`${formatNumber(quoted)} quote your sources`);if(summarized)key('M',`${formatNumber(summarized)} cite summaries`);
  $('mark-label').hidden=!summaryOnly.length;
  const toggleText=$('mark-label').lastChild;if(toggleText)toggleText.textContent=summaryOnly.length===1?'Mark the claim that rests only on summaries':`Mark the ${formatNumber(summaryOnly.length)} claims that rest only on summaries`;
 }
 $('mark-summaries').checked=markStore.get()!=='off';applyMark();
 renderNotes(counts);
 textIfChanged($('evidence-count'),citeIndex.size?formatNumber(citeIndex.size):'');
}
function applyMark(){const on=$('mark-summaries').checked&&!$('mark-label').hidden;if(on)$('answer-view').dataset.mark='on';else delete $('answer-view').dataset.mark;}
function renderNotes(counts){
 const box=$('notes');box.replaceChildren();
 if(!current?.answer||!citedOrder.length){box.hidden=true;return;}
 box.hidden=false;
 const head=make('div','notes-h'),all=make('button','link','All evidence');all.type='button';all.addEventListener('click',()=>showTab('evidence'));
 head.append(make('h2','','Cited evidence'),make('span','',`${formatNumber(citedOrder.length)} of ${plural(citeIndex.size,'item')}`),all);box.append(head);
 for(const id of citedOrder){
  const entry=citeIndex.get(id),note=make('button','note'),row1=make('span','row1'),row2=make('span','row2'),level=levelFor(entry.row.rating);
  note.type='button';note.dataset.cite=id;note.dataset.kind=entry.kind;note.setAttribute('aria-label',`${id}: ${entry.identity.title||''}. Open in the reader`);
  row1.append(make('span','cite-badge',id),make('span','t',entry.identity.title||entry.row.title||''));
  row2.append(make('span','kind',kindName(entry.kind)),make('span','loc',entry.identity.locator||''),make('span','grow'),seg(level,10,entry.kind==='M'?'M':''),make('span','',`cited ${counts[id]||1}×`));
  note.append(row1,row2,make('span','ex',excerptOf(entry)));
  note.addEventListener('click',()=>openReader(id));
  for(const name of ['mouseenter','focus'])note.addEventListener(name,()=>setActive(id,null));
  for(const name of ['mouseleave','blur'])note.addEventListener(name,()=>clearActive(id));
  box.append(note);
 }
 scheduleLayout();
}
let layoutFrame=null;
function scheduleLayout(){if(!browserLayout()||layoutFrame!==null||typeof requestAnimationFrame!=='function')return;layoutFrame=requestAnimationFrame(()=>{layoutFrame=null;layoutNotes();drawWire();});}
function firstBlock(id){return answerBlocks.find(entry=>entry.cites.includes(id))?.element||null;}
function layoutNotes(){
 const box=$('notes');if(!browserLayout()||box.hidden||screen!=='inquiry'||narrow()||$('answer-panel').dataset.reading)return;
 const base=box.getBoundingClientRect().top;let bottom=(box.firstChild?.offsetHeight||0)+4;
 for(const note of [...box.children].slice(1)){
  const block=firstBlock(note.dataset.cite),top=Math.max(block?block.getBoundingClientRect().top-base-2:bottom,bottom+8);
  note.style.top=`${top}px`;bottom=top+note.offsetHeight;
 }
 box.style.minHeight=`${bottom}px`;
}
let clearTimer=null;
function setActive(id,chip){
 if(clearTimer!==null){clearTimeout(clearTimer);clearTimer=null;}
 activeCite={id,chip};
 for(const element of document.querySelectorAll?.('.citation[data-active],.note[data-active]')||[])delete element.dataset.active;
 for(const element of document.querySelectorAll?.(`.citation[data-cite="${id}"],.note[data-cite="${id}"]`)||[])if(element.classList.contains('note')||element===chip)element.dataset.active='true';
 for(const entry of answerBlocks)if(entry.cites.includes(id)&&(!chip||entry.element.contains(chip)))entry.element.dataset.claim='on';else delete entry.element.dataset.claim;
 if(browserLayout()){layoutNotes();drawWire();}
}
function clearActive(id){
 if(readerCite)return;
 clearTimer=setTimeout(()=>{clearTimer=null;if(activeCite?.id!==id)return;activeCite=null;for(const element of document.querySelectorAll?.('[data-active]')||[])delete element.dataset.active;for(const entry of answerBlocks)delete entry.element.dataset.claim;drawWire();layoutNotes();},140);
}
function drawWire(){
 const wire=$('wire'),path=$('wire-path'),dot=$('wire-dot');
 if(!browserLayout()||!activeCite||narrow()||screen!=='inquiry'){path.setAttribute('d','');dot.setAttribute('r','0');return;}
 const layout=$('answer-panel').getBoundingClientRect(),id=activeCite.id,entry=citeIndex.get(id);
 const chip=activeCite.chip&&activeCite.chip.isConnected?activeCite.chip:document.querySelector(`#answer-view .citation[data-cite="${id}"]`);
 const target=readerCite?$('reader').querySelector('mark')||$('reader-title'):document.querySelector(`.note[data-cite="${id}"]`);
 if(!chip||!target||(readerCite&&readerCite!==id)){path.setAttribute('d','');dot.setAttribute('r','0');return;}
 const block=chip.closest('p,li')||chip,range=document.createRange();range.selectNodeContents(block);
 const rects=[...range.getClientRects()],chipBox=chip.getBoundingClientRect(),line=rects.filter(r=>Math.abs(r.top-chipBox.top)<chipBox.height).pop()||chipBox;
 const sx=Math.max(line.right,chipBox.right)+6-layout.left,sy=chipBox.top+chipBox.height/2-layout.top;
 const tb=readerCite?$('reader').getBoundingClientRect():target.getBoundingClientRect();
 const ex=tb.left-1-layout.left,ey=Math.max(14,Math.min((readerCite?target.getBoundingClientRect().top+14:tb.top+19)-layout.top,(tb.bottom-layout.top)-8));
 const bend=$('answer-view').getBoundingClientRect().right+20-layout.left,dir=Math.sign(ey-sy)||1,r=Math.min(6,Math.abs(ey-sy)/2);
 if(ex<=bend){path.setAttribute('d','');dot.setAttribute('r','0');return;}
 path.setAttribute('d',`M${sx} ${sy} H${bend-r} Q${bend} ${sy} ${bend} ${sy+dir*r} V${ey-dir*r} Q${bend} ${ey} ${bend+r} ${ey} H${ex}`);
 dot.setAttribute('cx',sx);dot.setAttribute('cy',sy);dot.setAttribute('r','2.4');wire.dataset.kind=entry?.kind==='M'?'M':'';
}

/* ---------- source reader ---------- */
function claimsFor(id){return answerBlocks.filter(entry=>entry.cites.includes(id));}
function plainClaim(entry){return entry.line.replace(/\[[UMP]\d+\]/g,'').replace(/\*\*/g,'').replace(/\s+([.,;:])/g,'$1').replace(/\s+/g,' ').trim();}
async function openReader(id,{updateHash=true}={}){
 if(!current)return;const entry=citeIndex.get(id);if(!entry){showCitation(id);return;}
 readerCite=id;const run=current.context.run_id;
 if(screen!=='inquiry')show('inquiry');
 showTab('answer');
 $('reader').hidden=false;$('answer-panel').dataset.reading='true';
 const order=citedOrder.length?citedOrder:[...citeIndex.keys()],position=order.indexOf(id);
 textIfChanged($('reader-position'),position>=0?`${citedOrder.length?'Citation':'Item'} ${position+1} of ${order.length}`:'');
 $('reader-prev').disabled=position<=0;$('reader-next').disabled=position<0||position>=order.length-1;
 const kind=$('reader-kind');kind.replaceChildren(make('span','cite-badge',id),document.createTextNode(entry.kind==='M'?'Summary of a source, not its own words':entry.kind==='U'?'Complete source, given to the model in full':'Passage, quoted exactly from the source'));
 if(kind.firstChild?.dataset)kind.firstChild.dataset.kind=entry.kind;
 textIfChanged($('reader-title'),entry.identity.title||entry.row.title||id);
 const meta=$('reader-meta');meta.replaceChildren();
 const fact=(label,value)=>{const row=make('div');const dd=make('dd');if(value instanceof Object&&value.nodeType)dd.append(value);else dd.textContent=value;row.append(make('dt','',label),dd);meta.append(row);};
 const sourceLabel=sources.find(source=>source.id===entry.identity.source_id)?.label;
 if(sourceLabel)fact('Collection',sourceLabel);
 fact('Location',entry.identity.locator||'Not recorded');
 const level=levelFor(entry.row.rating),rating=make('span');rating.append(seg(level,10,entry.kind==='M'?'M':''),document.createTextNode(Number.isFinite(entry.row.rating?.score)?` ${shortLabel(entry.row.rating.label)||''} ${entry.row.rating.score.toFixed(1)}/100`.replace(/\s+/g,' '):' Not rated'));
 fact('Jev rating',rating);
 const claims=current.answer?claimsFor(id):[];
 $('reader-cited').hidden=!claims.length;
 textIfChanged($('reader-cited-title'),claims.length===1?'Cited for 1 claim in this answer':`Cited for ${formatNumber(claims.length)} claims in this answer`);
 const list=$('reader-claims');list.replaceChildren();
 for(const claim of claims){const quote=make('blockquote','',`“${plainClaim(claim)}”`);quote.append(make('small','',claim.section||'Answer'));list.append(quote);}
 const canJudge=features.includes('verdicts')&&Boolean(current.answer)&&claims.length>0;$('verdict').hidden=!canJudge;
 textIfChanged($('verdict-question'),claims.length===1?'Does this support the claim?':'Does this support them?');
 updateVerdictButtons();
 $('reader-full').hidden=true;
 await renderSourceText(entry);
 if(updateHash)navigate(`#/inquiry/${encodeURIComponent(run)}/${id}`);
 activeCite={id,chip:null};
 for(const element of document.querySelectorAll?.('.citation[data-active],.note[data-active]')||[])delete element.dataset.active;
 for(const element of document.querySelectorAll?.(`.citation[data-cite="${id}"]`)||[])element.dataset.active='true';
 for(const block of answerBlocks)if(block.cites.includes(id))block.element.dataset.claim='on';else delete block.element.dataset.claim;
 scheduleLayout();
}
function closeReader(updateHash=true){
 if(!readerCite)return;readerCite=null;activeCite=null;$('reader').hidden=true;delete $('answer-panel').dataset.reading;
 for(const element of document.querySelectorAll?.('[data-active]')||[])delete element.dataset.active;for(const entry of answerBlocks)delete entry.element.dataset.claim;
 if(updateHash&&current)navigate(`#/inquiry/${encodeURIComponent(current.context.run_id)}`);
 scheduleLayout();
}
function textWithMark(parent,text,start,end,before=''){
 parent.replaceChildren();
 if(before)parent.append(make('span','ctx',before));
 if(!(Number.isInteger(start)&&Number.isInteger(end)&&0<=start&&start<end&&end<=text.length)){parent.append(document.createTextNode(text));return null;}
 const mark=make('mark','',text.slice(start,end));
 parent.append(make('span','ctx',text.slice(0,start)),mark,make('span','ctx',text.slice(end)));
 return mark;
}
async function unitText(identity){
 if(!features.includes('unit_text')||!identity?.id)return null;
 if(unitCache.has(identity.id))return unitCache.get(identity.id);
 try{const unit=await api('/api/unit?id='+encodeURIComponent(identity.id));if(typeof unit?.text!=='string')return null;if(unitCache.size>24)unitCache.delete(unitCache.keys().next().value);unitCache.set(identity.id,unit);return unit;}catch{return null;}
}
async function renderSourceText(entry){
 const box=$('reader-text'),id=entry.row.citation,label=$('reader-source-label');box.replaceChildren();
 if(entry.kind==='M'){
  label.textContent='Summary';
  box.append(make('span','summary-text',entry.row.summary||''),make('p','reader-note','Summaries describe a source; Mori stores one for every source unit. Quote the source itself, not the summary.'));revealMark(null);
  const full=$('reader-full');full.hidden=!features.includes('unit_text');
  full.onclick=async()=>{full.disabled=true;const unit=await unitText(entry.identity);full.disabled=false;if(readerCite!==id)return;if(unit){full.hidden=true;label.textContent='Complete source';const text=make('div');textWithMark(text,unit.text);box.replaceChildren(text);}else status('The complete source could not be loaded in this room.','error');};
  return;
 }
 if(entry.kind==='U'){label.textContent='The complete source';box.append(document.createTextNode(entry.row.text||''));revealMark(null);return;}
 label.textContent='In the source';
 const target=entry.row.target||{},unit=await unitText(entry.row.unit);
 if(readerCite!==id)return;
 let mark;
 if(unit&&unit.text.slice(target.start,target.end)===target.text)mark=textWithMark(box,unit.text,target.start,target.end);
 else{
  const context=entry.row.context;
  if(context&&typeof context.text==='string'&&Number.isInteger(context.start)&&context.text.slice(target.start-context.start,target.end-context.start)===target.text)mark=textWithMark(box,context.text,target.start-context.start,target.end-context.start);
  else mark=textWithMark(box,target.text||'',0,(target.text||'').length);
  if(!features.includes('unit_text'))box.append(make('p','reader-note','Restart this room to read the passage inside its complete source.'));
 }
 revealMark(mark);
}
function revealMark(mark){
 // Start at the top so the title and facts stay visible; scroll only when the marked passage would be out of view.
 const reader=$('reader');if(!browserLayout()||!reader)return;
 if(!mark){reader.scrollTop=0;return;}
 const top=mark.offsetTop,view=reader.clientHeight;
 reader.scrollTop=top+Math.min(mark.offsetHeight,120)<=view-32?0:Math.max(0,top-Math.min(160,Math.round(view/3)));
}
function updateVerdictButtons(){
 const value=readerCite?current?.verdicts?.[readerCite]:null;
 for(const button of $('verdict').querySelectorAll?.('[data-verdict]')||[])button.setAttribute('aria-pressed',String(button.dataset.verdict===value));
}
async function saveVerdict(choice){
 if(!current||!readerCite||!features.includes('verdicts'))return;
 const id=readerCite,run=current.context.run_id,next=current.verdicts?.[id]===choice?null:choice;
 try{
  const value=await api(`/api/runs/${encodeURIComponent(run)}/verdicts`,{citation:id,verdict:next});
  current.verdicts=value.verdicts||{};updateVerdictButtons();
  const row=historyRuns.find(item=>item.id===run);
  if(row&&Number.isSafeInteger(row.cited)){row.checked=citedOrder.filter(cite=>current.verdicts[cite]).length;renderLibrary();}
  status(next?`Saved: ${next==='yes'?'supports':next==='partly'?'partly supports':'does not support'} the claim.`:'Check cleared.');
 }catch(e){status(e.message,'error');}
}
for(const button of $('verdict').querySelectorAll?.('[data-verdict]')||[])button.addEventListener('click',()=>saveVerdict(button.dataset.verdict));
function stepReader(delta){const order=citedOrder.length?citedOrder:[...citeIndex.keys()],index=order.indexOf(readerCite);if(index>=0&&order[index+delta])openReader(order[index+delta]);}
$('reader-prev').addEventListener('click',()=>stepReader(-1));
$('reader-next').addEventListener('click',()=>stepReader(1));
$('reader-close').addEventListener('click',()=>closeReader());
$('reader-copy').addEventListener('click',()=>{
 const entry=readerCite&&citeIndex.get(readerCite);if(!entry)return;
 const text=entry.kind==='M'?entry.row.summary:entry.kind==='U'?entry.row.text:entry.row.target?.text;
 copyText($('reader-copy').firstChild||$('reader-copy'),`“${String(text||'').trim()}” (${[entry.identity.title,entry.identity.locator].filter(Boolean).join(', ')})`,'Copy quote');
});
$('mark-summaries').addEventListener('change',()=>{markStore.set($('mark-summaries').checked?'':'off');applyMark();});

/* ---------- requests ---------- */
async function api(path,data){
 const controller=typeof AbortController==='function'?new AbortController():null;
 const timer=controller?setTimeout(()=>controller.abort(),15000):null;
 try{
  const options=data?{method:'POST',headers:{'Content-Type':'application/json','X-Mori-Token':document.querySelector('meta[name=mori-token]').content},body:JSON.stringify(data)}:{};
  if(controller)options.signal=controller.signal;
  const r=await fetch(path,options),v=await r.json();
  if(!r.ok){const error=Error(v.error||'Request failed');error.responseReceived=true;error.httpStatus=r.status;throw error;}return v;
 }catch(e){
  if(e.name==='AbortError')throw Error('The server took too long to respond');
  if(e.name==='TypeError'&&!e.responseReceived)throw Error('Mori is not responding. Check that the research room is still running');
  if(e.name==='SyntaxError')throw Error('The server sent an unreadable response');
  throw e;
 }
 finally{if(timer!==null)clearTimeout(timer);}
}

/* ---------- composer ---------- */
function modelError(){return llmEnabled&&selectedModel&&!modelOptions.includes(selectedModel)?`The selected model (${selectedModel}) is unavailable. Choose an available answer model.`:'';}
function extensionError(){return promptExtension&&!llmEnabled?'Prompt extension requires a configured LLM. Turn it off in Settings to continue.':'';}
function rerankError(){return $('rerank_enabled').checked&&!llmEnabled?'Reranking requires a configured LLM. Turn it off in Settings to continue.':'';}
$('rerank_enabled').addEventListener('change',()=>{syncDepth();updateControls();});
$('query_expansion').addEventListener('change',()=>{if(busy||serverBusy||loadingRun||!sourcesReady||(!llmEnabled&&$('query_expansion').checked)){$('query_expansion').checked=promptExtension;return;}promptExtension=$('query_expansion').checked;updateControls();});
function syncModels(providers){
 llmEnabled=providers.llm!=='none';
 const configured=typeof providers.model==='string'?providers.model:'';
 modelOptions=[...new Set([configured,...(Array.isArray(providers.models)?providers.models:[])].filter(model=>typeof model==='string'&&model.length))];
 if(selectedModel===null||!selectedModel)selectedModel=configured;
 const key=JSON.stringify([llmEnabled,configured,modelOptions,modelError()]);
 if(key!==modelCatalogKey){
  modelCatalogKey=key;const select=$('llm_model');select.replaceChildren();
  const option=(value,label,disabled=false)=>{const item=document.createElement('option');item.value=value;item.textContent=label;item.disabled=disabled;select.append(item);};
  if(!llmEnabled)option('','No answer model');
  else{
   if(!configured)option('','Provider default');
   for(const model of modelOptions)option(model,model);
   if(modelError())option(selectedModel,`${selectedModel} (unavailable)`,true);
  }
  select.value=llmEnabled?selectedModel:'';
 }
 textIfChanged($('model-help'),llmEnabled?'Used for answers and optional reranking.':'This room has no answer model. Gather evidence and use Model handoff.');
 updateControls();
}
$('llm_model').addEventListener('change',()=>{if(busy||serverBusy||loadingRun||!sourcesReady)return;selectedModel=$('llm_model').value;updateControls();});
function contextMode(){return $('evidence_mode').value==='context';}
function budgetError(){
 if(!contextMode())return '';
 const limit=Number($('context_window_tokens').value),output=Number($('llm_max_output_tokens').value);
 if(!Number.isSafeInteger(limit)||limit<1||limit>2000000)return 'Choose a whole-number context window between 1 and 2,000,000 tokens.';
 if(!Number.isSafeInteger(output)||output<1)return 'Reserve at least one whole token for the answer.';
 return limit<=output+budgetOverhead?'The context window must leave room for evidence after the answer and overhead reserves.':'';
}
function updateBudgetControls(locked){
 const context=contextMode();$('context-controls').hidden=!context;
 for(const id of ['evidence_mode','context_window_tokens','llm_max_output_tokens'])$(id).disabled=locked;
 for(const key of ['final_passages','rerank_k']){if($(key))$(key).disabled=locked||context;if($(key+'-field'))$(key+'-field').hidden=context;}
 $('budget-help').textContent=context?`Reserve the complete source units and the most relevant summaries first, then fill the remaining space with whole passages from the Jev shortlist. Prompt text is included, with ${formatNumber(budgetOverhead)} tokens reserved for provider overhead. Each UTF-8 byte counts as one token: this conservative estimate can leave substantial spare capacity. Server character limits also apply; provider usage can differ.`:'Use the passage counts below. Full units and summaries are selected separately.';
}
for(const id of ['evidence_mode','context_window_tokens','llm_max_output_tokens'])$(id).addEventListener(id==='evidence_mode'?'change':'input',()=>{syncDepth();updateControls();});
const zeroAllowed=['full_units','summaries','summary_candidates','rating_neighbors','answer_neighbors'],fieldLimits={retrieval_k:10000,rating_neighbors:10,answer_neighbors:10};
function fieldError(){
 // Mirrors the server contract in the words used on screen, before anything is sent.
 if(!Object.keys(defaults).length)return '';
 const context=contextMode(),value=key=>Number($(key).value);
 for(const key of Object.keys(labels)){
  if(context&&['final_passages','rerank_k'].includes(key))continue;
  const raw=String($(key).value??'').trim(),number=Number(raw);
  if(key.includes('relevance')){if(!raw||!Number.isFinite(number)||number<0||number>100)return `${labels[key]} must be between 0 and 100.`;continue;}
  const minimum=zeroAllowed.includes(key)?0:1;
  if(!raw||!Number.isSafeInteger(number)||number<minimum)return `${labels[key]} must be a whole number of at least ${minimum}.`;
  if(fieldLimits[key]&&number>fieldLimits[key])return `${labels[key]} cannot exceed ${formatNumber(fieldLimits[key])}.`;
 }
 if(value('jev_k')>value('retrieval_k'))return `${labels.jev_k} cannot be larger than ${labels.retrieval_k}.`;
 const cap=$('rerank_enabled').checked?'rerank_k':'final_passages';
 if(!context&&value(cap)>value('jev_k'))return `${labels[cap]} cannot be larger than ${labels.jev_k}.`;
 return '';
}
function settings(){const result={query_expansion:promptExtension};for(const key of Object.keys(labels)){if(contextMode()&&['final_passages','rerank_k'].includes(key))continue;result[key]=Number($(key).value);}result.rerank_enabled=$('rerank_enabled').checked;result.exclude_full_unit_passages=$('exclude_full_unit_passages').checked;result.evidence_mode=contextMode()?'context':'fixed';if(contextMode()){result.context_window_tokens=Number($('context_window_tokens').value);result.llm_max_output_tokens=Number($('llm_max_output_tokens').value);}if(llmEnabled&&selectedModel)result.llm_model=selectedModel;return result;}
/* Depth presets fill the existing numbers; Standard is this room's configured defaults. */
const DEPTH_KEYS=['retrieval_k','jev_k','final_passages','rerank_k','full_units','summaries','summary_candidates'];
function presets(){
 const d=defaults,n=key=>Number(d[key])||0,scale=(key,factor,minimum)=>Math.max(minimum,Math.round(n(key)*factor));
 const shape=values=>{values.retrieval_k=Math.min(10000,values.retrieval_k);values.jev_k=Math.min(values.jev_k,values.retrieval_k);values.final_passages=Math.min(values.final_passages,values.jev_k);values.rerank_k=Math.min(values.rerank_k,values.jev_k);return values;};
 const standard={retrieval_k:n('retrieval_k'),jev_k:n('jev_k'),final_passages:n('final_passages'),rerank_k:n('rerank_k'),full_units:n('full_units'),summaries:n('summaries'),summary_candidates:n('summary_candidates')};
 const quick=shape({retrieval_k:scale('retrieval_k',.4,10),jev_k:scale('jev_k',.5,5),final_passages:scale('final_passages',.5,3),rerank_k:scale('rerank_k',.5,3),full_units:Math.min(n('full_units'),1),summaries:scale('summaries',.6,0),summary_candidates:n('summary_candidates')});
 const deep=shape({retrieval_k:Math.min(10000,n('retrieval_k')*2),jev_k:n('jev_k')*2,final_passages:n('final_passages')*2,rerank_k:n('rerank_k')*2,full_units:n('full_units')+1,summaries:Math.round(n('summaries')*1.6),summary_candidates:n('summary_candidates')});
 // Stay within the room's rating budget: cap how many summaries Jev rates, as the server would require.
 if(Number.isSafeInteger(maxRatingCalls)){const units=sources.reduce((sum,source)=>sum+(Number(source.units)||0),0),chunks=sources.reduce((sum,source)=>sum+(Number(source.chunks)||0),0);
  for(const values of [quick,deep]){const passages=Math.min(values.retrieval_k,chunks||values.retrieval_k),summaries=values.full_units||values.summaries?Math.min(values.summary_candidates||units,units):0;
   if(passages+summaries>maxRatingCalls&&values.full_units+values.summaries>0)values.summary_candidates=Math.max(values.full_units+values.summaries,maxRatingCalls-passages);}}
 return {quick,standard,deep};
}
function depthText(values){
 const context=contextMode(),cap=$('rerank_enabled').checked?values.rerank_k:values.final_passages;
 const extras=[values.full_units?(values.full_units===1?'one source read in full':`${formatNumber(values.full_units)} sources read in full`):'',values.summaries?plural(values.summaries,'summary','summaries'):''].filter(Boolean);
 const best=context?`as many of the best ${formatNumber(values.jev_k)} as fit the context budget`:`the best ${formatNumber(cap)}`;
 return `Jev rates the ${formatNumber(values.retrieval_k)} closest passages. The answer draws on ${best}${extras.length?(extras.length>1||context?', ':' and ')+extras.join(' and '):''}.`;
}
let applyingDepth=false;
function applyDepth(name){
 const values=presets()[name];if(!values)return;applyingDepth=true;
 for(const key of DEPTH_KEYS)if($(key))$(key).value=values[key];
 applyingDepth=false;syncDepth();updateControls();
}
function syncDepth(){
 if(!Object.keys(defaults).length)return;
 const all=presets(),values=Object.fromEntries(DEPTH_KEYS.map(key=>[key,Number($(key)?.value)]));
 const match=Object.entries(all).find(([,preset])=>DEPTH_KEYS.every(key=>(contextMode()&&['final_passages','rerank_k'].includes(key))||preset[key]===values[key]));
 for(const name of ['quick','standard','deep']){$('depth-'+name).checked=match?.[0]===name;textIfChanged($('depth-'+name+'-text'),depthText(all[name]));}
 $('depth-custom').hidden=Boolean(match);
}
for(const name of ['quick','standard','deep'])$('depth-'+name).addEventListener('change',()=>{if(busy||serverBusy||loadingRun||!sourcesReady){syncDepth();return;}if($('depth-'+name).checked)applyDepth(name);});
$('depth-reset').addEventListener('click',()=>applyDepth('standard'));
function costText(){
 if(!sourcesReady||!Object.keys(defaults).length||!sources.length)return '';
 const selected=selectedAvailableSources();if(!selected.length)return 'Choose at least one collection to search.';
 const chunks=selected.reduce((sum,source)=>sum+(Number(source.chunks)||0),0),units=selected.reduce((sum,source)=>sum+(Number(source.units)||0),0),value=key=>Number($(key)?.value)||0;
 const passages=Math.min(value('retrieval_k'),chunks),summaries=value('full_units')||value('summaries')?Math.min(value('summary_candidates')||units,units):0;
 const scope=summaries?(summaries===units?(units===1?' and the summary of the one source in scope':` and the summaries of all ${formatNumber(units)} sources in scope`):` and the summaries of ${formatNumber(summaries)} of the ${formatNumber(units)} sources in scope`):'';
 const extension=promptExtension?' Prompt extension adds one model call.':'';
 const answer=llmEnabled?` Then ${selectedModel||'the answer model'} writes one answer.`:' Then copy the evidence into your own model.';
 const limit=Number.isSafeInteger(maxRatingCalls)&&passages+summaries>maxRatingCalls?` That is more than this room’s limit of ${formatNumber(maxRatingCalls)} ratings; lower the depth or search fewer collections.`:'';
 return `Jev will rate ${plural(passages,'passage')}${scope}.${extension}${answer}${limit}`;
}
function selectedAvailableSources(){return sources.filter(source=>selectedSources.has(source.id));}
function unavailableSources(){const available=new Set(sources.map(source=>source.id));return [...selectedSources].filter(id=>!available.has(id));}
function selectionError(){const missing=unavailableSources();return missing.length?`Unavailable sources: ${missing.map(id=>id||'(empty source ID)').join(', ')}. Choose available collections or select all to continue.`:'';}
function updateControls(){
 const locked=busy||serverBusy||loadingRun||!sourcesReady,selected=selectedAvailableSources(),error=selectionError(),modelIssue=modelError()||extensionError()||rerankError(),budgetIssue=budgetError()||fieldError();
 $('query_expansion').checked=promptExtension;$('query_expansion').disabled=locked||(!llmEnabled&&!promptExtension);
 $('rerank_enabled').disabled=!llmEnabled&&!$('rerank_enabled').checked;
 $('composer').setAttribute('aria-busy',String(busy||serverBusy||loadingRun));
 $('source-picker').disabled=locked;
 const blocked=locked||!selected.length||Boolean(error)||Boolean(modelIssue)||Boolean(budgetIssue);
 for(const id of ['retrieve','plan'])$(id).disabled=blocked;
 // Answering cannot succeed without a model: offer the evidence handoff instead of a failing button.
 $('answer').disabled=blocked||!llmEnabled;
 $('answer-hint').hidden=llmEnabled||!sourcesReady||!sources.length;
 for(const name of ['quick','standard','deep'])$('depth-'+name).disabled=locked;$('depth-reset').disabled=locked;
 updateBudgetControls(locked);textIfChanged($('budget-error'),budgetIssue);$('budget-error').hidden=!budgetIssue;
 $('llm_model').disabled=locked||!llmEnabled||(!modelOptions.length&&!modelIssue);
 textIfChanged($('model-error'),modelIssue);$('model-error').hidden=!modelIssue;
 $('upload').disabled=locked||!allowUploads;
 $('summarize').disabled=!llmEnabled&&!$('summarize').checked;
 $('sources-all').disabled=locked||!sources.length||(!error&&selected.length===sources.length);
 $('sources-none').disabled=locked||!selectedSources.size;
 textIfChanged($('source-error'),error);$('source-error').hidden=!error;
 textIfChanged($('source-count'),!sourcesReady?'Loading…':!sources.length?'No sources available':!selected.length?'None selected':`${formatNumber(selected.reduce((sum,source)=>sum+source.chunks,0))} passages selected`);
 textIfChanged($('source-summary'),!sourcesReady?'Loading sources…':!sources.length?(allowUploads?'No sources available. Import a source to begin.':'No sources are available in this corpus.'):`${selected.length} of ${plural(sources.length,'collection')} selected: ${plural(selected.reduce((sum,source)=>sum+source.units,0),'source unit')}, ${plural(selected.reduce((sum,source)=>sum+source.chunks,0),'passage')}.${selected.length?'':' Choose at least one collection.'}`);
 const emptyCorpus=sourcesReady&&!sources.length;
 $('empty-corpus').hidden=!emptyCorpus;
 textIfChanged($('empty-corpus'),allowUploads?'This room has no sources yet. Open Add sources in Sources to import your first one.':'This research room has no sources yet.');
 if(emptyCorpus&&allowUploads&&!emptyCorpusOpened){emptyCorpusOpened=true;$('add-sources').open=true;}
 textIfChanged($('cost'),costText());
}
function working(value){busy=value;updateControls();}
function renderSources(){
 // Rebuild only when the catalog or selection changed: polling must not reset focus or hover.
 const key=JSON.stringify([sources.map(source=>[source.id,source.label,source.units,source.chunks]),[...selectedSources]]);
 if(key===sourcesKey){updateControls();return;}
 sourcesKey=key;$('source-options').replaceChildren();
 const most=Math.max(1,...sources.map(source=>Number(source.chunks)||0));
 for(const source of sources){
  const label=document.createElement('label'),input=document.createElement('input'),description=document.createElement('span'),name=document.createElement('span'),counts=document.createElement('small'),number=make('span','n',formatNumber(source.chunks)),bar=make('span','bar'),fill=make('b');
  label.className='source-option';input.type='checkbox';input.value=source.id;input.checked=selectedSources.has(source.id);
  name.className='name';name.textContent=source.label;counts.textContent=plural(source.units,'source unit');description.append(name,counts);
  if(fill.style)fill.style.width=`${Math.max(2,(Number(source.chunks)||0)/most*100)}%`;bar.append(fill);
  label.append(input,description,number,bar);
  input.addEventListener('change',()=>{
   if(busy||serverBusy||loadingRun||!sourcesReady)return;
   // Changing the selection explicitly discards unavailable URL or historical IDs.
   selectedSources=new Set(selectedAvailableSources().map(row=>row.id));selectAllSources=false;
   if(input.checked)selectedSources.add(source.id);else selectedSources.delete(source.id);
   updateControls();
  });
  $('source-options').append(label);
 }
 syncDepth();updateControls();
}
function syncSources(next){
 if(!Array.isArray(next)){sourcesReady=false;updateControls();throw Error('The source list is unavailable. Reload before submitting a question.');}
 sources=next;
 if(!selectionInitialized&&requestedSources!==null){selectedSources=new Set(requestedSources);selectAllSources=false;}
 else if(selectAllSources)selectedSources=new Set(sources.map(source=>source.id));
 sourcesReady=true;selectionInitialized=true;renderSources();renderCollections();
}
function querySources(){
 if(!sourcesReady)throw Error('Sources are still loading.');
 if(busy||serverBusy||loadingRun)throw Error('Wait for the current operation to finish.');
 const error=selectionError();if(error)throw Error(error);
 const modelIssue=modelError()||extensionError()||rerankError();if(modelIssue)throw Error(modelIssue);
 const budgetIssue=budgetError()||fieldError();if(budgetIssue)throw Error(budgetIssue);
 const ids=selectedAvailableSources().map(source=>source.id);if(!ids.length)throw Error('Choose at least one source.');return ids;
}
for(const [id,all]of [['sources-all',true],['sources-none',false]])$(id).addEventListener('click',()=>{
 if(busy||serverBusy||loadingRun||!sourcesReady)return;
 selectAllSources=all;selectedSources=new Set(all?sources.map(source=>source.id):[]);renderSources();
});

/* ---------- room, library and recent ---------- */
function renderRoom(s){
 const units=s.stats?.units??0,chunks=s.stats?.chunks??0;
 textIfChanged($('stats'),`${plural(chunks,'passage')} in ${plural(units,'source unit')}`);
 const providers=$('providers'),key=JSON.stringify([s.providers?.embedding,s.providers?.llm,selectedModel]);
 if(changedKey(providers,key)){providers.replaceChildren();
  if(s.providers?.llm==='none')providers.append(document.createTextNode('No answer model: evidence handoff'));
  else providers.append(document.createTextNode('Answers by '),make('b','',selectedModel||'the configured model'));
  if(s.providers?.embedding==='demo')providers.append(make('span','warn',' Demo embeddings, not semantic'));}
 textIfChanged($('room'),!sources.length?'Empty research room':sources.length===1?sources[0].label:`${sources.length} collections`);
 textIfChanged($('nav-sources-count'),sources.length?formatNumber(sources.length):'');
 const facts=$('room-facts'),factKey=JSON.stringify([units,chunks,s.providers,allowUploads,features]);
 if(changedKey(facts,factKey)){facts.replaceChildren();
  const fact=(label,value)=>facts.append(make('dt','',label),make('dd','',value));
  fact('Corpus',`${plural(chunks,'passage')} in ${plural(units,'source unit')} and ${plural(sources.length,'collection')}`);
  fact('Embeddings',s.providers?.embedding==='demo'?'Demo fixture: matches words, not meaning':String(s.providers?.embedding||'Unknown'));
  fact('Relevance ratings','Jev');
  fact('Answer model',s.providers?.llm==='none'?'None configured: gather evidence and use Model handoff':`${s.providers?.model||'Provider default'} (${s.providers?.llm})`);
  fact('Importing',allowUploads?'Allowed':'Turned off in this room');
  fact('Live ratings and Stop',features.includes('live')?'Available':'Restart this room to turn them on');
  fact('Privacy','The room runs on this computer. Sources go only to the providers configured here.');}
}
function renderCollections(){
 const key=JSON.stringify([sources,allowUploads]);if(key===collectionsKey)return;collectionsKey=key;
 const box=$('collections');box.replaceChildren();const most=Math.max(1,...sources.map(source=>Number(source.chunks)||0));
 const units=sources.reduce((sum,source)=>sum+(Number(source.units)||0),0),chunks=sources.reduce((sum,source)=>sum+(Number(source.chunks)||0),0);
 textIfChanged($('library-total'),sources.length?`${plural(chunks,'passage')} from ${plural(units,'source unit')} in ${plural(sources.length,'collection')}`:'No sources yet.');
 for(const source of sources){const row=make('div','coll'),r1=make('div','r1'),bar=make('div','bar'),fill=make('b');r1.append(make('span','',source.label),make('span','',formatNumber(source.chunks)));if(fill.style)fill.style.width=`${Math.max(2,(Number(source.chunks)||0)/most*100)}%`;bar.append(fill);row.append(r1,make('div','r2',plural(source.units,'source unit')),bar);box.append(row);}
 $('uploads-off').hidden=allowUploads;
 const select=$('library-collection'),labelsList=[...new Set(historyRuns.flatMap(run=>run.sources||[]))];
 if(changedKey(select,labelsList.join('|'))){const keep=libraryCollection;select.replaceChildren();const any=make('option','','Any collection');any.value='';select.append(any);for(const label of labelsList){const option=make('option','',label);option.value=label;select.append(option);}select.value=labelsList.includes(keep)?keep:'';libraryCollection=select.value;select.hidden=labelsList.length<2;}
}
function liveRun(){
 const running=historyRuns.find(run=>run.status==='running');if(running)return running;
 // An older server cannot say which saved folder is running: it is the newest unfinished one with the running question.
 const question=lastJob?.status==='running'&&lastJob.kind!=='upload'?requestInfo(lastJob)?.question:null;
 return question?historyRuns.find(run=>run.status==='interrupted'&&run.question===question):undefined;
}
function runKind(run){return run.status==='complete'?(run.answer?'answer':'evidence'):run.status==='running'?'running':run.status==='failed'||run.status==='stopped'?'stopped':'other';}
function runResult(run){return run.status==='complete'?(run.answer?'Answered':'Evidence only'):run.status==='running'?'Running now':run.status==='failed'?'Stopped by an error':run.status==='stopped'?'Stopped on request':run.status==='interrupted'?'Not finished':'Unavailable';}
function dayGroup(time,now=Date.now()){
 if(time===null)return 'Date unknown';
 const today=new Date(now),start=new Date(today.getFullYear(),today.getMonth(),today.getDate()).getTime(),day=86400000;
 if(time>=start)return 'Today';if(time>=start-day)return 'Yesterday';if(time>=start-6*day)return 'Earlier this week';
 const d=new Date(time);return `${MONTHS[d.getMonth()]} ${d.getFullYear()}`;
}
function whenLabel(time,now=Date.now()){
 if(time===null)return '';const group=dayGroup(time,now);if(group==='Today'||group==='Yesterday')return clock(time);
 const d=new Date(time);return group==='Earlier this week'?WEEKDAYS[d.getDay()]:`${d.getDate()} ${MONTHS[d.getMonth()].slice(0,3)}`;
}
function runMetaLine(run){
 const parts=[];
 if(Number.isSafeInteger(run.citations))parts.push(`${plural(run.citations,'citation')} to ${plural(run.cited,'evidence item')}`);
 const depth=depthName(run.settings);if(depth&&depth!=='custom')parts.push(`${depth[0].toUpperCase()+depth.slice(1)} depth`);
 if(Array.isArray(run.sources)&&run.sources.length)parts.push(run.sources.length===1?run.sources[0]:plural(run.sources.length,'collection'));
 return parts.join('. ')+(parts.length?'.':'');
}
function matchesLibrary(run,kind=runKind(run)){
 if(libraryFilter==='answer'&&kind!=='answer')return false;if(libraryFilter==='evidence'&&kind!=='evidence')return false;if(libraryFilter==='stopped'&&!['stopped','other'].includes(kind))return false;
 if(libraryCollection&&!(run.sources||[]).includes(libraryCollection))return false;
 if(libraryQuery){const fold=text=>String(text).toLowerCase().normalize('NFD').replace(/[̀-ͯ]/g,'');if(!fold(run.question).includes(fold(libraryQuery)))return false;}
 return true;
}
function renderLibrary(){
 const box=$('history');box.replaceChildren();const now=Date.now(),live=liveRun();let group=null,shown=0;
 const kindOf=run=>run===live?'running':runKind(run),resultOf=run=>run===live?'Running now':runResult(run);
 const counts={all:historyRuns.length,answer:0,evidence:0,stopped:0};for(const run of historyRuns){const kind=kindOf(run);if(kind==='answer')counts.answer++;else if(kind==='evidence')counts.evidence++;else if(kind!=='running')counts.stopped++;}
 for(const key of Object.keys(counts))textIfChanged($('filter-'+key),formatNumber(counts[key]));
 textIfChanged($('library-count'),historyRuns.length?`${formatNumber(historyRuns.length)} saved`:'');textIfChanged($('nav-library-count'),historyRuns.length?formatNumber(historyRuns.length):'');
 for(const run of historyRuns){
  if(!matchesLibrary(run,kindOf(run)))continue;shown++;
  const time=validDate(run.created),label=dayGroup(time,now);
  if(label!==group){group=label;box.append(make('div','group',label));}
  const b=document.createElement('button'),question=document.createElement('span'),meta=document.createElement('small'),result=make('span','history-result'),what=make('span','what'),dot=make('i','dot');
  const kind=kindOf(run),running=kind==='running';b.type='button';b.runId=run.id;b.dataset.kind=kind;b.title=run.question;question.className='history-question';question.textContent=run.question;
  meta.className='history-meta';meta.textContent=runMetaLine(run);meta.hidden=!meta.textContent;
  dot.dataset.kind=kind;what.append(dot,document.createTextNode(resultOf(run)));result.append(what);
  if(run.status==='complete'&&run.answer&&Number.isSafeInteger(run.cited)&&Number.isSafeInteger(run.checked)){
   const checked=make('span','checked'),bar=make('span','minibar');for(let k=0;k<Math.min(run.cited,8);k++){const square=make('i');if(k<Math.round(run.checked/Math.max(1,run.cited)*Math.min(run.cited,8)))square.dataset.on='true';bar.append(square);}
   checked.dataset.done=String(run.checked>=run.cited&&run.cited>0);checked.append(bar,document.createTextNode(run.checked>=run.cited&&run.cited>0?`All ${formatNumber(run.cited)} checked`:`${formatNumber(run.checked)} of ${formatNumber(run.cited)} checked`));result.append(checked);
  }
  const when=make('span','history-when',whenLabel(time,now));if(time!==null)when.title=longDate(time);
  // Saved answers open; the running inquiry shows its progress; anything stopped can be asked again.
  if(run.status!=='complete'&&!running)result.append(make('span','again','Ask again'));
  b.append(question,meta,result,when);
  b.addEventListener('click',()=>{
   if(run.status==='complete'){navigate(`#/inquiry/${encodeURIComponent(run.id)}`);if(!globalThis.addEventListener)openRun(run.id).catch(e=>status(e.message,'error'));}
   else if(running)navigate('#/progress');
   else askAgainWith(run.question,'Question restored. Ratings already made for it are reused.');
  });
  box.append(b);
 }
 const empty=$('library-empty');empty.hidden=shown>0||!historyLoaded;
 textIfChanged(empty,historyRuns.length?'No saved inquiries match these filters.':'No saved inquiries yet. Ask your first question to start your library.');
 markCurrentRun(current?.context?.run_id);
}
function renderRecent(){
 // Places to go back to: the inquiry running now, then saved results. Stopped ones stay in the library.
 const box=$('recent');box.replaceChildren();
 const add=(href,kind,text,id)=>{const link=make('a'),dot=make('i','dot');dot.dataset.kind=kind;link.href=href;if(id)link.runId=id;link.append(dot,make('span','',text));box.append(link);};
 const running=liveRun(),jobRunning=lastJob?.status==='running'&&lastJob.kind!=='upload';
 if(running||jobRunning)add('#/progress','running',running?.question||requestInfo(lastJob)?.question||'Inquiry in progress',running?.id);
 const rows=historyRuns.filter(run=>run.status==='complete').slice(0,6);
 for(const run of rows)add(`#/inquiry/${encodeURIComponent(run.id)}`,runKind(run),run.question,run.id);
 if(historyRuns.length>rows.length+(running?1:0)){const more=make('a','more',historyRuns.length===1?'Open the library':`All ${formatNumber(historyRuns.length)} inquiries`);more.href='#/library';box.append(more);}
 markCurrentRun(current?.context?.run_id);
}
function markCurrentRun(id){for(const list of [$('history').children||[],$('recent').children||[]])for(const item of list){if(id&&item.runId===id)item.setAttribute('aria-current','true');else item.removeAttribute?.('aria-current');}}
async function refreshHistory(){
 if(historyLoading)return;
 historyLoading=true;
 try{
  const runs=await api('/api/runs');historyRuns=Array.isArray(runs)?runs.filter(run=>run&&typeof run.id==='string'&&typeof run.question==='string'):[];
  historyLoaded=true;renderLibrary();renderRecent();collectionsKey=null;renderCollections();$('history-status').textContent='';
 }
 catch(e){historyLoaded=false;$('history-status').textContent=`Saved inquiries could not refresh: ${e.message}. Progress tracking continues.`;}
 finally{historyLoading=false;}
}
for(const chip of $('library-filters').querySelectorAll?.('[data-filter]')||[])chip.addEventListener('click',()=>{libraryFilter=chip.dataset.filter;for(const other of $('library-filters').querySelectorAll('[data-filter]'))other.setAttribute('aria-pressed',String(other===chip));renderLibrary();});
$('library-search').addEventListener('input',()=>{libraryQuery=String($('library-search').value||'').trim();renderLibrary();});
$('library-collection').addEventListener('change',()=>{libraryCollection=$('library-collection').value;renderLibrary();});

/* ---------- state ---------- */
async function refresh(){
 const s=await api('/api/state');serverBusy=s.job?.status==='running';connectionLost=false;lastStateAt=Date.now();
 features=Array.isArray(s.features)?s.features.filter(name=>typeof name==='string'):[];
 if(Number.isSafeInteger(s.limits?.max_rating_calls))maxRatingCalls=s.limits.max_rating_calls;
 renderProgress(s.job);
 syncModels(s.providers);
 allowUploads=s.allow_uploads===true;$('add-sources').hidden=!allowUploads;syncSources(s.sources);
 if(!Object.keys(defaults).length){defaults=s.settings;for(const [key,label]of Object.entries(labels)){const l=document.createElement('label');l.id=key+'-field';l.textContent=label;const input=document.createElement('input');input.type='number';input.id=key;input.min=key.includes('relevance')||['full_units','summaries','summary_candidates','rating_neighbors','answer_neighbors'].includes(key)?'0':'1';if(key.includes('relevance')){input.max='100';input.step='any';}input.value=defaults[key];input.addEventListener('input',()=>{if(!applyingDepth)syncDepth();updateControls();});l.append(input);$('fields').append(l);}$('rerank_enabled').checked=defaults.rerank_enabled;$('exclude_full_unit_passages').checked=defaults.exclude_full_unit_passages;$('evidence_mode').value=defaults.evidence_mode||'fixed';$('context_window_tokens').value=defaults.context_window_tokens??100000;$('llm_max_output_tokens').value=defaults.llm_max_output_tokens??4096;syncDepth();}
 if(Number.isSafeInteger(s.context_budget?.overhead_reserve_tokens))budgetOverhead=s.context_budget.overhead_reserve_tokens;
 renderRoom(s);updateControls();
 if(!routed){routed=true;if(explicitRoute)route();else if(['running','failed','stopped'].includes(s.job?.status))show('progress');else if(s.job?.status!=='complete')show('ask');}
 // Slow or unavailable history must never delay the running job's progress. Saved inquiries change only
 // when a job starts or ends, so polls during a run do not re-read every saved request.
 const job=s.job||{},nextHistoryKey=`${job.status||'idle'}|${job.progress?.started_at||''}`;
 if(!historyLoaded||nextHistoryKey!==historyKey){historyKey=nextHistoryKey;refreshHistory();}
 else renderRecent();
 return s;
}
async function watch(){
 if(polling)return;polling=true;let retry=false;
 try{
  const s=await refresh(),job=s.job||{status:'idle'};
  if(job.status==='running'){working(true);retry=true;status(progressSuppressed&&screen!=='progress'?'An inquiry is running in the background. Open it from Recent.':'');}
  else{
   working(false);
   if(job.status==='failed')status(job.error||'The inquiry stopped after an error.','error');
   else if(job.status==='stopped')status(job.error||'The inquiry was stopped.');
   else if(job.status==='complete'&&job.kind==='upload'){
    const added=job.result?.added_units,kept=job.result?.unchanged_units;
    status(Number.isSafeInteger(added)?`Import complete: ${plural(added,'new unit')}${Number.isSafeInteger(kept)&&kept?`, ${formatNumber(kept)} already in the corpus`:''}.`:'Import complete.');
    if(uploadSubmitted){uploadSubmitted=false;$('files').value='';droppedFile=null;if(screen==='progress')navigate('#/sources');}
   }
   else if(job.status==='complete'){
    status(progressSuppressed&&screen!=='progress'?'The background inquiry is complete. Open it from Recent.':'');
    if(!progressSuppressed&&job.result?.run_id&&current?.context?.run_id!==job.result.run_id)await openRun(job.result.run_id,current!==null||requestedSources===null,true,{navigate:followSubmission||screen==='progress'||!explicitRoute});
   }
  }
 }catch(e){
  connectionLost=true;retry=true;updateProgressTime();updateControls();
  status(`${e.message}. Retrying automatically${serverBusy||lastJob?.status==='running'?'; your inquiry may still be running':''}.`,'error');
 }finally{
  if(retry)setTimeout(()=>{polling=false;watch();},connectionLost?3000:1500);
  else polling=false;
 }
}
async function openRun(id,restoreSelection=true,followJob=false,options={}){
 if(!followJob){progressSuppressed=true;status(serverBusy?'An inquiry is running in the background. Open it from Recent.':'');}
 const version=++readingVersion;loadingRun=true;updateControls();
 try{
 const value=await api('/api/runs/'+encodeURIComponent(id));if(version!==readingVersion)return;
 if(restoreSelection&&Array.isArray(value.context.sources)){selectAllSources=false;selectedSources=new Set(value.context.sources.map(source=>source.id));renderSources();}
 current=value;current.verdicts=value.verdicts&&typeof value.verdicts==='object'?value.verdicts:{};
 readerCite=null;$('reader').hidden=true;delete $('answer-panel').dataset.reading;activeCite=null;
 // Opening a saved inquiry never replaces a question the user is still writing.
 const draft=$('question').value,keptDraft=Boolean(draft.trim())&&draft!==filledQuestion&&draft!==value.context.question;
 if(!keptDraft){$('question').value=value.context.question;filledQuestion=value.context.question;updateQuestionCount();}
 const title=$('result-title');title.textContent=value.context.question;title.setAttribute('data-length',value.context.question.length>140?'long':'');
 if(options.navigate!==false){show('inquiry');navigate(`#/inquiry/${encodeURIComponent(value.context.run_id)}${options.cite?'/'+options.cite:''}`);}
 renderMeta(value.meta,Boolean(value.answer),value.context);$('copy-answer').hidden=!value.answer;
 renderAnswer(value.answer||'This inquiry gathered evidence only. Read it under Evidence, or copy the prompt from Model handoff into your own model.');
 if(!value.answer)answerBlocks=[];
 $('prompt').textContent=value.prompt||'No handoff prompt was saved for this inquiry.';$('copy-prompt').disabled=$('download-prompt').disabled=!value.prompt;$('counts').replaceChildren();
 for(const [key,one,many]of [['full_units','complete unit','complete units'],['summaries','summary','summaries'],['passages','passage','passages']]){const span=document.createElement('span');span.textContent=plural(value.context.counts[key],one,many);$('counts').append(span);}
 renderContextUsage(value.context.context_budget);
 renderSearchPhrasings(value.search_phrasings);
 $('notices').textContent=value.context.notices.join('. ');$('evidence-view').replaceChildren();
 for(const [group,key]of [['full_units','text'],['summaries','summary'],['passages','target']])for(const row of value.context[group]){
  const details=document.createElement('details'),summary=document.createElement('summary'),meta=document.createElement('p'),pre=document.createElement('pre'),open=document.createElement('button');
  details.dataset.citation=row.citation;details.dataset.kind=String(row.citation)[0];summary.textContent=`${row.citation}  ${row.title||row.unit.title}, ${group==='summaries'?'summary only':group==='full_units'?'complete source':'quoted passage'}`;
  const identity=row.unit||row,level=levelFor(row.rating);meta.className='meta';meta.textContent=`${identity.locator}. Source: ${identity.source_id}. Relevance: ${row.rating.score.toFixed(2)}/100${level!==null&&row.rating.label?` (${shortLabel(row.rating.label)})`:''}`;
  pre.textContent=key==='target'?(row.context?.text??row.target.text+row.neighbors.map(n=>'\n\nAdjacent context:\n'+n.text).join('')):row[key];
  if(key==='target'){const words=text=>text.trim().split(/\s+/u).filter(Boolean).length;meta.textContent+=`. ${words(row.target.text)}-word passage`;if(row.context&&(row.context.start!==row.target.start||row.context.end!==row.target.end))meta.textContent+=`, ${words(row.context.text)} words with context`;}
  open.type='button';open.className='link open-reader';open.textContent='Open in the reader';open.addEventListener('click',()=>openReader(row.citation));
  details.append(summary,meta,pre,open);$('evidence-view').append(details);
 }
 showTab(value.answer?'answer':'evidence');
 renderInsights();
 markCurrentRun(value.context.run_id);
 if(followJob&&followSubmission){followSubmission=false;revealResult();}
 if(keptDraft&&!followJob)status(DRAFT_KEPT);
 if(options.cite&&citeIndex.has(options.cite))openReader(options.cite,{updateHash:false});
 }finally{if(version===readingVersion){loadingRun=false;updateControls();}}
}
function renderMeta(meta,answered,context){
 const parts=[];
 if(meta&&typeof meta==='object'){
  const time=validDate(meta.created);
  if(answered)parts.push((typeof meta.llm_model==='string'&&meta.llm_model?`Answered by ${meta.llm_model}`:'Answered by the configured model')+(time!==null?` on ${longDate(time)}`:''));
  else if(meta.answer===false)parts.push('Evidence only'+(time!==null?`, gathered on ${longDate(time)}`:''));
  else if(time!==null)parts.push(longDate(time));
  const chosen=Array.isArray(context?.sources)?context.sources:null;
  if(parts.length&&chosen&&chosen.length)parts.push(chosen.length===1?`Searched ${chosen[0].label||chosen[0].id}`:`Searched ${plural(chosen.length,'collection')}`);
 }
 $('result-meta').textContent=parts.join('. ')+(parts.length?'.':'');$('result-meta').hidden=!parts.length;
}
function renderContextUsage(budget){
 const fields=['limit_tokens','prompt_tokens','output_reserve_tokens','overhead_reserve_tokens','remaining_tokens'];
 const valid=budget&&budget.mode==='context'&&fields.every(key=>Number.isSafeInteger(budget[key])&&budget[key]>=0)&&budget.limit_tokens>0;
 $('context-usage').hidden=!valid;if(!valid)return;
 const used=budget.prompt_tokens+budget.output_reserve_tokens+budget.overhead_reserve_tokens,format=formatNumber;
 $('context-progress').max=budget.limit_tokens;$('context-progress').value=Math.min(used,budget.limit_tokens);
 $('context-usage-summary').textContent=`: ${format(used)} of ${format(budget.limit_tokens)} estimated tokens`;
 $('context-usage-detail').textContent=`Prompt ${format(budget.prompt_tokens)}, answer reserve ${format(budget.output_reserve_tokens)}, overhead reserve ${format(budget.overhead_reserve_tokens)}, remaining ${format(budget.remaining_tokens)}. ${budget.counting_note||'Conservative estimate; provider usage can differ.'}`;
}
function renderSearchPhrasings(rows){
 const panel=$('search-phrasings'),list=$('search-phrasing-list');list.replaceChildren();panel.open=false;panel.hidden=true;
 if(!Array.isArray(rows)||rows.length>5)return;
 const names={original:'Original question',improved:'Clearer restatement',specific:'Specific detail',broad:'Related concepts',hypothetical_answer:'Hypothetical answer, a search hint only'};
 for(const row of rows){
  if(!row||!Object.hasOwn(names,row.kind)||typeof row.text!=='string'||!row.text.trim()||row.text.length>4000)continue;
  const item=document.createElement('li'),label=document.createElement('strong'),text=document.createElement('p');
  label.textContent=names[row.kind];text.textContent=row.text;item.append(label,text);list.append(item);
 }
 panel.hidden=!list.children.length;
}
function revealResult(){
 // Bring the finished result into view unless the user is busy typing a new question.
 const active=document.activeElement,typing=Boolean(active&&['TEXTAREA','INPUT','SELECT'].includes(active.tagName)&&!(active.id==='question'&&active.value===filledQuestion));
 if(typing)return;
 $('result').scrollIntoView({block:'start',behavior:reduceMotion()?'auto':'smooth'});
 $('result-title').focus({preventScroll:true});
}
function showTab(name){
 for(const key of ['answer','evidence','prompt']){const tab=$(key+'-tab');tab.setAttribute('aria-selected',String(name===key));tab.setAttribute('tabindex',name===key?'0':'-1');$(key+'-view').hidden=name!==key;}
 $('answer-panel').hidden=name!=='answer';
 for(const id of ['check'])if(name!=='answer'&&$(id).dataset)$(id).dataset.dim='true';else if($(id).dataset)delete $(id).dataset.dim;
 if(name==='answer')scheduleLayout();
}
for(const name of ['answer','evidence','prompt']){$(name+'-tab').addEventListener('click',()=>showTab(name));$(name+'-tab').addEventListener('keydown',e=>{const names=['answer','evidence','prompt'];let next=names.indexOf(name);if(e.key==='ArrowRight')next=(next+1)%3;else if(e.key==='ArrowLeft')next=(next+2)%3;else if(e.key==='Home')next=0;else if(e.key==='End')next=2;else return;e.preventDefault();showTab(names[next]);$(names[next]+'-tab').focus();});}

/* ---------- actions ---------- */
async function submitQuery(payload){
 if(busy||serverBusy||loadingRun)return;
 working(true);lastSubmission=payload;submitting=true;beginProgress('query');
 try{
  try{await api('/api/query',payload);}finally{submitting=false;}
  followSubmission=true;draftStore.set('');
  current=null;$('result').hidden=screen!=='inquiry';serverBusy=true;watch();
 }catch(e){
  if(e.httpStatus===409){serverBusy=true;status('Another operation is already running. Following its progress…');watch();}
  // A rejected request never started: show why, without a failed workflow panel.
  else if(e.responseReceived){working(false);renderProgress(null);show('ask');navigate('#/new');status(e.message,'error');}
  else if(lastJob?.status==='running'){followSubmission=true;serverBusy=true;connectionLost=true;status(`${e.message}. Checking whether your inquiry started…`);watch();}
  else{working(false);status(e.message,'error');}
 }
}
function startQuery(answer){
 if(busy||serverBusy||loadingRun)return;
 const question=$('question').value;
 if(!question.trim()){if(screen!=='ask')show('ask');status('Type a question first.','error');$('question').setAttribute('aria-invalid','true');$('question').focus();return;}
 let sources;
 try{sources=querySources();}catch(e){status(e.message,'error');return;}
 submitQuery({question,settings:settings(),answer,sources});
}
for(const [id,answer]of [['retrieve',false],['answer',true]])$(id).addEventListener('click',()=>startQuery(answer));
$('workflow-retry').addEventListener('click',()=>{if(lastSubmission)submitQuery(lastSubmission);});
$('stop').addEventListener('click',async()=>{
 if(stopRequested||!lastJob||lastJob.status!=='running')return;stopRequested=true;renderProgress(lastJob);
 try{await api('/api/cancel',{});status('Stopping. Ratings already made stay cached for a retry.');}
 catch(e){stopRequested=false;renderProgress(lastJob);status(e.httpStatus===409?'The inquiry had already finished.':e.message,e.httpStatus===409?'':'error');}
});
function askAgainWith(question,note=''){
 if(typeof question!=='string'||!question)return;
 $('question').value=question;filledQuestion=question;$('question').removeAttribute('aria-invalid');updateQuestionCount();navigate('#/new');show('ask');$('question').focus?.();
 if(note)status(note);
}
$('ask-again').addEventListener('click',()=>{if(current)askAgainWith(current.context.question);});
$('question').addEventListener('keydown',e=>{
 if(e.key!=='Enter'||!(e.ctrlKey||e.metaKey))return;
 e.preventDefault();
 if(!$('answer').disabled)startQuery(true);else if(!$('retrieve').disabled)startQuery(false);
});
function growQuestion(){const box=$('question');if(!box.style||!browserLayout())return;box.style.height='auto';box.style.height=`${Math.min(box.scrollHeight+2,420)}px`;}
function updateQuestionCount(){const length=String($('question').value??'').length,counter=$('question-count');counter.hidden=length<3600;if(length>=3600)textIfChanged(counter,`${formatNumber(length)} / 4,000 characters`);counter.setAttribute('data-limit',length>=4000?'reached':'');growQuestion();}
$('question').addEventListener('input',()=>{
 $('question').removeAttribute('aria-invalid');if($('status').textContent===DRAFT_KEPT)status('');
 const value=$('question').value;draftStore.set(value===filledQuestion?'':value);updateQuestionCount();
});
$('plan').addEventListener('click',async()=>{if(busy||serverBusy||loadingRun)return;try{const sources=querySources();working(true);const p=await api('/api/plan',{settings:settings(),sources});status(planMessage(p));}catch(e){status(e.message,'error');}finally{working(false);}});
function fileData(file){return new Promise((resolve,reject)=>{const r=new FileReader();r.onload=()=>resolve(String(r.result).split(',')[1]);r.onerror=reject;r.readAsDataURL(file);});}
function uploadError(files){
 if(!files.length)return 'Choose a file to import.';
 if(files.length!==1)return 'Import one file at a time. A JSONL file can contain many units.';
 if(files[0].size>8*1024*1024)return 'Upload exceeds 8 MB. Split into meaningful units.';
 // Every unit needs a summary; only JSONL can supply one without a model.
 if(!/\.jsonl$/i.test(files[0].name)&&!$('summarize').checked)return llmEnabled?'Documents need a summary before indexing. Turn on Write missing summaries to send this document to your configured model, or import JSONL units that already include summaries.':'Documents need a summary before indexing, and no model is configured to write one. Import JSONL units that already include summaries.';
 return '';
}
$('upload').addEventListener('click',async()=>{
 if(busy||serverBusy||loadingRun||!sourcesReady||!allowUploads)return;
 const files=droppedFile?[droppedFile]:Array.from($('files').files||[]),problem=uploadError(files);
 if(problem){status(problem,'error');return;}
 let submitted=false;
 try{
  working(true);beginProgress('upload');lastJob={...lastJob,request:{filename:files[0].name}};renderProgress(lastJob);const data=await fileData(files[0]);submitted=true;
  await api('/api/upload',{filename:files[0].name,data,split:$('split').value,summarize:$('summarize').checked});uploadSubmitted=true;serverBusy=true;watch();
 }catch(e){
  if(e.httpStatus===409){serverBusy=true;status('Another operation is already running. Following its progress…');watch();}
  else if(submitted&&!e.responseReceived){uploadSubmitted=true;serverBusy=true;connectionLost=true;status(`${e.message}. Checking whether your import started…`);watch();}
  else{working(false);renderProgress(null);show('library');status(e.message,'error');}
 }
});
$('files').addEventListener('change',()=>{droppedFile=null;});
const zone=$('dropzone');
zone.addEventListener('dragover',e=>{e.preventDefault();zone.dataset.over='true';});
zone.addEventListener('dragleave',()=>{delete zone.dataset.over;});
zone.addEventListener('drop',e=>{e.preventDefault();delete zone.dataset.over;const files=Array.from(e.dataTransfer?.files||[]);if(files.length!==1){status('Drop one file at a time. A JSONL file can contain many units.','error');return;}droppedFile=files[0];status(`Ready to import ${files[0].name}. Choose the options, then Import and index.`);});
function saveFile(text,type,name){const url=URL.createObjectURL(new Blob([text],{type}));const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
async function copyText(button,text,label){
 const show=message=>{button.textContent=message;setTimeout(()=>{button.textContent=label;},2000);};
 try{
  if(!globalThis.navigator?.clipboard?.writeText)throw Error('Clipboard unavailable');
  await globalThis.navigator.clipboard.writeText(text);show('Copied');
 }catch{show('Copy failed');status('Copying was blocked by the browser. Select the text and copy it manually.','error');}
}
$('download').addEventListener('click',()=>{if(current)saveFile(JSON.stringify(current.context,null,2),'application/json','mori-'+current.context.run_id+'.json');});
$('download-prompt').addEventListener('click',()=>{if(current?.prompt)saveFile(current.prompt,'text/plain;charset=utf-8','mori-'+current.context.run_id+'-prompt.txt');});
$('copy-prompt').addEventListener('click',()=>{if(current?.prompt)copyText($('copy-prompt'),current.prompt,'Copy prompt');});
$('copy-answer').addEventListener('click',()=>{if(current?.answer)copyText($('copy-answer').lastChild||$('copy-answer'),current.answer,'Copy answer');});
$('menu').addEventListener('click',openMenu);$('menu-close').addEventListener('click',closeMenu);$('scrim').addEventListener('click',closeMenu);
document.addEventListener?.('keydown',e=>{
 const typing=['INPUT','TEXTAREA','SELECT'].includes(document.activeElement?.tagName)||document.activeElement?.isContentEditable;
 if(e.key==='Escape'){if(readerCite){closeReader();e.preventDefault();}else closeMenu();return;}
 if(typing||e.ctrlKey||e.metaKey||e.altKey)return;
 if(e.key==='n'||e.key==='N'){e.preventDefault();navigate('#/new');}
 else if(e.key==='/'){e.preventDefault();navigate('#/library');show('library');$('library-search').focus();}
});
globalThis.addEventListener?.('hashchange',()=>{route();});
globalThis.addEventListener?.('resize',()=>{scheduleLayout();if(lastLive&&screen==='progress'){liveKeys.grid=null;renderLive(lastLive,lastJob);}});
document.fonts?.ready?.then(()=>scheduleLayout());
const savedDraft=draftStore.get();
if(savedDraft&&!$('question').value)$('question').value=savedDraft;
updateQuestionCount();
working(true);
watch();
