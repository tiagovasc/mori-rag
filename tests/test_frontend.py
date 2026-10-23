"""Run the real browser script against a small DOM/API double, with no network."""
from html.parser import HTMLParser
from pathlib import Path
import shutil
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which('node')
HARNESS = r"""
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const script = fs.readFileSync('mori/static/app.js', 'utf8');
const settle = () => new Promise(resolve => setImmediate(resolve));
function deferred() {let resolve, reject; const promise = new Promise((yes, no) => {resolve=yes; reject=no;}); return {promise,resolve,reject};}
const alpha = {id:'alpha',label:'First work',units:1,chunks:3};
const beta = {id:'beta',label:'Second work',units:3,chunks:7};
function packet(question, sources) {
  const context={question,run_id:question,counts:{full_units:0,summaries:0,passages:0},notices:[],full_units:[],summaries:[],passages:[]};
  if(sources !== undefined)context.sources=sources;
  return {context,answer:'Synthetic answer.',prompt:'Synthetic prompt.'};
}
function browser(options={}) {
  const elements=new Map();
  class Element {
    constructor(tag='div') {this.tagName=tag.toUpperCase();this.children=[];this.events={};this.attributes={};this.dataset={};this.disabled=false;this.checked=false;this.hidden=false;this.value='';this._text='';}
    set id(value) {this._id=value;elements.set(value,this);} get id(){return this._id;}
    set textContent(value){this._text=String(value);this.children=[];}
    get textContent(){return this._text+this.children.map(child=>child.textContent).join('');}
    set innerHTML(value){throw Error('Untrusted HTML must not be rendered');}
    append(...children){this.children.push(...children);}
    replaceChildren(...children){this._text='';this.children=[...children];}
    setAttribute(key,value){this.attributes[key]=String(value);}
    removeAttribute(key){delete this.attributes[key];if(key==='value')this.value=undefined;}
    addEventListener(name,listener){(this.events[name]??=[]).push(listener);}
    async emit(name){for(const listener of this.events[name]||[])await listener({target:this,preventDefault(){}});}
    focus(){} scrollIntoView(){}
  }
  const document={
    getElementById(id){if(!elements.has(id)){const e=new Element();e.id=id;}return elements.get(id);},
    createElement(tag){return new Element(tag);},
    createTextNode(text){const e=new Element('#text');e.textContent=text;return e;},
    querySelector(selector){if(selector==='meta[name=mori-token]')return {content:'synthetic-token'};return null;},
  };
  const settings={retrieval_k:500,jev_k:100,final_passages:20,full_units:1,summaries:5,min_relevance:0,summary_min_relevance:0,summary_candidates:0,rerank_k:20,rating_neighbors:0,answer_neighbors:0,rerank_enabled:false,exclude_full_unit_passages:true};
  const server={
    state:{stats:{units:4,chunks:10},settings,providers:{embedding:'demo',llm:'none'},job:{status:'idle'},sources:[alpha,beta],allow_uploads:true,...options.state},
    stateGate:options.stateGate,runReplies:{},requests:[],timers:[],intervals:[],
    plan:{candidate_passages:10,summary_ratings:4,rating_attempt_ceiling:14,full_units:1,summaries:2,final_passage_cap:10},query:{},
  };
  const fetch=async(path,options)=>{
    const body=options?.body?JSON.parse(options.body):undefined;server.requests.push({path,body});
    let value;
    if(path==='/api/state'){
      if(server.hangState)await new Promise((resolve,reject)=>options.signal.addEventListener('abort',()=>reject(Object.assign(Error('Aborted'),{name:'AbortError'}))));
      if(server.stateGate)await server.stateGate.promise;if(server.stateError)throw server.stateError;value=server.state;
    }
    else if(path==='/api/runs'){if(server.runsGate)await server.runsGate.promise;if(server.runsError)throw server.runsError;value=[];}
    else if(path.startsWith('/api/runs/')){value=server.runReplies[decodeURIComponent(path.slice('/api/runs/'.length))];assert.ok(value,'Unknown synthetic run');}
    else if(path==='/api/plan')value=server.plan;
    else if(path==='/api/query')value=server.query;
    else if(path==='/api/upload')value=server.upload;
    else throw Error('Unexpected request: '+path);
    if(typeof value==='function')value=await value();
    return {ok:!value?.error,status:value?.httpStatus||200,json:async()=>value};
  };
  class FileReader {
    readAsDataURL(){Promise.resolve(server.fileGate?.promise).then(()=>{this.result='data:text/plain;base64,e30=';this.onload();});}
  }
  const context=vm.createContext({document,fetch,FileReader,location:{search:options.search||''},URLSearchParams,setTimeout:fn=>server.timers.push(fn),setInterval:fn=>server.intervals.push(fn)});
  if(options.abort){
    context.AbortController=AbortController;
    context.setTimeout=fn=>{server.timers.push(fn);return fn;};
    context.clearTimeout=fn=>{const index=server.timers.indexOf(fn);if(index>=0)server.timers.splice(index,1);};
  }
  vm.runInContext(script,context);
  const el=id=>document.getElementById(id);
  const inputs=()=>el('source-options').children.map(label=>label.children[0]);
  return {
    server,el,context,inputs,eval:code=>vm.runInContext(code,context),
    selected:()=>inputs().filter(input=>input.checked).map(input=>input.value),
    async click(id){if(!el(id).disabled)await el(id).emit('click');},
    async choose(id,checked){const input=inputs().find(input=>input.value===id);assert.ok(input);input.checked=checked;await input.emit('change');},
    refresh:()=>vm.runInContext('refresh()',context),
    openRun:id=>vm.runInContext('openRun('+JSON.stringify(id)+')',context),
  };
}
function queries(app){return app.server.requests.filter(row=>row.path==='/api/query'||row.path==='/api/plan');}
function assertQueryDisabled(app,expected){for(const id of ['retrieve','answer','plan'])assert.equal(app.el(id).disabled,expected,id);}
"""


@unittest.skipUnless(NODE, 'Node.js is required for frontend behavior checks')
class SourceSelectionTests(unittest.TestCase):
    def run_ui(self, scenario):
        code = HARNESS + '\n(async()=>{\n' + scenario + '\n})().catch(error=>{console.error(error);process.exitCode=1;});'
        result = subprocess.run([NODE, '-e', code], cwd=ROOT, capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_initial_loading_and_default_all_sources(self):
        self.run_ui(r"""
const gate=deferred(),app=browser({stateGate:gate});
assertQueryDisabled(app,true);assert.equal(app.el('upload').disabled,true);
assert.equal(app.el('source-picker').disabled,true);
gate.resolve();await settle();
assert.deepEqual(app.selected(),['alpha','beta']);assertQueryDisabled(app,false);
assert.equal(app.el('source-summary').textContent,'2 of 2 sources selected · 4 source units · 10 passages');
assert.equal(app.el('source-count').textContent,'All 2 selected');
assert.equal(app.el('sources-all').disabled,true);assert.equal(app.el('add-sources').hidden,false);
assert.equal(app.server.requests.filter(r=>r.path==='/api/state').length,1);
""")

    def test_no_selection_blocks_requests_and_bodies_use_checked_sources(self):
        self.run_ui(r"""
const app=browser();await settle();
await app.click('sources-none');assertQueryDisabled(app,true);
assert.match(app.el('source-summary').textContent,/0 of 2 sources selected/);
assert.equal(app.el('source-count').textContent,'None selected');
await app.click('answer');await app.click('plan');assert.equal(queries(app).length,0);
assert.throws(()=>app.eval('querySources()'),/Choose at least one source/);
await app.choose('beta',true);assertQueryDisabled(app,false);
assert.equal(app.el('source-count').textContent,'1 of 2 selected');
await app.click('plan');assert.deepEqual(queries(app)[0].body.sources,['beta']);
app.el('question').value='Synthetic question';await app.click('answer');await settle();
assert.deepEqual(queries(app)[1].body.sources,['beta']);assert.equal(queries(app)[1].body.answer,true);
assert.equal(queries(app)[1].body.question,'Synthetic question');assert.deepEqual(app.selected(),['beta']);
await app.click('retrieve');await settle();assert.equal(queries(app)[2].body.answer,false);
assert.deepEqual(queries(app)[2].body.sources,['beta']);
""")

    def test_url_selection_and_unknown_ids_fail_closed(self):
        self.run_ui(r"""
const app=browser({search:'?sources=beta&sources=beta'});await settle();
assert.deepEqual(app.selected(),['beta']);await app.click('plan');assert.deepEqual(queries(app)[0].body.sources,['beta']);
const missing=browser({search:'?sources=alpha&sources=unknown'});await settle();
assert.deepEqual(missing.selected(),['alpha']);assertQueryDisabled(missing,true);
assert.match(missing.el('source-error').textContent,/Unavailable sources: unknown/);
await missing.click('answer');assert.equal(queries(missing).length,0);
await missing.choose('beta',true);assertQueryDisabled(missing,false);assert.equal(missing.el('source-error').hidden,true);
await missing.click('plan');assert.deepEqual(queries(missing)[0].body.sources,['alpha','beta']);
const empty=browser({search:'?sources='});await settle();assert.deepEqual(empty.selected(),[]);assertQueryDisabled(empty,true);
assert.match(empty.el('source-error').textContent,/empty source ID/);
await empty.click('sources-all');assert.deepEqual(empty.selected(),['alpha','beta']);assertQueryDisabled(empty,false);
""")

    def test_polls_preserve_choices_and_handle_catalog_changes(self):
        self.run_ui(r"""
const app=browser();await settle();const gamma={id:'gamma',label:'Third',units:2,chunks:9};
app.server.state.sources.push(gamma);await app.refresh();assert.deepEqual(app.selected(),['alpha','beta','gamma']);
await app.choose('alpha',false);const input=app.inputs()[1];await app.choose('beta',false);assert.equal(app.inputs()[1],input);
await app.choose('beta',true);
app.server.state.sources.push({id:'delta',label:'Fourth',units:1,chunks:1});await app.refresh();
assert.deepEqual(app.selected(),['beta','gamma']);
app.server.state.sources=app.server.state.sources.filter(row=>row.id!=='beta');await app.refresh();
assert.deepEqual(app.selected(),['gamma']);assertQueryDisabled(app,true);assert.match(app.el('source-error').textContent,/beta/);
await app.click('sources-none');await app.refresh();assert.deepEqual(app.selected(),[]);assert.equal(app.el('source-error').hidden,true);
await app.click('sources-all');assert.deepEqual(app.selected(),['alpha','gamma','delta']);assertQueryDisabled(app,false);
""")

    def test_url_selection_survives_previous_job_automatically_opening(self):
        self.run_ui(r"""
const app=browser({search:'?sources=beta',state:{job:{status:'complete',result:{run_id:'previous'}}}});
app.server.runReplies.previous=packet('Previous',[{id:'alpha',label:'First work'}]);await settle();
assert.equal(app.el('question').value,'Previous');assert.deepEqual(app.selected(),['beta']);
await app.openRun('previous');assert.deepEqual(app.selected(),['alpha']);
const missing=browser({search:'?sources=unknown',state:{job:{status:'complete',result:{run_id:'previous'}}}});
missing.server.runReplies.previous=packet('Previous',[{id:'alpha',label:'First work'}]);await settle();
assert.deepEqual(missing.selected(),[]);assertQueryDisabled(missing,true);assert.match(missing.el('source-error').textContent,/unknown/);
""")

    def test_saved_run_restores_selection_and_latest_read_wins(self):
        self.run_ui(r"""
const app=browser();await settle();
app.server.runReplies.saved=packet('Saved',[{id:'beta',label:'Second work'}]);await app.openRun('saved');
assert.deepEqual(app.selected(),['beta']);assert.equal(app.el('question').value,'Saved');
app.server.runReplies.removed=packet('Removed',[{id:'gone',label:'Gone'}]);await app.openRun('removed');
assert.deepEqual(app.selected(),[]);assertQueryDisabled(app,true);assert.match(app.el('source-error').textContent,/gone/);
await app.click('sources-all');app.server.runReplies.legacy=packet('Legacy');await app.openRun('legacy');
assert.deepEqual(app.selected(),['alpha','beta']);
const old=deferred(),latest=deferred();app.server.runReplies.old=()=>old.promise;app.server.runReplies.latest=()=>latest.promise;
const readOld=app.openRun('old'),readLatest=app.openRun('latest');assertQueryDisabled(app,true);
latest.resolve(packet('Latest',[{id:'beta',label:'Second work'}]));await readLatest;assertQueryDisabled(app,false);
old.resolve(packet('Old',[{id:'alpha',label:'First work'}]));await readOld;
assert.deepEqual(app.selected(),['beta']);assert.equal(app.el('question').value,'Latest');
""")

    def test_upload_visibility_empty_catalog_and_failed_state(self):
        self.run_ui(r"""
const app=browser({state:{allow_uploads:false}});await settle();
assert.equal(app.el('add-sources').hidden,true);assert.equal(app.el('upload').disabled,true);await app.click('upload');
const empty=browser({state:{sources:[]}});await settle();assertQueryDisabled(empty,true);assert.equal(empty.el('upload').disabled,false);
assert.match(empty.el('source-summary').textContent,/No sources available/);
empty.server.state.sources=[alpha];await empty.refresh();assert.deepEqual(empty.selected(),['alpha']);assertQueryDisabled(empty,false);
const failed=browser({state:{sources:undefined}});await settle();assertQueryDisabled(failed,true);
assert.match(failed.el('status').textContent,/source list is unavailable/);
const explicit=browser({search:'?sources=alpha'});await settle();await explicit.choose('beta',true);
explicit.server.state.sources=undefined;await assert.rejects(explicit.refresh(),/source list is unavailable/);
explicit.server.state.sources=[alpha,beta];await explicit.refresh();assert.deepEqual(explicit.selected(),['alpha','beta']);
""")

    def test_labels_and_ids_are_rendered_as_text(self):
        self.run_ui(r"""
const source={id:'odd</label>&?"',label:'<img src=x onerror=alert(1)>',units:1,chunks:1};
const app=browser({state:{sources:[source]},search:'?sources='+encodeURIComponent(source.id)});await settle();
const label=app.el('source-options').children[0];assert.equal(label.tagName,'LABEL');
assert.equal(label.children[1].children[0].textContent,source.label);
assert.deepEqual(label.children.map(e=>e.tagName),['INPUT','SPAN']);
await app.click('plan');assert.deepEqual(queries(app)[0].body.sources,[source.id]);
const missing=browser({search:'?sources='+encodeURIComponent('<svg onload=alert(1)>')});await settle();
assert.match(missing.el('source-error').textContent,/<svg onload=alert\(1\)>/);assertQueryDisabled(missing,true);
""")

    def test_busy_budget_and_server_jobs_lock_controls(self):
        self.run_ui(r"""
const app=browser();await settle();const gate=deferred();app.server.plan=()=>gate.promise;
const budget=app.click('plan');assertQueryDisabled(app,true);assert.equal(app.el('source-picker').disabled,true);
await app.click('answer');assert.equal(queries(app).length,1);
gate.resolve({error:'Synthetic budget error'});await budget;assertQueryDisabled(app,false);
assert.match(app.el('status').textContent,/Synthetic budget error/);
app.server.query={error:'Synthetic query error'};await app.click('answer');assertQueryDisabled(app,false);
assert.deepEqual(app.selected(),['alpha','beta']);
const running=browser({state:{job:{status:'running'}}});await settle();assertQueryDisabled(running,true);
assert.equal(running.el('source-picker').disabled,true);assert.equal(running.server.timers.length,1);
const historyFailure=browser({state:{job:{status:'running'}}});historyFailure.server.runsError=Error('History unavailable');
await settle();assertQueryDisabled(historyFailure,true);assert.match(historyFailure.el('history-status').textContent,/History unavailable/);
assert.equal(historyFailure.el('source-picker').disabled,true);
assert.equal(historyFailure.server.timers.length,1);
""")

    def test_progress_renders_real_counts_and_incremental_steps(self):
        self.run_ui(r"""
const started=new Date(Date.now()-65000).toISOString();
const progress={started_at:started,updated_at:new Date().toISOString(),finished_at:null,percent:20,current_stage:'rate_passages',steps:[
 {id:'prepare',label:'Prepare sources',status:'complete',completed:null,total:null,cached:0,message:''},
 {id:'rate_passages',label:'Score passages',status:'running',completed:42,total:500,cached:12,message:'Ranking relevance'},
 {id:'answer',label:'Write answer',status:'pending',completed:null,total:null,cached:0,message:''},
]};
const app=browser({state:{job:{status:'running',kind:'query',progress}}});await settle();
assert.equal(app.el('workflow').hidden,false);assert.equal(app.el('workflow-progress').value,20);
assert.equal(app.el('workflow-current').textContent,'Score passages');assert.match(app.el('workflow-summary').textContent,/1 of 3 steps finished/);
assert.match(app.el('workflow-steps').children[1].textContent,/42 \/ 500 passages scored · 12 reused from cache/);
assert.match(app.el('workflow-elapsed').textContent,/1m \d+s elapsed/);
assert.equal(app.el('workflow-steps').children[0].dataset.state,'complete');
progress.steps[1].completed=73;progress.percent=23;await app.refresh();
assert.equal(app.el('workflow-progress').value,23);assert.match(app.el('workflow-announcement').textContent,/73 \/ 500/);
assert.equal(app.el('workflow-steps').children[2].dataset.state,'pending');assertQueryDisabled(app,true);
""")

    def test_progress_completion_failure_and_new_query_reset(self):
        self.run_ui(r"""
const stamp=new Date().toISOString(),progress={started_at:stamp,updated_at:stamp,finished_at:stamp,percent:100,current_stage:null,steps:[
 {id:'rerank',label:'Rerank',status:'skipped',message:'Not requested'},
 {id:'save',label:'Save inquiry',status:'complete',message:''},
]};
const app=browser({state:{job:{status:'complete',result:{run_id:'done'},progress}}});
app.server.runReplies.done=packet('done',[alpha]);await settle();
assert.equal(app.el('workflow-progress').value,100);assert.equal(app.el('workflow-title').textContent,'Inquiry complete');
assert.equal(app.el('workflow-steps').children[0].dataset.state,'skipped');assert.equal(app.el('result').hidden,false);assertQueryDisabled(app,false);
app.server.query=()=>{app.server.state.job={status:'running',kind:'query'};return {};};await app.click('answer');await settle();
assert.equal(app.el('workflow-progress').value,undefined);assert.equal(app.el('workflow-steps').children.length,0);
assert.equal(app.el('workflow-title').textContent,'Building your inquiry');assert.equal(app.el('result').hidden,true);assertQueryDisabled(app,true);
app.server.state.job={status:'failed',error:'Provider failed',progress:{...progress,percent:30,steps:[{id:'answer',label:'Write answer',status:'failed',message:'Provider failed'}]}};
app.server.timers.shift()();await settle();
assert.equal(app.el('workflow-title').textContent,'Inquiry stopped');assert.equal(app.el('workflow-progress').value,30);
assert.equal(app.el('workflow-steps').children[0].dataset.state,'failed');assert.match(app.el('status').textContent,/Provider failed/);assertQueryDisabled(app,false);
""")

    def test_progress_legacy_malformed_and_injection_are_safe(self):
        self.run_ui(r"""
const app=browser({state:{job:{status:'running'}}});await settle();
assert.equal(app.el('workflow-progress').value,undefined);assert.match(app.el('workflow-note').textContent,/not reported individual steps/);
app.server.state.job.progress={percent:'75',started_at:'invalid',updated_at:'invalid',steps:[null,3,{},
 {id:'odd',label:'<script>bad()</script>',status:'made-up',completed:-4,total:'30',cached:-1,message:'<img onerror=bad()>'}]};
await app.refresh();assert.equal(app.el('workflow-progress').value,undefined);
const row=app.el('workflow-steps').children[0];assert.equal(row.dataset.state,'pending');
assert.match(row.textContent,/<script>bad\(\)<\/script>/);assert.match(row.textContent,/<img onerror=bad\(\)>/);
assert.doesNotMatch(row.textContent,/-4|30|reused/);
app.server.state.job.progress.percent=Infinity;await app.refresh();assert.equal(app.el('workflow-progress').value,undefined);
""")

    def test_progress_retries_connection_loss_and_history_does_not_block(self):
        self.run_ui(r"""
const app=browser({state:{job:{status:'running',progress:{percent:25,steps:[]}}}});await settle();
app.server.stateError=Error('Connection dropped');app.server.timers.shift()();await settle();
assert.equal(app.el('workflow-progress').value,25);assertQueryDisabled(app,true);
assert.match(app.el('workflow-connection').textContent,/Connection interrupted/);assert.match(app.el('status').textContent,/Retrying automatically/);
assert.equal(app.server.timers.length,1);
app.server.stateError=null;app.server.runsGate=deferred();app.server.state.job.progress.percent=35;
app.server.timers.shift()();await settle();assert.equal(app.el('workflow-progress').value,35);
assert.match(app.el('workflow-connection').textContent,/Connected/);assert.equal(app.server.timers.length,1);
app.server.runsGate.resolve();await settle();
app.server.runsError=Error('History unavailable');app.server.state.job.progress.percent=45;
app.server.timers.shift()();await settle();assert.equal(app.el('workflow-progress').value,45);
assert.match(app.el('history-status').textContent,/History unavailable/);assert.equal(app.server.timers.length,1);
""")

    def test_saved_inquiry_hides_unrelated_progress(self):
        self.run_ui(r"""
const started=new Date().toISOString(),app=browser({state:{job:{status:'running',progress:{started_at:started,percent:20,steps:[]}}}});await settle();
app.server.runReplies.saved=packet('Saved inquiry',[beta]);await app.openRun('saved');assert.equal(app.el('workflow').hidden,true);
await app.refresh();assert.equal(app.el('workflow').hidden,true);assert.equal(app.el('question').value,'Saved inquiry');
app.server.state.job={status:'complete',result:{run_id:'background'},progress:{started_at:started,percent:100,steps:[]}};
app.server.timers.shift()();await settle();assert.equal(app.el('workflow').hidden,true);
assert.equal(app.el('question').value,'Saved inquiry');assert.match(app.el('status').textContent,/background inquiry is complete/);
""")

    def test_progress_waiting_and_failed_workers_still_lock_controls(self):
        self.run_ui(r"""
const stamp=new Date(Date.now()-30000).toISOString(),progress={started_at:stamp,updated_at:stamp,percent:60,steps:[{id:'answer',label:'Write answer',status:'running'}]};
const app=browser({state:{job:{status:'running',progress}}});await settle();
assert.match(app.el('workflow-wait').textContent,/No new progress reported/);assertQueryDisabled(app,true);
progress.steps[0].status='failed';progress.finished_at=new Date().toISOString();await app.refresh();
assert.equal(app.el('workflow-title').textContent,'Inquiry stopped');assert.equal(app.el('workflow').dataset.state,'failed');
assert.match(app.el('workflow-wait').textContent,/in-flight requests/);assertQueryDisabled(app,true);
""")

    def test_conflicting_submission_follows_actual_running_job(self):
        self.run_ui(r"""
const app=browser();await settle();app.server.query=()=>{
 app.server.state.job={status:'running',progress:{percent:40,steps:[{id:'answer',label:'Write answer',status:'running'}]}};
 return {error:'One operation is already running',httpStatus:409};
};
await app.click('answer');await settle();assertQueryDisabled(app,true);
assert.equal(app.el('workflow-title').textContent,'Building your inquiry');assert.equal(app.el('workflow-progress').value,40);
assert.equal(app.server.timers.length,1);
""")

    def test_upload_steps_render_without_query_stage_assumptions(self):
        self.run_ui(r"""
const app=browser({state:{job:{status:'running',kind:'upload',progress:{percent:25,current_stage:'read',steps:[
 {id:'prepare',label:'Prepare import',status:'complete'},
 {id:'read',label:'Read source file',status:'running',completed:1,total:4},
 {id:'import',label:'Import and embed',status:'pending'},
 {id:'save',label:'Save index',status:'pending'},
]}}}});await settle();
assert.equal(app.el('workflow-title').textContent,'Importing sources');assert.equal(app.el('workflow-current').textContent,'Read source file');
assert.equal(app.el('workflow-steps').children.length,4);assert.match(app.el('workflow-steps').children[1].textContent,/1 \/ 4 completed/);
""")

    def test_upload_locks_during_file_read_and_conflict_follows_job(self):
        self.run_ui(r"""
const app=browser();await settle();app.el('files').files=[{name:'example.jsonl',size:20}];
app.server.fileGate=deferred();app.server.upload=()=>{
 app.server.state.job={status:'running',kind:'upload',progress:{percent:40,steps:[{id:'import',label:'Import sources',status:'running'}]}};
 return {error:'One operation is already running',httpStatus:409};
};
const pending=app.click('upload');assertQueryDisabled(app,true);assert.equal(app.el('upload').disabled,true);
await app.click('answer');await app.click('upload');assert.equal(queries(app).length,0);
app.server.fileGate.resolve();await pending;await settle();assertQueryDisabled(app,true);
assert.equal(app.server.requests.filter(row=>row.path==='/api/upload').length,1);
assert.equal(app.el('workflow-title').textContent,'Importing sources');assert.equal(app.el('workflow-progress').value,40);
assert.equal(app.server.timers.length,1);
""")

    def test_unknown_submission_outcome_checks_state_before_unlocking(self):
        self.run_ui(r"""
const app=browser();await settle();app.server.query=()=>{
 app.server.state.job={status:'running',progress:{percent:10,steps:[{id:'embed',label:'Embed question',status:'running'}]}};
 throw Error('Response connection lost');
};
await app.click('answer');await settle();assertQueryDisabled(app,true);
assert.equal(app.el('workflow-progress').value,10);assert.equal(app.server.timers.length,1);
assert.equal(queries(app).length,1);
""")

    def test_hanging_state_request_times_out_then_recovers(self):
        self.run_ui(r"""
const app=browser({abort:true,state:{job:{status:'running',progress:{percent:20,steps:[]}}}});await settle();
assert.equal(app.server.timers.length,1);app.server.hangState=true;
app.server.timers.shift()();await settle();assert.equal(app.server.timers.length,1);
app.server.timers.shift()();await settle();
assert.match(app.el('status').textContent,/server took too long.*Retrying automatically/);
assertQueryDisabled(app,true);assert.equal(app.el('workflow-progress').value,20);assert.equal(app.server.timers.length,1);
app.server.hangState=false;app.server.state.job.progress.percent=40;app.server.timers.shift()();await settle();
assert.equal(app.el('workflow-progress').value,40);assert.match(app.el('workflow-connection').textContent,/Connected/);
assert.equal(app.server.timers.length,1);
""")


class SourceMarkupTests(unittest.TestCase):
    def test_selector_is_in_composer_and_does_not_depend_on_form_navigation(self):
        class Parser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.elements = {}
                self.forms = 0

            def handle_starttag(self, tag, attrs):
                attrs = dict(attrs)
                if 'id' in attrs:
                    self.elements[attrs['id']] = (tag, attrs)
                self.forms += tag == 'form'

        html = (ROOT / 'mori/static/index.html').read_text(encoding='utf-8')
        parsed = Parser()
        parsed.feed(html)
        self.assertEqual(parsed.elements['sources'][0], 'details')
        self.assertNotIn('open', parsed.elements['sources'][1])
        source_menu = html.split('<details id="sources">', 1)[1].split('</details>', 1)[0]
        self.assertIn('<summary>Sources ', source_menu)
        self.assertIn('id="source-count"', source_menu)
        self.assertIn('id="source-picker"', source_menu)
        self.assertNotIn('id="source-error"', source_menu)
        self.assertEqual(parsed.elements['source-picker'][0], 'fieldset')
        self.assertEqual(parsed.elements['source-picker'][1]['aria-label'], 'Search sources')
        self.assertEqual(parsed.elements['source-error'][1]['role'], 'alert')
        self.assertIn('source-summary', parsed.elements['source-picker'][1]['aria-describedby'])
        self.assertLess(html.index('id="composer"'), html.index('id="source-picker"'))
        self.assertLess(html.index('id="source-picker"'), html.index('id="settings"'))
        for name in ('sources-all', 'sources-none'):
            self.assertEqual(parsed.elements[name][1]['type'], 'button')
        self.assertIn('hidden', parsed.elements['add-sources'][1])
        self.assertEqual(parsed.elements['workflow-progress'][0], 'progress')
        self.assertEqual(parsed.elements['workflow-progress'][1]['max'], '100')
        self.assertEqual(parsed.elements['workflow'][1]['aria-labelledby'], 'workflow-title')
        self.assertIn('hidden', parsed.elements['workflow'][1])
        self.assertEqual(parsed.elements['workflow-announcement'][1]['aria-live'], 'polite')
        self.assertEqual(parsed.elements['workflow-steps'][0], 'ol')
        self.assertLess(html.index('id="composer"'), html.index('id="workflow"'))
        self.assertLess(html.index('id="workflow"'), html.index('id="result"'))
        self.assertIn('prefers-reduced-motion:reduce', (ROOT / 'mori/static/styles.css').read_text(encoding='utf-8'))
        self.assertEqual(parsed.forms, 0)


if __name__ == '__main__':
    unittest.main()
