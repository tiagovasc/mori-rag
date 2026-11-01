"""Run the real browser script against a small DOM/API double, with no network."""
from html.parser import HTMLParser
from pathlib import Path
import re
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
    else if(path==='/api/runs'){if(server.runsGate)await server.runsGate.promise;if(server.runsError)throw server.runsError;value=server.runs||[];}
    else if(path.startsWith('/api/runs/')){value=server.runReplies[decodeURIComponent(path.slice('/api/runs/'.length))];assert.ok(value,'Unknown synthetic run');}
    else if(path==='/api/plan')value=server.plan;
    else if(path==='/api/live')value=server.live??{job:server.state.job,live:null};
    else if(path==='/api/query')value=server.query;
    else if(path==='/api/upload')value=server.upload;
    else throw Error('Unexpected request: '+path);
    if(typeof value==='function')value=await value();
    return {ok:!value?.error,status:value?.httpStatus||200,json:async()=>value};
  };
  class FileReader {
    readAsDataURL(){Promise.resolve(server.fileGate?.promise).then(()=>{this.result='data:text/plain;base64,e30=';this.onload();});}
  }
  const context=vm.createContext({document,fetch,FileReader,location:{search:options.search||''},URLSearchParams,setTimeout:fn=>server.timers.push(fn),setInterval:fn=>server.intervals.push(fn),...options.globals});
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
function assertQueryDisabled(app,expected){for(const id of ['retrieve','plan'])assert.equal(app.el(id).disabled,expected,id);
 // Without a configured LLM, answering is never offered; evidence handoff stays available.
 assert.equal(app.el('answer').disabled,expected||!app.eval('llmEnabled'),'answer');}
"""


@unittest.skipUnless(NODE, 'Node.js is required for frontend behavior checks')
class SourceSelectionTests(unittest.TestCase):
    def test_saved_search_phrasings_are_separate_safe_and_clear_for_legacy_runs(self):
        self.run_ui(r"""
const app=browser();await settle();
app.server.runReplies.enabled={...packet('original',['alpha']),search_phrasings:[{kind:'original',text:'original'},{kind:'hypothetical_answer',text:'<script>unsafe</script>'}]};
await app.openRun('enabled');assert.equal(app.el('search-phrasings').hidden,false);assert.equal(app.el('search-phrasings').open,false);
assert.equal(app.el('search-phrasing-list').children.length,2);
assert.equal(app.el('search-phrasing-list').children[1].children[1].textContent,'<script>unsafe</script>');
assert.match(app.el('search-phrasing-list').textContent,/search hint only/);
assert.equal(app.el('prompt').textContent,'Synthetic prompt.');assert.equal(app.el('evidence-view').children.length,0);
app.server.runReplies.legacy=packet('old');await app.openRun('legacy');
assert.equal(app.el('search-phrasings').hidden,true);assert.equal(app.el('search-phrasing-list').children.length,0);
""")

    def test_prompt_extension_opt_in_survives_polling_and_provider_changes(self):
        self.run_ui(r"""
const app=browser({state:{providers:{embedding:'demo',llm:'http',model:'model-a'}}});await settle();
assert.equal(app.el('query_expansion').checked,false);
await app.click('plan');assert.equal(queries(app)[0].body.settings.query_expansion,false);
app.el('query_expansion').checked=true;await app.el('query_expansion').emit('change');
await app.refresh();assert.equal(app.el('query_expansion').checked,true);
app.server.plan.expansion_model_calls=1;app.server.plan.maximum_query_embeddings=5;
await app.click('plan');assert.equal(queries(app)[1].body.settings.query_expansion,true);
assert.match(app.el('status').textContent,/Prompt extension adds 1 model call and up to 5 query embeddings/);
app.el('question').value='Synthetic extension inquiry';await app.click('retrieve');await settle();
assert.equal(queries(app)[2].body.settings.query_expansion,true);assert.equal(queries(app)[2].body.answer,false);
app.server.state.job={status:'running'};await app.refresh();assert.equal(app.el('query_expansion').disabled,true);
app.el('query_expansion').checked=false;await app.el('query_expansion').emit('change');assert.equal(app.el('query_expansion').checked,true);
app.server.state.job={status:'idle'};app.server.state.providers.llm='none';await app.refresh();
assert.equal(app.el('query_expansion').checked,true);assertQueryDisabled(app,true);
assert.equal(app.el('query_expansion').disabled,false);
app.el('query_expansion').checked=false;await app.el('query_expansion').emit('change');
assertQueryDisabled(app,false);assert.equal(app.el('query_expansion').disabled,true);
""")

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
assert.equal(app.el('source-summary').textContent,'2 of 2 collections selected: 4 source units, 10 passages.');
assert.equal(app.el('source-count').textContent,'10 passages selected');
assert.equal(app.el('sources-all').disabled,true);assert.equal(app.el('add-sources').hidden,false);
assert.equal(app.server.requests.filter(r=>r.path==='/api/state').length,1);
""")

    def test_model_choice_survives_refresh_and_reaches_budget_and_query(self):
        self.run_ui(r"""
const app=browser({state:{providers:{embedding:'demo',llm:'command',model:'model-a',models:['model-a','model-b']}}});await settle();
assert.equal(app.el('llm_model').value,'model-a');assert.equal(app.el('llm_model').disabled,false);
assert.deepEqual(app.el('llm_model').children.map(e=>e.value),['model-a','model-b']);
app.el('llm_model').value='model-b';await app.el('llm_model').emit('change');
const option=app.el('llm_model').children[0];await app.refresh();
assert.equal(app.el('llm_model').value,'model-b');assert.equal(app.el('llm_model').children[0],option);
await app.choose('alpha',false);await app.click('plan');
assert.equal(queries(app)[0].body.settings.llm_model,'model-b');
app.el('question').value='Synthetic model choice';await app.click('answer');await settle();
assert.equal(queries(app)[1].body.settings.llm_model,'model-b');assert.deepEqual(queries(app)[1].body.sources,['beta']);
assert.equal(app.server.state.providers.model,'model-a');
""")

    def test_model_disabled_without_provider_and_implicit_default_omits_override(self):
        self.run_ui(r"""
const app=browser();await settle();assert.equal(app.el('llm_model').disabled,true);
assert.equal(app.el('llm_model').children[0].textContent,'No answer model');
await app.click('plan');assert.equal('llm_model' in queries(app)[0].body.settings,false);
const implicit=browser({state:{providers:{embedding:'demo',llm:'http',model:'',models:[]}}});await settle();
assert.equal(implicit.el('llm_model').disabled,true);assert.equal(implicit.el('llm_model').children[0].textContent,'Provider default');
await implicit.click('plan');assert.equal('llm_model' in queries(implicit)[0].body.settings,false);
""")

    def test_unavailable_model_blocks_submission_until_user_changes_it(self):
        self.run_ui(r"""
const app=browser({state:{providers:{embedding:'demo',llm:'http',model:'model-a',models:['model-a','model-b']}}});await settle();
app.el('llm_model').value='model-b';await app.el('llm_model').emit('change');
app.server.state.providers.models=['model-a'];await app.refresh();
assertQueryDisabled(app,true);assert.equal(app.el('model-error').hidden,false);
assert.match(app.el('model-error').textContent,/model-b.*unavailable/);assert.equal(app.el('llm_model').value,'model-b');
await app.click('answer');assert.equal(queries(app).length,0);
app.el('llm_model').value='model-a';await app.el('llm_model').emit('change');assertQueryDisabled(app,false);
await app.refresh();assert.equal(app.el('model-error').hidden,true);assert.equal(app.el('llm_model').value,'model-a');
""")

    def test_model_cannot_change_during_running_request(self):
        self.run_ui(r"""
const app=browser({state:{providers:{embedding:'demo',llm:'command',model:'model-a',models:['model-a','model-b']}}});await settle();
const gate=deferred();app.server.plan=()=>gate.promise;
const pending=app.click('plan');assert.equal(app.el('llm_model').disabled,true);
app.el('llm_model').value='model-b';await app.el('llm_model').emit('change');
assert.equal(app.eval('selectedModel'),'model-a');
gate.resolve(app.server.plan);await pending;assert.equal(app.el('llm_model').disabled,false);
app.server.state.job={status:'running'};await app.refresh();assert.equal(app.el('llm_model').disabled,true);
""")

    def test_removed_last_model_can_return_to_implicit_provider_default(self):
        self.run_ui(r"""
const app=browser({state:{providers:{embedding:'demo',llm:'http',model:'',models:['model-b']}}});await settle();
app.el('llm_model').value='model-b';await app.el('llm_model').emit('change');
app.server.state.providers.models=[];await app.refresh();assertQueryDisabled(app,true);
assert.equal(app.el('llm_model').disabled,false);
assert.equal(app.el('llm_model').children[0].textContent,'Provider default');
app.el('llm_model').value='';await app.el('llm_model').emit('change');await app.refresh();
assertQueryDisabled(app,false);assert.equal(app.el('model-error').hidden,true);
await app.click('plan');assert.equal('llm_model' in queries(app)[0].body.settings,false);
""")

    def test_context_mode_preserves_counts_and_sends_budget_to_plan_and_query(self):
        self.run_ui(r"""
const app=browser();await settle();
assert.equal(app.el('evidence_mode').value,'fixed');assert.equal(app.el('context-controls').hidden,true);
app.el('final_passages').value='13';app.el('rerank_k').value='7';
app.el('evidence_mode').value='context';await app.el('evidence_mode').emit('change');
assert.equal(app.el('context-controls').hidden,false);assert.equal(app.el('context_window_tokens').value,100000);
assert.equal(app.el('final_passages-field').hidden,true);assert.equal(app.el('rerank_k').disabled,true);
assert.equal(app.el('summaries').value,5);assert.equal(app.el('full_units').value,1);
app.el('context_window_tokens').value='90000';app.el('llm_max_output_tokens').value='3000';
await app.refresh();assert.equal(app.el('context_window_tokens').value,'90000');assert.equal(app.el('evidence_mode').value,'context');
app.server.plan={...app.server.plan,evidence_mode:'context',context_window_tokens:90000,final_passage_cap:100};
await app.click('plan');assert.equal(queries(app)[0].body.settings.context_window_tokens,90000);
assert.equal(queries(app)[0].body.settings.llm_max_output_tokens,3000);assert.equal(queries(app)[0].body.settings.evidence_mode,'context');
assert.equal('final_passages' in queries(app)[0].body.settings,false);assert.equal('rerank_k' in queries(app)[0].body.settings,false);
assert.match(app.el('status').textContent,/90,000-token context budget/);assert.match(app.el('status').textContent,/100 ranked passages/);
app.el('question').value='Synthetic context inquiry';await app.click('retrieve');await settle();assert.equal(queries(app)[1].body.settings.evidence_mode,'context');
app.el('evidence_mode').value='fixed';await app.el('evidence_mode').emit('change');
assert.equal(app.el('final_passages-field').hidden,false);assert.equal(app.el('final_passages').value,'13');
assert.equal(app.el('rerank_k').value,'7');assert.equal(app.el('context-controls').hidden,true);
await app.click('plan');assert.equal(queries(app)[2].body.settings.final_passages,13);
assert.equal('context_window_tokens' in queries(app)[2].body.settings,false);
assert.equal('llm_max_output_tokens' in queries(app)[2].body.settings,false);
""")

    def test_context_budget_validation_and_busy_controls(self):
        self.run_ui(r"""
const app=browser({state:{context_budget:{overhead_reserve_tokens:2048}}});await settle();
app.el('evidence_mode').value='context';await app.el('evidence_mode').emit('change');
assert.match(app.el('budget-help').textContent,/2,048 tokens reserved/);
for(const value of ['',0,1.5,2000001]){app.el('context_window_tokens').value=String(value);await app.el('context_window_tokens').emit('input');assertQueryDisabled(app,true);}
app.el('context_window_tokens').value='6000';await app.el('context_window_tokens').emit('input');
assert.match(app.el('budget-error').textContent,/leave room/);await app.click('answer');assert.equal(queries(app).length,0);
app.el('context_window_tokens').value='100000';app.el('llm_max_output_tokens').value='-1';await app.el('llm_max_output_tokens').emit('input');assertQueryDisabled(app,true);
app.el('llm_max_output_tokens').value='4096';await app.el('llm_max_output_tokens').emit('input');assertQueryDisabled(app,false);
const gate=deferred();app.server.plan=()=>gate.promise;const pending=app.click('plan');
for(const id of ['evidence_mode','context_window_tokens','llm_max_output_tokens'])assert.equal(app.el(id).disabled,true);
gate.resolve({error:'Synthetic plan error'});await pending;assert.equal(app.el('evidence_mode').disabled,false);
app.server.state.job={status:'running'};await app.refresh();assert.equal(app.el('evidence_mode').disabled,true);
""")

    def test_saved_context_budget_is_visible_and_legacy_runs_clear_it(self):
        self.run_ui(r"""
const app=browser();await settle();app.server.runReplies.budget=packet('Budget',[alpha]);
app.server.runReplies.budget.context.context_budget={mode:'context',limit_tokens:100000,prompt_tokens:80000,output_reserve_tokens:4096,overhead_reserve_tokens:1024,remaining_tokens:14880,counting_note:'Conservative estimate; provider usage can differ.'};
await app.openRun('budget');assert.equal(app.el('context-usage').hidden,false);
assert.equal(app.el('context-progress').max,100000);assert.equal(app.el('context-progress').value,85120);
assert.match(app.el('context-usage-summary').textContent,/85,120 of 100,000 estimated tokens/);
assert.match(app.el('context-usage-detail').textContent,/remaining 14,880/);
app.server.runReplies.old=packet('Old',[alpha]);await app.openRun('old');assert.equal(app.el('context-usage').hidden,true);
app.server.runReplies.bad=packet('Bad',[alpha]);app.server.runReplies.bad.context.context_budget={mode:'context',limit_tokens:0,prompt_tokens:-1};
await app.openRun('bad');assert.equal(app.el('context-usage').hidden,true);
""")

    def test_no_selection_blocks_requests_and_bodies_use_checked_sources(self):
        self.run_ui(r"""
const app=browser({state:{providers:{embedding:'demo',llm:'http',model:'model-a'}}});await settle();
await app.click('sources-none');assertQueryDisabled(app,true);
assert.match(app.el('source-summary').textContent,/0 of 2 collections selected/);
assert.equal(app.el('source-count').textContent,'None selected');
await app.click('answer');await app.click('plan');assert.equal(queries(app).length,0);
assert.throws(()=>app.eval('querySources()'),/Choose at least one source/);
await app.choose('beta',true);assertQueryDisabled(app,false);
assert.equal(app.el('source-count').textContent,'7 passages selected');
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
assert.deepEqual(label.children.map(e=>e.tagName),['INPUT','SPAN','SPAN','SPAN']);
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
assert.match(app.el('workflow-steps').children[1].textContent,/42 \/ 500 passages scored\. 12 reused from cache/);
assert.match(app.el('workflow-elapsed').textContent,/Running for 1 min \d+ s/);
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
const app=browser({state:{providers:{embedding:'demo',llm:'http',model:'model-a'},job:{status:'complete',result:{run_id:'done'},progress}}});
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
const app=browser({state:{job:{status:'running',progress:{percent:25,started_at:'2026-10-31T10:00:00Z',steps:[]}}}});await settle();
app.server.stateError=Error('Connection dropped');app.server.timers.shift()();await settle();
assert.equal(app.el('workflow-progress').value,25);assertQueryDisabled(app,true);
assert.match(app.el('workflow-connection').textContent,/Connection interrupted/);assert.match(app.el('status').textContent,/Retrying automatically/);
assert.equal(app.server.timers.length,1);
const historyReads=()=>app.server.requests.filter(row=>row.path==='/api/runs').length,initialReads=historyReads();
app.server.stateError=null;app.server.state.job.progress.percent=30;app.server.timers.shift()();await settle();
assert.equal(app.el('workflow-progress').value,30);assert.equal(historyReads(),initialReads);
app.server.runsGate=deferred();app.server.state.job={status:'running',progress:{percent:35,started_at:'2026-10-31T10:05:00Z',steps:[]}};
app.server.timers.shift()();await settle();assert.equal(app.el('workflow-progress').value,35);assert.equal(historyReads(),initialReads+1);
assert.match(app.el('workflow-connection').textContent,/Connected/);assert.equal(app.server.timers.length,1);
app.server.runsGate.resolve();await settle();
app.server.runsError=Error('History unavailable');app.server.state.job={...app.server.state.job,status:'failed',progress:{...app.server.state.job.progress,percent:45}};
app.server.timers.shift()();await settle();assert.equal(app.el('workflow-progress').value,45);
assert.match(app.el('history-status').textContent,/History unavailable/);assert.equal(app.server.timers.length,0);
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
const app=browser({state:{providers:{embedding:'demo',llm:'http',model:'model-a'}}});await settle();app.server.query=()=>{
 app.server.state.job={status:'running',progress:{percent:40,steps:[{id:'answer',label:'Write answer',status:'running'}]}};
 return {error:'One operation is already running',httpStatus:409};
};
app.el('question').value='Synthetic conflict';await app.click('answer');await settle();assertQueryDisabled(app,true);
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
const app=browser({state:{providers:{embedding:'demo',llm:'http',model:'model-a'}}});await settle();app.server.query=()=>{
 app.server.state.job={status:'running',progress:{percent:10,steps:[{id:'embed',label:'Embed question',status:'running'}]}};
 throw Error('Response connection lost');
};
app.el('question').value='Synthetic lost response';await app.click('answer');await settle();assertQueryDisabled(app,true);
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


@unittest.skipUnless(NODE, 'Node.js is required for frontend behavior checks')
class ProductionPolishTests(unittest.TestCase):
    """Regressions for the 31 October 2026 MVP-to-production pass."""
    run_ui = SourceSelectionTests.run_ui

    def test_empty_question_is_explained_without_a_request(self):
        self.run_ui(r"""
const app=browser({state:{LLM}});await settle();
app.el('question').value='   ';await app.click('answer');await app.click('retrieve');
assert.equal(queries(app).length,0);assert.equal(app.el('workflow').hidden,true);
assert.match(app.el('status').textContent,/Type a question first/);assert.equal(app.el('status').attributes['data-tone'],'error');
assert.equal(app.el('question').attributes['aria-invalid'],'true');
await app.el('question').emit('input');assert.equal('aria-invalid' in app.el('question').attributes,false);
""".replace('LLM', "providers:{embedding:'demo',llm:'http',model:'model-a'}"))

    def test_rejected_request_shows_reason_without_failed_panel(self):
        self.run_ui(r"""
const app=browser({state:{providers:{embedding:'demo',llm:'http',model:'model-a'}}});await settle();
app.server.query={error:'Passage counts must follow retrieval_k >= jev_k',httpStatus:400};
app.el('question').value='Synthetic';await app.click('answer');await settle();
assert.equal(app.el('workflow').hidden,true);assert.match(app.el('status').textContent,/Passage counts must follow/);
assert.equal(app.el('status').attributes['data-tone'],'error');assertQueryDisabled(app,false);
""")

    def test_answering_and_reranking_require_a_configured_llm(self):
        self.run_ui(r"""
const ctrlEnter=app=>app.el('question').events.keydown[0]({key:'Enter',ctrlKey:true,preventDefault(){}});
const app=browser();await settle();
assert.equal(app.el('answer').disabled,true);assert.equal(app.el('answer-hint').hidden,false);assert.equal(app.el('retrieve').disabled,false);
assert.equal(app.el('rerank_enabled').disabled,true);assert.equal(app.el('summarize').disabled,true);
app.el('question').value='Synthetic';ctrlEnter(app);await settle();assert.equal(queries(app)[0].body.answer,false);
const configured=browser({state:{providers:{embedding:'demo',llm:'http',model:'model-a'}}});await settle();
assert.equal(configured.el('answer').disabled,false);assert.equal(configured.el('answer-hint').hidden,true);
assert.equal(configured.el('rerank_enabled').disabled,false);assert.equal(configured.el('summarize').disabled,false);
configured.el('question').value='Synthetic';ctrlEnter(configured);await settle();assert.equal(queries(configured)[0].body.answer,true);
const settings={retrieval_k:500,jev_k:100,final_passages:20,full_units:1,summaries:5,min_relevance:0,summary_min_relevance:0,summary_candidates:0,rerank_k:20,rating_neighbors:0,answer_neighbors:0,rerank_enabled:true,exclude_full_unit_passages:true};
const rerank=browser({state:{settings}});await settle();
assert.match(rerank.el('model-error').textContent,/Reranking requires a configured LLM/);assertQueryDisabled(rerank,true);
assert.equal(rerank.el('rerank_enabled').disabled,false);
rerank.el('rerank_enabled').checked=false;await rerank.el('rerank_enabled').emit('change');
assertQueryDisabled(rerank,false);assert.equal(rerank.el('rerank_enabled').disabled,true);
""")

    def test_counts_use_singular_and_plural_forms(self):
        self.run_ui(r"""
const app=browser();await settle();
assert.equal(app.el('source-options').children[0].children[1].children[1].textContent,'1 source unit');
assert.equal(app.el('source-options').children[0].children[2].textContent,'3');
assert.equal(app.el('stats').textContent,'10 passages in 4 source units');
const one=packet('One');one.context.counts={full_units:1,summaries:1,passages:2};app.server.runReplies.one=one;
await app.openRun('one');assert.deepEqual(app.el('counts').children.map(chip=>chip.textContent),['1 complete unit','1 summary','2 passages']);
app.server.plan={candidate_passages:1,summary_ratings:1,rating_attempt_ceiling:3,full_units:1,summaries:1,final_passage_cap:1,expansion_model_calls:0};
await app.click('plan');
assert.equal(app.el('status').textContent,'Up to 1 passage rating and 1 summary rating, at most 3 attempts including retries; cached decisions are reused. Final packet: 1 complete unit, 1 summary and up to 1 passage. No provider calls were made for this check.');
""")

    def test_answers_render_headings_lists_and_line_breaks_as_text(self):
        self.run_ui(r"""
const app=browser();await settle();
app.server.runReplies.listed={...packet('Listed'),answer:'## Heading\r\nIntro line one\nline two [P1]\n\n- First [U1]\n- Second **bold**\n  continued\n\n3. Third\n4. <b>Fourth</b>\n\n#### Small heading'};
await app.openRun('listed');
const view=app.el('answer-view').children;
assert.deepEqual(view.map(element=>element.tagName),['H3','P','UL','OL','H4']);
assert.equal(view[1].textContent,'Intro line one\nline two P1');
assert.equal(view[2].children.length,2);assert.equal(view[2].children[1].textContent,'Second bold continued');
assert.equal(view[3].attributes.start,'3');assert.equal(view[3].children[1].textContent,'<b>Fourth</b>');
assert.equal(view[1].children.find(child=>child.className==='citation').attributes['aria-label'],'Show evidence P1');
""")

    def test_opening_a_saved_inquiry_keeps_an_unsent_draft(self):
        self.run_ui(r"""
const app=browser();await settle();
app.server.runReplies.first=packet('First',[alpha]);await app.openRun('first');assert.equal(app.el('question').value,'First');
app.server.runReplies.second=packet('Second',[alpha]);await app.openRun('second');assert.equal(app.el('question').value,'Second');
app.el('question').value='My unsent draft';await app.el('question').emit('input');
await app.openRun('first');assert.equal(app.el('question').value,'My unsent draft');assert.equal(app.el('result-title').textContent,'First');
assert.match(app.el('status').textContent,/unsent question is still in the box/);
await app.el('question').emit('input');assert.equal(app.el('status').textContent,'');
""")

    def test_draft_survives_reload_and_storage_failures_are_harmless(self):
        self.run_ui(r"""
const saved=new Map([['mori.draft','Draft from last visit']]);
const localStorage={getItem:key=>saved.get(key)??null,setItem:(key,value)=>saved.set(key,String(value)),removeItem:key=>saved.delete(key)};
const app=browser({globals:{localStorage},state:{providers:{embedding:'demo',llm:'http',model:'model-a'},job:{status:'complete',result:{run_id:'previous'}}}});
app.server.runReplies.previous=packet('Previous',[alpha]);await settle();
assert.equal(app.el('question').value,'Draft from last visit');assert.equal(app.el('result-title').textContent,'Previous');
app.el('question').value='Draft, edited';await app.el('question').emit('input');assert.equal(saved.get('mori.draft'),'Draft, edited');
app.server.query={};await app.click('answer');await settle();assert.equal(saved.has('mori.draft'),false);
const denied=()=>{throw Error('denied');};
const broken=browser({globals:{localStorage:{getItem:denied,setItem:denied,removeItem:denied}}});await settle();
broken.el('question').value='Still works';await broken.el('question').emit('input');assert.equal(broken.el('question').value,'Still works');
""")

    def test_history_shows_status_and_current_inquiry_without_rereading_every_poll(self):
        self.run_ui(r"""
const app=browser();app.server.runs=[
 {id:'r1',question:'Answered question',created:'2026-10-31T10:00:00Z',status:'complete',answer:true},
 {id:'r2',question:'Evidence question',created:'2026-10-31T09:00:00Z',status:'complete',answer:false},
 {id:'r3',question:'Stopped question',created:'2026-10-31T08:00:00Z',status:'failed'},
 {id:'r4',question:'Unfinished question',status:'interrupted'}];
await settle();
const items=app.el('history').children.filter(child=>child.runId);assert.equal(items.length,4);
assert.equal(items[0].children[2].textContent,'Answered');assert.equal(items[1].children[2].textContent,'Evidence only');
assert.equal(items[2].children[2].children[0].textContent,'Stopped by an error');assert.equal(items[2].children[2].children[1].textContent,'Ask again');assert.equal(items[2].disabled,false);
assert.equal(items[3].children[2].children[0].textContent,'Not finished');assert.equal(items[0].title,'Answered question');
assert.deepEqual(app.el('history').children.filter(child=>!child.runId).map(child=>child.textContent).slice(-1),['Date unknown']);
// Recent lists only places to go back to; stopped inquiries stay in the library, where they can be asked again.
const recent=app.el('recent').children;assert.deepEqual(recent.filter(child=>child.runId).map(child=>child.runId),['r1','r2']);assert.equal(recent.at(-1).textContent,'All 4 inquiries');
assert.equal(app.el('filter-stopped').textContent,'2');
await items[2].emit('click');assert.equal(app.el('question').value,'Stopped question');assert.match(app.el('status').textContent,/Question restored/);
const reply=packet('Answered question',[alpha]);reply.context.run_id='r1';app.server.runReplies.r1=reply;await app.openRun('r1');
assert.equal(items[0].attributes['aria-current'],'true');assert.equal('aria-current' in items[1].attributes,false);
const busy=browser({state:{job:{status:'running',kind:'query',request:{question:'Running question'},progress:{started_at:'2026-10-31T10:00:00Z',percent:10,steps:[]}}}});
busy.server.runs=[{id:'now',question:'Running question',status:'running'},{id:'r1',question:'Answered question',status:'complete',answer:true}];await settle();
// The running inquiry appears once: first in Recent and in the library, where it leads to its progress.
const busyRecent=busy.el('recent').children;assert.equal(busyRecent.length,2);assert.equal(busyRecent[0].href,'#/progress');assert.equal(busyRecent[0].textContent,'Running question');
const runningRow=busy.el('history').children.find(child=>child.runId==='now');assert.equal(runningRow.children[2].textContent,'Running now');assert.equal(runningRow.disabled,false);
assert.equal(busy.el('filter-stopped').textContent,'0');
const reads=()=>busy.server.requests.filter(row=>row.path==='/api/runs').length;assert.equal(reads(),1);
for(let poll=0;poll<3;poll++){busy.server.timers.shift()();await settle();}assert.equal(reads(),1);
busy.server.runReplies.done=packet('done',[alpha]);busy.server.state.job={...busy.server.state.job,status:'complete',result:{run_id:'done'}};
busy.server.timers.shift()();await settle();assert.equal(reads(),2);
""")

    def test_older_server_running_folder_is_not_offered_as_stopped(self):
        self.run_ui(r"""
// A pre-redesign server lists the running inquiry's folder as interrupted; the page still shows it once, as running.
const app=browser({state:{job:{status:'running',kind:'query',request:{question:'Same question'},progress:{percent:5,steps:[]}}}});
app.server.runs=[{id:'new',question:'Same question',status:'interrupted'},{id:'old',question:'Same question',status:'interrupted'},{id:'done',question:'Done question',status:'complete',answer:true}];await settle();
const rows=app.el('history').children.filter(child=>child.runId);
assert.deepEqual(rows.map(row=>row.dataset.kind),['running','other','answer']);assert.equal(rows[0].children[2].textContent,'Running now');
assert.equal(rows[1].children[2].children[1].textContent,'Ask again');assert.equal(app.el('filter-stopped').textContent,'1');
const recent=app.el('recent').children;assert.deepEqual(recent.map(link=>link.href),['#/progress','#/inquiry/done','#/library']);assert.equal(recent[2].textContent,'All 3 inquiries');
""")

    def test_live_polling_waits_for_the_server_to_accept_the_inquiry(self):
        self.run_ui(r"""
// Clicking Answer shows progress at once; a state read that races the request must not flip back to Ask.
const gate=deferred(),app=browser({state:{features:['live'],job:{status:'idle'}}});await settle();
app.el('question').value='Live question';app.server.query=async()=>{await gate.promise;return {status:'running'};};
await app.click('retrieve');await settle();
for(const run of app.server.timers.splice(0))run();await settle();
await app.refresh();await settle();
assert.equal(app.eval('screen'),'progress');assert.equal(app.server.requests.filter(row=>row.path==='/api/live').length,0);
app.server.state.job={status:'running',kind:'query',request:{question:'Live question'},progress:{started_at:'2026-11-01T10:00:00Z',percent:1,steps:[]}};
gate.resolve();await settle();await settle();await settle();
for(const run of app.server.timers.splice(0))run();await settle();await settle();
assert.equal(app.eval('screen'),'progress');assert.ok(app.server.requests.some(row=>row.path==='/api/live'));
""")

    def test_unchanged_sources_are_not_rebuilt_on_poll(self):
        self.run_ui(r"""
const app=browser({state:{job:{status:'running',progress:{percent:5,steps:[]}}}});await settle();
const first=app.inputs()[0];app.server.timers.shift()();await settle();assert.equal(app.inputs()[0],first);
app.server.state.sources=[...app.server.state.sources,{id:'gamma',label:'Third',units:2,chunks:9}];
app.server.timers.shift()();await settle();assert.notEqual(app.inputs()[0],first);assert.equal(app.inputs().length,3);
""")

    def test_unchanged_polls_do_not_rewrite_steps_or_live_regions(self):
        self.run_ui(r"""
const app=browser({state:{job:{status:'running',progress:{started_at:'p1',percent:20,steps:[
 {id:'prepare',label:'Prepare',status:'complete'},{id:'rate_passages',label:'Rate',status:'running',completed:2,total:9}]}}}});await settle();
const firstStep=app.el('workflow-steps').children[0],summary=app.el('source-summary');let text=summary.textContent,writes=0;
Object.defineProperty(summary,'textContent',{get:()=>text,set:value=>{writes++;text=value;}});
app.server.timers.shift()();await settle();assert.equal(app.el('workflow-steps').children[0],firstStep);assert.equal(writes,0);
app.server.state.job.progress.steps[1].completed=3;app.server.timers.shift()();await settle();
assert.notEqual(app.el('workflow-steps').children[0],firstStep);assert.match(app.el('workflow-steps').children[1].textContent,/3 \/ 9 passages scored/);
assert.equal(writes,0);
""")

    def test_copy_and_download_the_handoff_prompt_and_answer(self):
        self.run_ui(r"""
const app=browser();await settle();const copied=[],blobs=[],links=[];
app.context.navigator={clipboard:{writeText:async text=>{copied.push(text);}}};
app.context.Blob=class{constructor(parts,options){this.parts=parts;this.type=options.type;}};
app.context.URL={createObjectURL:blob=>{blobs.push(blob);return 'blob:synthetic';},revokeObjectURL(){}};
const create=app.context.document.createElement;app.context.document.createElement=tag=>{const element=create(tag);if(tag==='a')element.click=()=>links.push(element.download);return element;};
app.server.runReplies.saved={...packet('Saved',[alpha]),answer:'Answer text [U1]',prompt:'Exact\r\nprompt'};await app.openRun('saved');
assert.equal(app.el('copy-answer').hidden,false);
await app.click('copy-prompt');await settle();await app.click('copy-answer');await settle();
assert.deepEqual(copied,['Exact\r\nprompt','Answer text [U1]']);assert.equal(app.el('copy-prompt').textContent,'Copied');
for(const timer of app.server.timers.splice(0))timer();assert.equal(app.el('copy-prompt').textContent,'Copy prompt');
await app.click('download-prompt');assert.deepEqual(links,['mori-Saved-prompt.txt']);assert.equal(blobs[0].parts.length,1);assert.equal(blobs[0].parts[0],'Exact\r\nprompt');
app.context.navigator=undefined;await app.click('copy-prompt');await settle();assert.match(app.el('status').textContent,/copy it manually/);
app.server.runReplies.evidence={...packet('Evidence',[alpha]),answer:'',prompt:''};await app.openRun('evidence');
assert.equal(app.el('copy-answer').hidden,true);assert.equal(app.el('copy-prompt').disabled,true);
assert.match(app.el('prompt').textContent,/No handoff prompt was saved/);
""")

    def test_settings_are_checked_in_the_words_shown_on_screen(self):
        self.run_ui(r"""
const app=browser();await settle();
const set=async(key,value)=>{app.el(key).value=value;await app.el(key).emit('input');};
await set('jev_k','5');assertQueryDisabled(app,true);assert.equal(app.el('budget-error').hidden,false);
assert.match(app.el('budget-error').textContent,/Final passages \(no reranker\) cannot be larger than Jev shortlist/);
await set('jev_k','600');assert.match(app.el('budget-error').textContent,/Jev shortlist cannot be larger than Semantic candidates/);
await set('jev_k','2.5');assert.match(app.el('budget-error').textContent,/Jev shortlist must be a whole number of at least 1/);
await set('jev_k','100');await set('min_relevance','101');assert.match(app.el('budget-error').textContent,/Minimum passage relevance must be between 0 and 100/);
await set('min_relevance','42.5');assert.equal(app.el('budget-error').hidden,true);assertQueryDisabled(app,false);
assert.throws(()=>app.eval("$('rating_neighbors').value='11';querySources()"),/Neighbors shown to Jev cannot exceed 10/);
""")

    def test_upload_explains_missing_file_and_summary_requirement(self):
        self.run_ui(r"""
const app=browser();await settle();
await app.click('upload');assert.match(app.el('status').textContent,/Choose a file to import/);
app.el('files').files=[{name:'notes.txt',size:20}];await app.click('upload');
assert.match(app.el('status').textContent,/no model is configured to write one/);
assert.equal(app.server.requests.filter(row=>row.path==='/api/upload').length,0);assert.equal(app.el('workflow').hidden,true);
const model=browser({state:{providers:{embedding:'demo',llm:'http',model:'model-a'}}});await settle();model.el('files').files=[{name:'notes.txt',size:20}];
await model.click('upload');assert.match(model.el('status').textContent,/Turn on Write missing summaries/);
model.el('summarize').checked=true;model.server.upload=()=>{model.server.state.job={status:'complete',kind:'upload',result:{added_units:2,unchanged_units:1}};return {};};
model.el('files').value='C:\\fakepath\\notes.txt';await model.click('upload');await settle();await settle();
assert.equal(model.server.requests.filter(row=>row.path==='/api/upload')[0].body.summarize,true);
assert.equal(model.el('status').textContent,'Import complete: 2 new units, 1 already in the corpus.');assert.equal(model.el('files').value,'');
""")

    def test_failed_inquiry_can_be_retried_with_the_same_request(self):
        self.run_ui(r"""
const app=browser({state:{providers:{embedding:'demo',llm:'http',model:'model-a'}}});await settle();
app.server.query=()=>{app.server.state.job={status:'failed',error:'Provider failed',progress:{started_at:'s1',percent:30,steps:[{id:'rate_passages',label:'Rate',status:'failed'}]}};return {};};
app.el('question').value='Retry me';await app.click('answer');await settle();
assert.equal(app.el('workflow').dataset.state,'failed');assert.equal(app.el('workflow-retry').hidden,false);assert.equal(app.el('workflow-details').open,true);
app.el('question').value='Edited later';
app.server.query=()=>{app.server.state.job={status:'running',progress:{started_at:'s2',percent:5,steps:[]}};return {};};
await app.click('workflow-retry');await settle();
const sent=queries(app).filter(row=>row.path==='/api/query');assert.equal(sent.length,2);
assert.deepEqual(sent[1].body,sent[0].body);assert.equal(sent[1].body.question,'Retry me');assert.equal(app.el('workflow-retry').hidden,true);
""")

    def test_finished_steps_fold_and_the_result_comes_into_view(self):
        self.run_ui(r"""
const app=browser({state:{providers:{embedding:'demo',llm:'http',model:'model-a'}}});await settle();
let scrolled=0;const focused=[];app.el('result').scrollIntoView=()=>{scrolled++;};app.el('result-title').focus=options=>focused.push(options);
app.server.query=()=>{app.server.state.job={status:'running',kind:'query',progress:{started_at:'q1',percent:10,steps:[{id:'prepare',label:'Prepare',status:'running'}]}};return {};};
app.el('question').value='Show me';await app.click('answer');await settle();assert.equal(app.el('workflow-details').open,true);
app.server.runReplies.done={...packet('Show me',[alpha]),meta:{created:'2026-10-31T10:00:00Z',llm_model:'model-a',answer:true}};
app.server.state.job={status:'complete',result:{run_id:'done'},progress:{started_at:'q1',finished_at:'2026-10-31T10:01:00Z',percent:100,steps:[{id:'prepare',label:'Prepare',status:'complete'}]}};
app.server.timers.shift()();await settle();await settle();
assert.equal(app.el('workflow-details').open,false);assert.equal(app.el('workflow-details-summary').textContent,'Workflow steps (1)');
assert.equal(scrolled,1);assert.equal(focused.length,1);assert.equal(focused[0].preventScroll,true);assert.match(app.el('result-meta').textContent,/^Answered by model-a on /);
app.el('workflow-details').open=true;await app.refresh();assert.equal(app.el('workflow-details').open,true);
app.server.runReplies.other=packet('Other',[alpha]);await app.openRun('other');assert.equal(scrolled,1);assert.equal(app.el('result-meta').hidden,true);
""")

    def test_empty_corpus_guides_the_first_import(self):
        self.run_ui(r"""
const app=browser({state:{sources:[],stats:{units:0,chunks:0}}});await settle();
assert.equal(app.el('empty-corpus').hidden,false);assert.match(app.el('empty-corpus').textContent,/Open Add sources/);assert.equal(app.el('add-sources').open,true);
assert.equal(app.el('stats').textContent,'0 passages in 0 source units');
app.el('add-sources').open=false;await app.refresh();assert.equal(app.el('add-sources').open,false);
app.server.state.sources=[alpha];await app.refresh();assert.equal(app.el('empty-corpus').hidden,true);
const closed=browser({state:{sources:[],allow_uploads:false}});await settle();
assert.match(closed.el('empty-corpus').textContent,/no sources yet/);assert.equal(closed.el('add-sources').open,undefined);
""")

    def test_question_counter_and_tab_keyboard_navigation(self):
        self.run_ui(r"""
const app=browser();await settle();
app.el('question').value='x'.repeat(3599);await app.el('question').emit('input');assert.equal(app.el('question-count').hidden,true);
app.el('question').value='x'.repeat(4000);await app.el('question').emit('input');assert.equal(app.el('question-count').hidden,false);
assert.match(app.el('question-count').textContent,/^4.?000 \/ 4.?000 characters$/);assert.equal(app.el('question-count').attributes['data-limit'],'reached');
const key=(name,value)=>app.el(name+'-tab').events.keydown[0]({key:value,preventDefault(){}});
key('answer','End');assert.equal(app.el('prompt-tab').attributes['aria-selected'],'true');
assert.equal(app.el('prompt-tab').attributes.tabindex,'0');assert.equal(app.el('answer-tab').attributes.tabindex,'-1');
key('prompt','Home');assert.equal(app.el('answer-view').hidden,false);assert.equal(app.el('prompt-view').hidden,true);
""")


class SourceMarkupTests(unittest.TestCase):
    def test_selector_is_in_composer_and_does_not_depend_on_form_navigation(self):
        class Parser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.elements = {}
                self.forms = 0
                self.radios = []

            def handle_starttag(self, tag, attrs):
                attrs = dict(attrs)
                if 'id' in attrs:
                    self.elements[attrs['id']] = (tag, attrs)
                if tag == 'input' and attrs.get('type') == 'radio':
                    self.radios.append(attrs)
                self.forms += tag == 'form'

        html = (ROOT / 'mori/static/index.html').read_text(encoding='utf-8')
        parsed = Parser()
        parsed.feed(html)
        # Collections are always visible in the composer; errors stay outside the picker for screen readers.
        self.assertEqual(parsed.elements['source-picker'][0], 'fieldset')
        self.assertEqual(parsed.elements['source-picker'][1]['aria-label'], 'Search sources')
        self.assertIn('source-summary', parsed.elements['source-picker'][1]['aria-describedby'])
        picker = html.split('<fieldset id="source-picker"', 1)[1].split('</fieldset>', 1)[0]
        for name in ('source-count', 'source-options', 'sources-all', 'sources-none', 'source-summary'):
            self.assertIn(f'id="{name}"', picker)
        self.assertNotIn('id="source-error"', picker)
        self.assertEqual(parsed.elements['source-error'][1]['role'], 'alert')
        self.assertLess(html.index('id="composer"'), html.index('id="source-picker"'))
        # Depth presets and the answer model sit in the composer, before the actions.
        self.assertEqual(parsed.elements['depth'][1]['role'], 'radiogroup')
        self.assertEqual([radio['value'] for radio in parsed.radios if radio.get('name') == 'depth'], ['quick', 'standard', 'deep'])
        self.assertIn('checked', next(radio for radio in parsed.radios if radio.get('value') == 'standard'))
        self.assertEqual(parsed.elements['llm_model'][0], 'select')
        self.assertEqual(parsed.elements['llm_model'][1]['aria-describedby'], 'model-help model-error')
        self.assertIn('<label for="llm_model">Answer model</label>', html)
        self.assertLess(html.index('id="source-picker"'), html.index('id="llm_model"'))
        self.assertLess(html.index('id="llm_model"'), html.index('id="answer"'))
        # Advanced controls live on the Settings screen.
        settings = html.split('<div id="settings" class="settings">', 1)[1].split('</section>\n\n', 1)[0]
        for name in ('query_expansion', 'rerank_enabled', 'exclude_full_unit_passages', 'evidence_mode',
                     'context_window_tokens', 'fields', 'room-facts'):
            self.assertIn(f'id="{name}"', settings)
        self.assertEqual(parsed.elements['evidence_mode'][0], 'select')
        self.assertEqual(parsed.elements['budget-error'][1]['role'], 'alert')
        self.assertEqual(parsed.elements['context-progress'][0], 'progress')
        self.assertEqual(parsed.elements['context-progress'][1]['aria-describedby'], 'context-usage-detail')
        for name in ('sources-all', 'sources-none', 'depth-reset', 'plan'):
            self.assertEqual(parsed.elements[name][1]['type'], 'button')
        self.assertIn('hidden', parsed.elements['add-sources'][1])
        self.assertEqual(parsed.elements['workflow-progress'][0], 'progress')
        self.assertEqual(parsed.elements['workflow-progress'][1]['max'], '100')
        self.assertEqual(parsed.elements['workflow'][1]['aria-labelledby'], 'workflow-title')
        self.assertEqual(parsed.elements['workflow-announcement'][1]['aria-live'], 'polite')
        self.assertEqual(parsed.elements['workflow-steps'][0], 'ol')
        self.assertEqual(parsed.elements['grid'][0], 'canvas')
        self.assertEqual(parsed.elements['grid'][1]['role'], 'img')
        # One screen shows at a time; only the question screen starts visible.
        for name in ('workflow', 'result', 'screen-library', 'screen-settings', 'reader'):
            self.assertIn('hidden', parsed.elements[name][1], name)
        self.assertNotIn('hidden', parsed.elements['screen-ask'][1])
        self.assertLess(html.index('id="composer"'), html.index('id="workflow"'))
        self.assertLess(html.index('id="workflow"'), html.index('id="result"'))
        self.assertIn('prefers-reduced-motion:reduce', (ROOT / 'mori/static/styles.css').read_text(encoding='utf-8'))
        self.assertEqual(parsed.forms, 0)

    def test_production_markup_contract(self):
        html = (ROOT / 'mori/static/index.html').read_text(encoding='utf-8')
        css = (ROOT / 'mori/static/styles.css').read_text(encoding='utf-8')
        files = html.split('id="files"', 1)[1].split('>', 1)[0]
        self.assertNotIn('multiple', files)
        self.assertIn('rel="icon" href="data:image/svg+xml,', html)
        self.assertIn('<meta name="mori-token" content="__MORI_TOKEN__">', html)
        details = html.split('<details id="workflow-details" class="steps" open>', 1)[1].split('</details>', 1)[0]
        self.assertIn('<ol id="workflow-steps"', details)
        self.assertIn('<h1 id="result-title" tabindex="-1">', html)
        for name in ('copy-answer', 'copy-prompt', 'download-prompt', 'workflow-retry', 'stop', 'ask-again',
                     'reader-close', 'reader-prev', 'reader-next', 'menu'):
            self.assertIn(f'id="{name}" type="button"', html)
        self.assertIn('aria-keyshortcuts="Control+Enter"', html)
        self.assertIn('tabindex="-1">Evidence<small id="evidence-count"></small></button>', html)
        # The strict CSP forbids inline styles and scripts; layout values are set through the CSS object model.
        self.assertNotIn(' style=', html)
        self.assertEqual(html.count('<script'), 1)
        self.assertIn('<script src="/app.js" defer></script>', html)
        # Every bundled font the stylesheet loads is served by the allowlisted route and has its licence beside it.
        from mori.web import FONT_FILES
        fonts = set(re.findall(r'url\(/fonts/([a-z0-9.-]+)\)', css))
        self.assertEqual(fonts, set(FONT_FILES))
        for name in fonts:
            self.assertTrue((ROOT / 'mori/static/fonts' / name).is_file(), name)
        for name in ('LICENSE-Literata.txt', 'LICENSE-Source-Sans-3.txt'):
            self.assertIn('SIL Open Font License', (ROOT / 'mori/static/fonts' / name).read_text(encoding='utf-8'))

if __name__ == '__main__':
    unittest.main()
