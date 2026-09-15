"""Meaningful invariants using provided records; no invented scoring labels."""
import csv
import gzip
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'submission'))
import script
from context import segment_record,select_segments
from script import Assets,make_messages,output_schema,parse_response,to_row

def response(p,e):
    return {f'v{i}':{'violation':a,'evidence':b} for i,(a,b) in enumerate(zip(p,e),1)}

@pytest.fixture(scope='module')
def records():
    with gzip.open(ROOT/'open/dev.jsonl.gz','rt',encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]

@pytest.fixture(scope='module')
def assets():
    return Assets(ROOT/'open/data',ROOT/'submission/model')

def test_every_span_is_individual_document_substring(records):
    for rec in records:
        spans=select_segments(rec,11000)
        assert sum(len(s['text']) for s in spans)<=11000
        assert [s['sid'] for s in spans]==list(range(1,len(spans)+1))
        for span in spans:
            assert 0<len(span['text'])<=360
            assert span['text']==rec['docs'][span['doc']]['text'][span['start']:span['end']]

def test_own_answer_never_retrieved(records,assets):
    for rec in records:
        selected=assets.examples(rec,'\n'.join(d['text'] for d in rec['docs']))
        assert all(e['id']!=rec['id'] for e in selected)

def test_fold_exclusion(records):
    manifest=json.loads((ROOT/'analysis/dev_folds/manifest.json').read_text(encoding='utf-8'))
    excluded=manifest['folds'][0]['excluded_retrieval_ids']
    assets=Assets(ROOT/'open/data',ROOT/'submission/model',excluded)
    assert not set(excluded)&{r['id'] for r in assets.bank}

def test_context_does_not_depend_on_other_evaluation_notices(records,assets):
    before=make_messages(records[7],assets)
    make_messages(records[0],assets)
    make_messages(records[-1],assets)
    assert before==make_messages(records[7],assets)

def test_incomplete_or_mock_responses_rejected():
    valid=json.dumps(response([0]*24,[0]*24))
    for finish in ('length','abort','missing','mock'):
        with pytest.raises(ValueError):
            parse_response({'text':valid,'finish_reason':finish},40)
    with pytest.raises(ValueError):
        parse_response({'text':'','finish_reason':'stop'},40)
    assert parse_response({'text':valid,'finish_reason':'stop'},40)==([0]*24,[0]*24)

def test_invalid_evidence_index_rejected():
    with pytest.raises(ValueError):
        parse_response({'text':json.dumps(response([0]*24,[999]*24)),'finish_reason':'stop'},5)

def test_grounded_evidence_and_absence_empty(records):
    rec=records[0]
    spans=select_segments(rec)
    # These values exercise serialization only, not predicted/claimed labels.
    p=[1]*24;e=[1]*24
    row=to_row(rec,p,e,spans)
    for i in range(1,25):
        ev=row[f'e{i}']
        if i in {10,11,16,18,20}: assert ev==''
        else: assert ev and any(ev in d['text'] for d in rec['docs'])
    assert all(not to_row(rec,[0]*24,e,spans)[f'e{i}'] for i in range(1,25))


def test_decoder_allows_only_current_notice_indices(records,assets):
    for rec in records:
        _,spans,_=make_messages(rec,assets)
        allowed=output_schema(len(spans))['properties']['v1']['properties']['evidence']['enum']
        assert allowed==[0]+[s['sid'] for s in spans]
        assert len(spans)+1 not in allowed
    assert output_schema(0)['properties']['v1']['properties']['evidence']['enum']==[0]


def test_negative_examples_cannot_loop(records,assets):
    with pytest.raises(ValueError,match='non-negative'):
        script.fit_messages(records[0],assets,script.MockRunner(),examples=-1)


@pytest.mark.parametrize('option,value',[('--examples','-1'),('--batch-size','0'),('--max-chars','0')])
def test_invalid_options_rejected_before_loading_model(monkeypatch,option,value):
    monkeypatch.setattr(sys,'argv',['script.py',option,value])
    with pytest.raises(SystemExit) as error:
        script.main()
    assert error.value.code==2


def test_vllm_receives_per_conversation_schemas(monkeypatch):
    # Capture the actual runner call without claiming GPU/model execution.
    monkeypatch.setitem(sys.modules,'vllm',SimpleNamespace(SamplingParams=SimpleNamespace))
    monkeypatch.setitem(sys.modules,'vllm.sampling_params',
                        SimpleNamespace(StructuredOutputsParams=SimpleNamespace))
    seen=[]
    def chat(messages,sampling_params,use_tqdm):
        assert len(messages)==len(sampling_params)
        seen.append([p.structured_outputs.json['properties']['v1']['properties']['evidence']['enum']
                     for p in sampling_params])
        return [SimpleNamespace(outputs=[SimpleNamespace(
            text=json.dumps(response([1]*24,[max(allowed)]*24)),finish_reason='stop')])
                for allowed in seen[-1]]
    runner=script.VLLMRunner.__new__(script.VLLMRunner)
    runner.llm=SimpleNamespace(chat=chat)
    counts=[32,5,0]
    results=runner.generate([[{'role':'user','content':'test'}]]*3,counts)
    assert seen[0]==[list(range(n+1)) for n in counts]
    for result,n in zip(results,counts):
        assert parse_response(result,n)[1]==[n]*24
    with pytest.raises(ValueError,match='own source span count'):
        runner.generate([[]],[])


@pytest.mark.parametrize('batch_failure',[False,True])
@pytest.mark.parametrize('retry_valid',[False,True])
def test_main_bounds_survive_fitting_and_retries(records,monkeypatch,tmp_path,
                                                batch_failure,retry_valid):
    selected=records[:3]
    calls=[]
    class FakeRunner:
        def __init__(self,model_dir): pass
        def count(self,messages):
            # Forces the production context-fitting path without a model.
            return sum(len(m['content']) for m in messages)
        def generate(self,messages,span_counts):
            assert len(messages)==len(span_counts)
            counts=[]
            for conversation in messages:
                import re
                ids=[int(n) for n in re.findall(r'\[S(\d+) ',conversation[1]['content'])]
                counts.append(max(ids,default=0))
            assert counts==span_counts
            calls.append((len(messages),[len(m) for m in messages],list(span_counts)))
            if batch_failure and len(messages)>1:
                raise RuntimeError('Injected batch failure')
            responses=[]
            for conversation,n in zip(messages,span_counts):
                # Invalid initial response forces the same production retry path
                # that failed in submission 87163; retry must retain its bound.
                index=n if len(conversation)>2 and retry_valid else n+1
                responses.append({'text':json.dumps(response([1]*24,[index]*24)),
                                  'finish_reason':'stop'})
            return responses
    monkeypatch.setattr(script,'VLLMRunner',FakeRunner)
    monkeypatch.setattr(script,'read_records',lambda path:selected)
    monkeypatch.setenv('PPS_MODEL_DIR',str(tmp_path))
    monkeypatch.setattr(sys,'argv',['script.py','--single-pass','--data-dir',str(ROOT/'open/data'),
        '--output',str(tmp_path),'--asset-dir',str(ROOT/'submission/model')])
    if retry_valid:
        script.main()
        with (tmp_path/'submission.csv').open(encoding='utf-8',newline='') as f:
            rows=list(csv.DictReader(f))
        assert [r['id'] for r in rows]==[r['id'] for r in selected]
        assert all(int(r[f'v{i}'])==1 for r in rows for i in range(1,25))
        audit=json.loads((tmp_path/'run_audit.json').read_text(encoding='utf-8'))
        assert audit['completed'] and all(r['normal_calls']==2 for r in audit['records'])
    else:
        with pytest.raises(ValueError,match='Evidence index outside current notice'):
            script.main()
        assert not (tmp_path/'submission.csv').exists()
    assert calls[0][0]==3
    assert any(lengths==[4] for _,lengths,_ in calls)
