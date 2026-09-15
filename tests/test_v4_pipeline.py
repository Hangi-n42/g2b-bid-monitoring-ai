"""Group routing, current-request bounds, leakage and failure atomicity."""
import csv
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'submission'))
import script


def test_groups_partition_and_decoder_rejects_other_group():
    flattened=[i for _,items,_ in script.GROUPS for i in items]
    assert sorted(flattened)==list(range(1,25))
    for _,items,_ in script.GROUPS:
        schema=script.output_schema(3,items=items)
        assert set(schema['required'])=={f'v{i}' for i in items}
        text=json.dumps({f'v{i}':{'violation':0,'evidence':0} for i in range(1,25)})
        with pytest.raises(ValueError):
            script.parse_response({'text':text,'finish_reason':'stop'},3,items=items)


def test_group_examples_do_not_leak_own_or_other_group():
    assets=script.Assets(ROOT/'open/data',ROOT/'submission/model')
    records=script.read_records(ROOT/'open/dev.jsonl.gz')
    for rec in records:
        for _,items,_ in script.GROUPS:
            examples=assets.examples(rec,'\n'.join(d['text'] for d in rec['docs']),2,items=items)
            assert len({e['id'] for e in examples})==len(examples)
            assert all(e['id']!=rec['id'] and e['fingerprint']!=script.fingerprint(rec) for e in examples)
            assert all(set(e['partial_labels'])&{f'v{i}' for i in items} for e in examples)


@pytest.mark.parametrize('batch_failure',[False,True])
@pytest.mark.parametrize('retry_valid',[False,True])
def test_group_main_routes_each_request_and_fails_closed(monkeypatch,tmp_path,batch_failure,retry_valid):
    records=script.read_records(ROOT/'open/dev.jsonl.gz')[:3]
    calls=[]
    class FakeRunner:
        def __init__(self,model_dir): pass
        def count(self,messages): return len(''.join(m['content'] for m in messages))//3
        def generate(self,messages,span_counts,items=None):
            assert items is not None
            calls.append((tuple(items),len(messages)))
            if batch_failure and len(messages)>1:
                raise RuntimeError('Injected batch failure')
            results=[]
            for message,n in zip(messages,span_counts):
                source_ids=[int(x) for x in re.findall(r'\[S(\d+) ',message[1]['content'])]
                assert max(source_ids,default=0)==n
                # Each group must retry with its own bound. Values are test data,
                # never a prediction or a legal interpretation of these records.
                index=n if len(message)>2 and retry_valid else n+1
                obj={f'v{i}':{'violation':1,'evidence':index} for i in items}
                results.append({'text':json.dumps(obj),'finish_reason':'stop'})
            return results
    monkeypatch.setattr(script,'VLLMRunner',FakeRunner)
    monkeypatch.setattr(script,'read_records',lambda path:records)
    monkeypatch.setenv('PPS_MODEL_DIR',str(tmp_path))
    monkeypatch.setattr(sys,'argv',['script.py','--data-dir',str(ROOT/'open/data'),
        '--output',str(tmp_path),'--asset-dir',str(ROOT/'submission/model')])
    if not retry_valid:
        with pytest.raises(ValueError,match='Evidence index outside current notice'):
            script.main()
        assert not (tmp_path/'submission.csv').exists()
        assert not json.loads((tmp_path/'run_audit.json').read_text())['completed']
        return
    script.main()
    audit=json.loads((tmp_path/'run_audit.json').read_text(encoding='utf-8'))
    assert audit['completed'] and audit['normal_calls']==18
    assert all(a['normal_calls']==6 and len(a['groups'])==3 for a in audit['records'])
    with (tmp_path/'submission.csv').open(encoding='utf-8',newline='') as f:
        reader=csv.DictReader(f)
        assert reader.fieldnames==script.COLUMNS
        rows=list(reader)
    assert [r['id'] for r in rows]==[r['id'] for r in records]
    for rec,row in zip(records,rows):
        for i in range(1,25):
            assert row[f'v{i}']=='1'
            ev=row[f'e{i}']
            assert ev=='' if i in script.ABSENCE else any(ev in d['text'] for d in rec['docs']) and bool(ev)
    assert {items for items,_ in calls}=={items for _,items,_ in script.GROUPS}


def test_vllm_group_schema_uses_actual_items_and_spans(monkeypatch):
    monkeypatch.setitem(sys.modules,'vllm',SimpleNamespace(SamplingParams=SimpleNamespace))
    monkeypatch.setitem(sys.modules,'vllm.sampling_params',SimpleNamespace(StructuredOutputsParams=SimpleNamespace))
    items=(9,19,20,21,22,23,24)
    def chat(messages,sampling_params,use_tqdm):
        results=[]
        for n,params in zip((2,7),sampling_params):
            props=params.structured_outputs.json['properties']
            assert set(props)=={f'v{i}' for i in items}
            assert params.max_tokens==512
            assert props['v9']['properties']['evidence']['enum']==list(range(n+1))
            obj={key:{'violation':1,'evidence':n} for key in props}
            results.append(SimpleNamespace(outputs=[SimpleNamespace(text=json.dumps(obj),finish_reason='stop')]))
        return results
    runner=script.VLLMRunner.__new__(script.VLLMRunner)
    runner.llm=SimpleNamespace(chat=chat)
    results=runner.generate([[{'role':'user','content':'x'}]]*2,[2,7],items=items)
    for result,n in zip(results,(2,7)):
        assert script.parse_response(result,n,items=items)==([1]*7,[n]*7)
