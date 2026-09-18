"""No model inference: test request boundaries and v5 production failure paths."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'submission_v5'))
spec=importlib.util.spec_from_file_location('v5_under_test',ROOT/'submission_v5/script.py')
v5=importlib.util.module_from_spec(spec)
spec.loader.exec_module(v5)


def response(n=2):
    obj={'facts':v5.mock_facts(),**{f'v{i}':{'violation':1,'evidence':n} for i in v5.B_ITEMS}}
    return obj


def test_fact_schema_and_parser_bounds():
    obj=response()
    schema=v5.output_schema(2,v5.B_ITEMS)
    assert set(schema['required'])==set(obj)
    assert not schema['additionalProperties']
    assert schema['properties']['facts']['required']==list(v5.FACT_STATES)
    assert v5.parse_response({'text':json.dumps(obj),'finish_reason':'stop'},2,items=v5.B_ITEMS)==([1]*9,[2]*9)
    for mutation in ('span','state','missing','extra','truncated'):
        bad=response()
        if mutation=='span': bad['facts']['legal_context']['source']='S3'
        if mutation=='state': bad['facts']['legal_context']['state']='invented'
        if mutation=='missing': del bad['facts']
        if mutation=='extra': bad['v1']={'violation':0,'evidence':0}
        result={'text':json.dumps(bad),'finish_reason':'length' if mutation=='truncated' else 'stop'}
        with pytest.raises(ValueError): v5.parse_response(result,2,items=v5.B_ITEMS)


def test_a_c_unchanged_and_fact_sources_are_local():
    assert 'facts' not in v5.output_schema(3,(1,2))['properties']
    assert 'facts' not in v5.output_schema(3,(9,19,20,21,22,23,24))['properties']
    assert 'S0' not in v5.fact_sources(0) and 'S1' not in v5.fact_sources(0)
    assert v5.output_limit(v5.B_ITEMS)==768
    assert v5.output_limit((1,2))==512


@pytest.mark.parametrize('batch_failure',[False,True])
@pytest.mark.parametrize('retry_valid',[False,True])
def test_v5_main_failure_and_retry(monkeypatch,tmp_path,batch_failure,retry_valid):
    records=v5.read_records(ROOT/'open/dev.jsonl.gz')[:2]
    class Runner(v5.MockRunner):
        def __init__(self,path): pass
        def generate(self,messages,span_counts,items=None):
            if batch_failure and len(messages)>1: raise RuntimeError('injected batch failure')
            results=[]
            for msg,n in zip(messages,span_counts):
                obj={f'v{i}':{'violation':1,'evidence':n} for i in items}
                if v5.uses_facts(items):
                    obj={'facts':v5.mock_facts(),**obj}
                    if len(msg)==2 or not retry_valid:
                        obj['facts']['legal_context']['source']=f'S{n+1}'
                results.append({'text':json.dumps(obj),'finish_reason':'stop'})
            return results
    monkeypatch.setattr(v5,'VLLMRunner',Runner)
    monkeypatch.setattr(v5,'read_records',lambda _: records)
    monkeypatch.setenv('PPS_MODEL_DIR',str(tmp_path))
    monkeypatch.setattr(sys,'argv',['script.py','--data-dir',str(ROOT/'open/data'),
        '--asset-dir',str(ROOT/'submission_v5/model'),'--output',str(tmp_path)])
    if not retry_valid:
        with pytest.raises(ValueError,match='Invalid fact'): v5.main()
        assert not (tmp_path/'submission.csv').exists()
    else:
        v5.main()
        from tools.validate_submission import validate
        report,_=validate(tmp_path/'submission.csv',records)
        assert not report['errors']
        audit=json.loads((tmp_path/'run_audit.json').read_text(encoding='utf-8'))
        assert audit['completed'] and audit['normal_calls']==8


def test_vllm_b_group_uses_fact_schema_and_budget(monkeypatch):
    monkeypatch.setitem(sys.modules,'vllm',SimpleNamespace(SamplingParams=SimpleNamespace))
    monkeypatch.setitem(sys.modules,'vllm.sampling_params',SimpleNamespace(StructuredOutputsParams=SimpleNamespace))
    def chat(messages,sampling_params,use_tqdm):
        outputs=[]
        for n,params in zip((1,9),sampling_params):
            assert params.max_tokens==768
            props=params.structured_outputs.json['properties']
            assert list(props)[0]=='facts'
            assert props['facts']['properties']['legal_context']['properties']['source']['enum']==v5.fact_sources(n)
            outputs.append(SimpleNamespace(outputs=[SimpleNamespace(text=json.dumps(response(n)),finish_reason='stop')]))
        return outputs
    runner=v5.VLLMRunner.__new__(v5.VLLMRunner)
    runner.llm=SimpleNamespace(chat=chat)
    for result,n in zip(runner.generate([[{}],[{}]],[1,9],v5.B_ITEMS),(1,9)):
        assert v5.parse_response(result,n,items=v5.B_ITEMS)==([1]*9,[n]*9)
