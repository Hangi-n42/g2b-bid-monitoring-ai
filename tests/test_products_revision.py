import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'submission'))
from script import Assets, read_records, make_messages, parse_response
import pytest

def test_body_codes_resolve_when_meta_is_null():
    assets=Assets(ROOT/'open/data',ROOT/'submission/model')
    rec=next(r for r in read_records(ROOT/'open/dev.jsonl.gz') if r['id']=='PPS-DEV-13')
    assert rec['meta']['세부품명번호목록'] is None
    rows=assets.products(rec)['코드_고시목록일치']
    assert {'6010989901','7215409902','8014198801'} <= set(rows)
    assert all(r['고시행']['세부품명번호']==code for code,r in rows.items())

def test_only_supported_partial_labels_are_shown():
    assets=Assets(ROOT/'open/data',ROOT/'submission/model')
    assert len(assets.bank)==56
    assert all(r['partial_labels'] and set(r['partial_labels'])==set(r['evidence']) for r in assets.bank)
    rec=read_records(ROOT/'open/dev.jsonl.gz')[0]
    messages,_,_=make_messages(rec,assets)
    assert '전체항목라벨' not in messages[1]['content']
    assert '표시되지 않은 항목을 0으로 해석하지 않는다' in messages[0]['content']

def test_named_output_order_does_not_shift_columns():
    obj={f'v{i}':{'violation':int(i==19),'evidence':int(i==19)} for i in reversed(range(1,25))}
    p,e=parse_response({'text':json.dumps(obj),'finish_reason':'stop'},1)
    assert [i+1 for i,v in enumerate(p) if v]==[19]
    assert e[18]==1
    del obj['v24']
    with pytest.raises(ValueError,match='named items'):
        parse_response({'text':json.dumps(obj),'finish_reason':'stop'},1)
